import math
import attr
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_btg.models.utils import MLP, extract_span_features_with_minus

import logging

logger = logging.getLogger("btg-seq2seq")

def sample_gumbel(shape, device, eps=1e-20):
    """
    Essentially the same as _gumbel
    """
    U = torch.rand(shape).to(device)
    return -1 * torch.log(-torch.log(U + eps) + eps)

A, B = 0, 1  # non-terminal symbols
NINF = -1e9  # it seems using float("-inf") does not work


@attr.s
class SpliTree:
    start = attr.ib()
    split = attr.ib()
    end = attr.ib()
    label = attr.ib()
    log_prob = attr.ib()
    l_child = attr.ib(default=None)
    r_child = attr.ib(default=None)

    def num_segments(self):
        if self.l_child is None and self.r_child is None:
            return 2
        elif self.l_child is None:
            return 1 + self.r_child.num_segments()
        elif self.r_child is None:
            return 1 + self.l_child.num_segments()
        else:
            return self.l_child.num_segments() + self.r_child.num_segments()

    def tree_log_prob(self):
        if self.l_child is None and self.r_child is None:
            return self.log_prob
        elif self.l_child is None:
            return self.log_prob + self.r_child.tree_log_prob()
        elif self.r_child is None:
            return self.log_prob + self.l_child.tree_log_prob()
        else:
            return (
                self.l_child.tree_log_prob()
                + self.log_prob
                + self.r_child.tree_log_prob()
            )

    def obtain_reordered_segments(self, internal_nodes=False):
        assert self.label in [A, B]
        if self.l_child is None and self.r_child is None:
            if self.label == A:
                ret = [(self.start, self.split), (self.split, self.end)]
            else:
                ret = [(self.split, self.end), (self.start, self.split)]
        elif self.r_child is None:
            if self.label == A:
                ret = self.l_child.obtain_reordered_segments(internal_nodes) + [(self.split, self.end)]
            else:
                ret = [(self.split, self.end)] + self.l_child.obtain_reordered_segments(internal_nodes)
        elif self.l_child is None:
            if self.label == A:
                ret = [(self.start, self.split)] + self.r_child.obtain_reordered_segments(internal_nodes)
            else:
                ret = self.r_child.obtain_reordered_segments(internal_nodes) + [(self.start, self.split)]
        else:
            if self.label == A:
                ret = self.l_child.obtain_reordered_segments(internal_nodes) + self.r_child.obtain_reordered_segments(internal_nodes)
            else:
                ret = self.r_child.obtain_reordered_segments(internal_nodes) + self.l_child.obtain_reordered_segments(internal_nodes)
        
        if internal_nodes:
            ret.append((self.start, self.end))
            assert len(ret) == (2 * self.num_segments() - 1)
        return ret

    def obtain_orig_segments(self, internal_nodes=False):
        if self.l_child is None and self.r_child is None:
            ret = [(self.start, self.split), (self.split, self.end)]
        elif self.r_child is None:
            ret = self.l_child.obtain_orig_segments(internal_nodes) + [(self.split, self.end)]
        elif self.l_child is None:
            ret = [(self.start, self.split)] + self.r_child.obtain_orig_segments(internal_nodes)
        else:
            ret = self.l_child.obtain_orig_segments(internal_nodes) + self.r_child.obtain_orig_segments(internal_nodes)
        if internal_nodes:
            ret.append((self.start, self.end))
            assert len(ret) == (2 * self.num_segments() - 1)
        return ret
    
    def get_rule_info(self):
        parent_label = self.label
        if self.l_child is None:
            left_n = 1
        else:
            left_n = self.l_child.num_segments()

        # infer left and right label
        is_right_AB = False  # indicates whether we need to marginalize the rightmost AB
        if parent_label == A:
            if self.l_child is not None and self.r_child is not None:
                left_label = self.l_child.label
                right_label = self.r_child.label
            elif self.l_child is not None:
                left_label = self.l_child.label
                right_label = A
                is_right_AB = True
            elif self.r_child is not None:
                left_label = B
                right_label = self.r_child.label
            else:
                left_label, right_label = B, A
                is_right_AB = True
            rule_idx = [(B, A), (B, B)].index((left_label, right_label))
        else:
            if self.l_child is not None and self.r_child is not None:
                left_label = self.l_child.label
                right_label = self.r_child.label
            elif self.l_child is not None:
                left_label = self.l_child.label
                right_label = A
                is_right_AB = True
            elif self.r_child is not None:
                left_label = A
                right_label = self.r_child.label
            else:
                left_label, right_label = A, A
                is_right_AB = True
            assert parent_label == B, "invalid label"
            rule_idx = [(A, A), (A, B)].index((left_label, right_label))

        return (parent_label, left_label, right_label, left_n, rule_idx, is_right_AB)
    

def _check_max_segment_len(sent_len, max_segment_len, num_segments):
    """
    decide max_segment_len, there is an identical function for decoding as well
    """
    if num_segments * max_segment_len < sent_len:
        # a heuristic way to decide the length, the assumption is that in case of num_seg + 1 segments,
        # all segments have the same length
        new_max_segment_len = min(
            sent_len, math.ceil(sent_len * 2 / (num_segments + 1))
        )
        logger.debug(
            f"max span length is invalid! sent length: {sent_len}, max_seg_len: {max_segment_len}, updated_seg_len: {new_max_segment_len}"
        )
        return _check_max_segment_len(sent_len, new_max_segment_len, num_segments)
    else:
        return max_segment_len


class BTGRule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score_f = MLP(input_size=hidden_size, output_size=2) 

    def span2score(self, span_rep, lengths=None):
        n = span_rep.size(0) - 1  # (N + 1) * (N + 1) * bs * hidden_size, left-close, right-open

        if lengths:
            assert n == max(lengths)

        decision_scores = {}  # scores for straight or invert
        for w in range(2, n + 1):  # span length
            for i in range(0, n - w + 1):  # span start
                k = i + w  # span end

                left_spans, right_spans = [], []
                for j in range(i + 1, k):
                    left_spans.append(span_rep[i, j])
                    right_spans.append(span_rep[j, k])
                left_span_m = torch.stack(left_spans, dim=0)
                right_span_m = torch.stack(right_spans, dim=0)
                score_v = self.score_f(left_span_m, right_span_m)
                decision_scores[i, k] = score_v

        return decision_scores


class Segre:
    """
    Segre, short for (Seg)mentation and (Re)ordering module.
    If right_heaving branching, we use the following rules:
        S -> A, B
        A -> BA, BB
        B -> AA, AB
    
    Otherwise, we use the following rules:
        S -> A, B
        A -> AB, BB
        B -> AA, BA
    
    For pre-terminals, we can also use a pre-terminal C with the following rule.
        A -> CC, A -> BC, A -> CB 
        B -> CC, B -> AC, B -> CA
        C -> w

    In this implementation, we a more compact version,
        A -> w, B -> w

    Using C as a pre-terminal, we will have one-to-one corresponding to each 
    possible reordering and segmentation. In our case, the one-to-one correspondence
    is still retained at tree topology level, but not at leaf node level. But this 
    spuriousness can be easily resolved by marginalize certain trees.
    """

    def __init__(self, rule_score, lengths, max_span_len, device) -> None:
        """
        Args:
            rule_score: map a span i, j(exclusive) to a score vector of size (j - i - 1) * bs * 2
            max_span_len: this is used across all the examples in a batch for efficiency. 
            unit_nt: if True, we use only use only one non-terminal, which will only used for segmentation. Note that 
                this is different from sampling segmentations from a segre with two nonterminals, though they maybe used
                for the same purpose of segmentatijon.
        """
        self.rule_score = rule_score
        self._device = device

        self.lengths = lengths
        self.max_span_len = max_span_len

        # though it's good to cache computed pcfg scores, but knowing when 
        # to update it is tricky, e.g., num_segment to num_segments + 1
        self.beta_cache = None
        self.pcfg_cache = None
    
    def _inside(self, num_segments):
        """
        Constructing a chart in a bottom-up manner. We also store the chart and beta
        so that next time you call with num_segments + 1, you can reuse the chart and beta.
        """
        # fmt: off

        # ideally we would enforce the constraint for each instance, here we only use the max_sent_len
        max_sent_len = max(self.lengths)
        bs = len(self.lengths)
        max_span_len = _check_max_segment_len(max_sent_len, self.max_span_len, num_segments)

        # beta is an auxilary varaible: beta[n, s, e] = torch.logsumexp(chart[n, s, e], dim=0)
        beta = dict()
        # pcfg is a normalized chart for sampling
        pcfg = dict()

        # build leaf nodes with length constraint
        # assign -inf score seems to be most convenient way to handle invalid spans
        n = 1
        for s in range(max_sent_len):
            for e in range(s + 1, max_sent_len + 1):

                # incrementally construct the chart
                if self.beta_cache is not None and (n, s, e) in self.beta_cache:
                    beta[n, s, e] = self.beta_cache[n, s, e]
                    continue

                if e - s > max_span_len:
                    beta[n, s, e] = torch.empty(bs, 2).fill_(NINF).to(self._device)
                else:
                    beta[n, s, e] = torch.zeros(bs, 2).to(self._device)
                    pcfg[n, s, e] = torch.zeros(bs, 1).to(self._device)

        # build internal tree nodes
        for n in range(2, num_segments + 1):
            for w in range(2, max_sent_len + 1):  # span length
                for s in range(0, max_sent_len - w + 1):  # span start
                    e = s + w  # span end

                    # incrementally construct the score
                    if self.beta_cache is not None and (n, s, e) in self.beta_cache:
                        beta[n, s, e] = self.beta_cache[n, s, e]
                        pcfg[n, s, e] = self.pcfg_cache[n, s, e]
                        continue

                    score_list = []
                    for left_n in range(1, n):
                        right_n = n - left_n

                        for m in range(s + 1, e):
                            if (left_n, s, m) not in beta or (right_n, m, e) not in beta:
                                combined_beta = (torch.empty(bs, 2, 2).fill_(NINF).to(self._device))
                            else:
                                left_beta = beta[left_n, s, m].unsqueeze(-1)  # bs * 2 * 1
                                right_beta = beta[right_n, m, e].unsqueeze(-2)  # bs * 1 * 2
                                combined_beta = (left_beta + right_beta)  # bs * 2 * 2 [AA, AB; BA, BB]

                            # A -> BA, A -> BB
                            score4A = combined_beta[:, 1, :] + self.rule_score[s, e][m - s - 1, :, 0:1]  # bs * 2
                            score4A = score4A.transpose(0, 1)  # 2 * bs

                            # B -> AA, B -> AB
                            score4B = combined_beta[:, 0, :] + self.rule_score[s, e][m - s - 1, :, 1:2]  # bs * 2
                            score4B = score4B.transpose(0, 1)  # 2 * bs
                            _score = torch.stack([score4A, score4B], dim=-1)  # 2 * bs * 2, with first 2 being the rule index
                            score_list.append(_score)

                    # pcfg with dimensions: d1 * d2 * d3 * d4 * d5 where d1 is the number of combinations of segments 
                    # (e.g., 1-2, 2-1 for num_seg 3), d2 is the number of split points, 
                    # d3 is the rule idx, d4 is bs, d5 = 2 for A and B
                    score = torch.cat(score_list, dim=0)  # (num_comb * num_split_points * 2) * bs * 2
                    beta[n, s, e] = torch.logsumexp(score, dim=0)
                    pcfg[n, s, e] = torch.log_softmax(score, dim=0).view(n - 1, e - s - 1, 2, bs, 2)
        
        # this will create root for all n \in [2, num_segments] 
        for n in range(2, num_segments + 1):
            root_score_l = [beta[n, 0, l][idx, :] for idx, l in enumerate(self.lengths)]
            root_score = torch.stack(root_score_l, dim=0)  # bs * 2
            pcfg["root", n] = torch.log_softmax(root_score, dim=-1)   # bs * 2

        # update the cache
        self.beta_cache = beta
        self.pcfg_cache = pcfg

        # fmt: on
        return pcfg
    
    def obtain_effective_segments(self, num_segments, argmax=True):
        """
        Return all the segments (excluding the original seq) used for segre
        """
        seg_set = set()

        if self.pcfg_cache is None:
            return seg_set
        
        if argmax:
            for n in range(2, num_segments + 1):
                argmax_sample = self.map_inference(0, n)
                argmax_segs = argmax_sample.obtain_orig_segments()
                seg_set = seg_set.union(set(argmax_segs))
        else:
            # add all segs
            for seg in self.pcfg_cache:
                if len(seg) == 2:  # root node
                    continue
                
                n, s, e = seg
                if (s, e) not in seg_set:
                    seg_set.add((s, e))
        return seg_set

    def sample(self, batch_idx, num_segments, num_samples, allow_reordering=True):
        """
        Top-down sampling for a particular example (i.e., not batched)

        If allowing reordering is deactivated, only the following rules are allowed:
            S -> A, A -> BA, A -> w, B -> w
        """

        def _3d_mat_max(mat):
            d1, d2, d3 = mat.size()
            _, max_idx = mat.view(-1).max(dim=0)
            max_idx = max_idx.item()
            d3_v = max_idx % d3
            d2_v = (max_idx // d3) % d2
            d1_v = (max_idx // d3 // d2) % d1
            return d1_v, d2_v, d3_v

        def _sample_split(pcfg, start, end, num_seg, label):
            if num_seg == 1:
                return None  # leaf node

            if allow_reordering:
                log_prob_mat = pcfg[num_seg, start, end][:, :, :, batch_idx, label]
            else:
                # this enforce left_n == 1, rule [A -> B A]
                log_prob_mat = torch.log_softmax(pcfg[num_seg, start, end][0:1, :, 0:1, batch_idx, label], dim=1)

            assert end - start - 1 == log_prob_mat.size(1)
            perturbed_mat = log_prob_mat + sample_gumbel(
                log_prob_mat.size(), self._device
            )
            max_ind = _3d_mat_max(perturbed_mat)
            # original index is by probability, and need to be transferred to absolute index
            left_n, split_point, rule_idx = (
                max_ind[0] + 1,
                max_ind[1] + start + 1,
                max_ind[2],
            )
            
            if not allow_reordering:
                assert left_n == 1 and rule_idx == 0

            if label == A:
                left_label, right_label = [(B, A), (B, B)][rule_idx]
            else:
                assert label == B, "invalid label"
                left_label, right_label = [(A, A), (A, B)][rule_idx]
            
            right_n = num_seg - left_n
            # if right is 1, we will merge it to a C. Intuition, A -> BA, A -> BB will be 
            # equivalent, if the second nonterminal on the right-hand size is a leaf node.
            if right_n == 1:
                split_log_prob = log_prob_mat[max_ind[0], max_ind[1], :].logsumexp(dim=-1)
            else:
                split_log_prob = log_prob_mat[max_ind]

            tree = SpliTree(start, split_point, end, label, split_log_prob)
            tree.l_child = _sample_split(pcfg, start, split_point, left_n, left_label)
            tree.r_child = _sample_split(pcfg, split_point, end, right_n, right_label)
            return tree

        # so if it's a unseen number of segments, we need to run the inside again
        sent_len = self.lengths[batch_idx]
        if (
            self.pcfg_cache is None
            or ("root", num_segments) not in self.pcfg_cache
        ):
            pcfg = self._inside(num_segments)
        else:
            pcfg = self.pcfg_cache

        samples = []
        for _ in range(num_samples):
            try:
                root_score = pcfg["root", num_segments][batch_idx, :]  # 2
                if allow_reordering:
                    perturbed_root_score = root_score + sample_gumbel(root_score.size(), self._device)
                    root_label = perturbed_root_score.argmax().item()
                    sample = _sample_split(pcfg, 0, sent_len, num_segments, root_label)
                    sample.log_prob = sample.log_prob.clone() + root_score[root_label]  # add the log prob of the root
                else:
                    root_label = A
                    # does not need to attach root score
                    sample = _sample_split(pcfg, 0, sent_len, num_segments, root_label)

                samples.append(sample)

            except KeyError as e:
                # in some cases, invalid segre tree are sampled despite their very low prob
                logger.warning(f"Sampled an invalid segre tree and discard it.\t {str(e)}")
        return samples

    def map_inference(self, batch_idx, num_segments, allow_reordering=True):
        """
        The same algo structure as sampling, the only difference is that we
        don't add noise to the log scores
        """

        def _3d_mat_max(mat):
            d1, d2, d3 = mat.size()
            _, max_idx = mat.view(-1).max(dim=0)
            max_idx = max_idx.item()
            d3_v = max_idx % d3
            d2_v = (max_idx // d3) % d2
            d1_v = (max_idx // d3 // d2) % d1
            return d1_v, d2_v, d3_v

        def _map_split(pcfg, start, end, num_seg, label):
            if num_seg == 1:
                return None  # leaf node

            if allow_reordering:
                log_prob_mat = pcfg[num_seg, start, end][:, :, :, batch_idx, label]
            else:
                # this enforce left_n == 1, rule [A -> B A]
                log_prob_mat = torch.log_softmax(pcfg[num_seg, start, end][0:1, :, 0:1, batch_idx, label], dim=1)

            max_ind = _3d_mat_max(log_prob_mat)
            left_n, split_point, rule_idx = (
                max_ind[0] + 1,
                max_ind[1] + start + 1,
                max_ind[2],
            )

            if not allow_reordering:
                assert left_n == 1 and rule_idx == 0

            if label == A:
                left_label, right_label = [(B, A), (B, B)][rule_idx]
            else:
                assert label == B, "invalid label"
                left_label, right_label = [(A, A), (A, B)][rule_idx]

            right_n = num_seg - left_n
            # merge equivalent rules, see comments in the sample function
            if right_n == 1:
                split_log_prob = log_prob_mat[max_ind[0], max_ind[1], :].logsumexp(dim=-1)
            else:
                split_log_prob = log_prob_mat[max_ind]

            tree = SpliTree(start, split_point, end, label, split_log_prob)
            tree.l_child = _map_split(pcfg, start, split_point, left_n, left_label)
            tree.r_child = _map_split(pcfg, split_point, end, right_n, right_label)
            return tree

        sent_len = self.lengths[batch_idx]
        if (
            self.pcfg_cache is None
            or ("root", num_segments) not in self.pcfg_cache
        ):
            pcfg = self._inside(num_segments)
        else:
            pcfg = self.pcfg_cache

        if allow_reordering:
            root_label = pcfg["root", num_segments][batch_idx, :].argmax().item()
            tree = _map_split(pcfg, 0, sent_len, num_segments, root_label)
            tree.log_prob = tree.log_prob.clone() + pcfg["root", num_segments][batch_idx, root_label]
        else:
            root_label = A
            # does not need to attach root score
            tree = _map_split(pcfg, 0, sent_len, num_segments, root_label)

        return tree

    def kl(self, num_segments, other):
        """
        Similar to KL of PCFG. Intuition: aggregate the KL locally at each split point, and store it along with the KL
        from sub-segments.
        """
        # fmt: off
        bs = len(self.lengths)
        max_sent_len = max(self.lengths)

        if self.pcfg_cache is None:
            self._inside(num_segments)
        pcfg = self.pcfg_cache

        if other.pcfg_cache is None:
            other._inside(num_segments)
        other_pcfg = other.pcfg_cache

        kl = dict()

        n = 1
        for s in range(max_sent_len):
            for e in range(s + 1, max_sent_len + 1):
                kl[n, s, e] = torch.zeros(bs, 2).to(self._device)

        for n in range(2, num_segments + 1):
            for w in range(2, max_sent_len + 1):  # span length
                for s in range(0, max_sent_len - w + 1):  # span start
                    e = s + w  # span end

                    kl_A_list, kl_B_list = [], []
                    for left_n in range(1, n):
                        right_n = n - left_n

                        for m in range(s + 1, e):
                            if (left_n, s, m) not in kl or (right_n, m, e) not in kl:
                                combined_kl = (torch.zeros(bs, 2, 2).to(self._device))
                            else:
                                left_kl = kl[left_n, s, m].unsqueeze(-1)  # bs * 2 * 1
                                right_kl = kl[right_n, m, e].unsqueeze(-2)  # bs * 1 * 2 
                                combined_kl = (left_kl + right_kl)  # bs * 2 * 2, [AA, AB; BA, BB]

                            kl_A_list.append(combined_kl[:, 1].transpose(0, 1))  # A -> BA, BB; 2 * bs
                            kl_B_list.append(combined_kl[:, 0].transpose(0, 1))  # B -> AA, AB; 2 * bs

                    kl_values = []
                    for label in [A, B]:
                        kl_list = [kl_A_list, kl_B_list][label]
                        cur_kl = torch.stack(kl_list, dim=0).view(n - 1, e - s - 1, 2, bs)  # (num_comb * num_split_points * rule_num) * bs
                        cur_logits = pcfg[n, s, e][:, :, :, :, label]  # num_comb * num_split_points * 2 * bs
                        other_logits = other_pcfg[n, s, e][:, :, :, :, label]
                        kl_tmp = (cur_kl - other_logits + cur_logits) * cur_logits.exp()
                        kl_values.append(kl_tmp.view(-1, bs).sum(dim=0))
                    kl[n, s, e] = torch.stack(kl_values, dim=1)  # bs * 2
        # fmt: on

        ret_list = []
        for batch_idx, length in enumerate(self.lengths):
            cur_root_kl = kl[num_segments, 0, length][batch_idx, :]  # 2
            cur_root_logits = pcfg["root", num_segments][batch_idx, :]
            other_root_logits = other_pcfg["root", num_segments][batch_idx, :]
            kl_tmp = (cur_root_kl - other_root_logits + cur_root_logits) * cur_root_logits.exp()
            kl["root", num_segments] = kl_tmp.sum(dim=0)  # scalar
            ret_list.append(kl["root", num_segments])
        return torch.stack(ret_list, dim=0)

    def entropy(self, num_segments):
        bs = len(self.lengths)
        max_sent_len = max(self.lengths)

        if self.pcfg_cache is None:
            self._inside(num_segments)
        pcfg = self.pcfg_cache

        # same computation graph as in kl, variable names reused
        kl = dict()

        n = 1
        for s in range(max_sent_len):
            for e in range(s + 1, max_sent_len + 1):
                kl[n, s, e] = torch.zeros(bs, 2).to(self._device)

        for n in range(2, num_segments + 1):
            for w in range(2, max_sent_len + 1):  # span length
                for s in range(0, max_sent_len - w + 1):  # span start
                    e = s + w  # span end

                    kl_A_list, kl_B_list = [], []
                    for left_n in range(1, n):
                        right_n = n - left_n

                        for m in range(s + 1, e):
                            if (left_n, s, m) not in kl or (right_n, m, e) not in kl:
                                combined_kl = (torch.zeros(bs, 2, 2).to(self._device))
                            else:
                                left_kl = kl[left_n, s, m].unsqueeze(-1)  # bs * 2 * 1
                                right_kl = kl[right_n, m, e].unsqueeze(-2)  # bs * 1 * 2 
                                combined_kl = (left_kl + right_kl)  # bs * 2 * 2, [AA, AB; BA, BB]

                            kl_A_list.append(combined_kl[:, 1].transpose(0, 1))  # A -> BA, BB; 2 * bs
                            kl_B_list.append(combined_kl[:, 0].transpose(0, 1))  # B -> AA, AB; 2 * bs

                    kl_values = []
                    for label in [A, B]:
                        kl_list = [kl_A_list, kl_B_list][label]
                        cur_kl = torch.stack(kl_list, dim=0).view(n - 1, e - s - 1, 2, bs)  # (num_comb * num_split_points * rule_num) * bs
                        cur_logits = pcfg[n, s, e][:, :, :, :, label]  # num_comb * num_split_points * 2 * bs
                        kl_tmp = (cur_kl - cur_logits) * cur_logits.exp()
                        kl_values.append(kl_tmp.view(-1, bs).sum(dim=0))
                    kl[n, s, e] = torch.stack(kl_values, dim=1)  # bs * 2
        # fmt: on

        ret_list = []
        for batch_idx, length in enumerate(self.lengths):
            cur_root_kl = kl[num_segments, 0, length][batch_idx, :]  # 2
            cur_root_logits = pcfg["root", num_segments][batch_idx, :]
            kl_tmp = (cur_root_kl - cur_root_logits) * cur_root_logits.exp()
            kl["root", num_segments] = kl_tmp.sum(dim=0)  # scalar
            ret_list.append(kl["root", num_segments])
        return torch.stack(ret_list, dim=0)


class Seg(Segre):
    """
    Probablistic Tree-based Segmentation Model. It's a simplified version of Segre without the
    reordering functionality.
    """

    def _inside(self, num_segments):
        """
        Constructing a chart in a bottom-up manner. We also store the chart and beta
        so that next time you call with num_segments + 1, you can reuse the chart and beta.
        """
        # fmt: off

        # ideally we would enforce the constraint for each instance, here we only use the max_sent_len
        max_sent_len = max(self.lengths)
        bs = len(self.lengths)
        max_span_len = _check_max_segment_len(max_sent_len, self.max_span_len, num_segments)

        # beta is an auxilary varaible: beta[n, s, e] = torch.logsumexp(chart[n, s, e], dim=0)
        beta = dict()
        # pcfg is a normalized chart for sampling
        pcfg = dict()

        # build leaf nodes with length constraint
        # assign -inf score seems to be most convenient way to handle invalid spans
        n = 1
        for s in range(max_sent_len):
            for e in range(s + 1, max_sent_len + 1):

                # incrementally construct the chart
                if self.beta_cache is not None and (n, s, e) in self.beta_cache:
                    beta[n, s, e] = self.beta_cache[n, s, e]
                    continue

                if e - s > max_span_len:
                    beta[n, s, e] = torch.empty(bs, 1).fill_(NINF).to(self._device)
                else:
                    beta[n, s, e] = torch.zeros(bs, 1).to(self._device)
                    pcfg[n, s, e] = torch.zeros(bs, 1).to(self._device)

        # build internal tree nodes
        for n in range(2, num_segments + 1):
            for w in range(2, max_sent_len + 1):  # span length
                for s in range(0, max_sent_len - w + 1):  # span start
                    e = s + w  # span end

                    # incrementally construct the score
                    if self.beta_cache is not None and (n, s, e) in self.beta_cache:
                        beta[n, s, e] = self.beta_cache[n, s, e]
                        pcfg[n, s, e] = self.pcfg_cache[n, s, e]
                        continue

                    score_list = []
                    for left_n in range(1, n):
                        right_n = n - left_n

                        for m in range(s + 1, e):
                            if (left_n, s, m) not in beta or (right_n, m, e) not in beta:
                                combined_beta = (torch.empty(bs, 1).fill_(NINF).to(self._device))
                            else:
                                left_beta = beta[left_n, s, m]  # bs * 1
                                right_beta = beta[right_n, m, e]  # bs * 1
                                combined_beta = (left_beta + right_beta)  # bs * 1

                            score = combined_beta + self.rule_score[s, e][m - s - 1, :, 0:1]  # use A dimension, bs * 1
                            score = score.transpose(0, 1)  # 1 * bs
                            score_list.append(score)

                    # pcfg with dimensions: d1 * d2 * d3 where d1 is the number of combinations of segments 
                    # (e.g., 1-2, 2-1 for num_seg 3), d2 is the number of split points, d3 is bs
                    score_t = torch.cat(score_list, dim=0)  # (num_comb * num_split_points) * bs
                    beta[n, s, e] = torch.logsumexp(score_t, dim=0)[:, None]  # bs * 1
                    pcfg[n, s, e] = torch.log_softmax(score_t, dim=0).view(n - 1, e - s - 1, bs)
        
        # update the cache
        self.beta_cache = beta
        self.pcfg_cache = pcfg

        # fmt: on
        return pcfg

    def cond_sample(self, batch_idx, num_segments, num_samples, segre_sample):
        """
        Args:
            segre_sample: a src segre tree whose topology will be used for normalization
        """

        def _sample_split(pcfg, start, end, num_seg, ref_tree):
            if num_seg == 1:
                return None  # leaf node

            parent_label, left_label, right_label, left_n, rule_idx, is_right_AB = ref_tree.get_rule_info()
            right_n = num_seg - left_n
            if parent_label == A:
                cur_left_n = left_n
            else:
                cur_left_n = right_n
            log_prob_mat = torch.log_softmax(pcfg[num_seg, start, end][cur_left_n - 1, :, batch_idx], dim=0)

            assert end - start - 1 == log_prob_mat.size(0)
            perturbed_mat = log_prob_mat + sample_gumbel(log_prob_mat.size(), self._device)
            max_ind = perturbed_mat.argmax(dim=0)
            split_point = max_ind.item() + start + 1
            split_log_prob = log_prob_mat[max_ind]

            tree = SpliTree(start, split_point, end, A, split_log_prob)  # always use A as the label
            if parent_label == A:
                tree.l_child = _sample_split(pcfg, start, split_point, left_n, ref_tree.l_child)
                tree.r_child = _sample_split(pcfg, split_point, end, right_n, ref_tree.r_child)
            else:
                tree.l_child = _sample_split(pcfg, start, split_point, right_n, ref_tree.r_child)
                tree.r_child = _sample_split(pcfg, split_point, end, left_n, ref_tree.l_child)
            return tree

        assert num_segments == segre_sample.num_segments()
        sent_len = self.lengths[batch_idx]
        if self.pcfg_cache is None or (num_segments, 0, sent_len) not in self.pcfg_cache:
            pcfg = self._inside(num_segments)
        else:
            pcfg = self.pcfg_cache

        samples = []
        for _ in range(num_samples):
            try:
                sample = _sample_split(pcfg, 0, sent_len, num_segments, segre_sample)
                samples.append(sample)

            except KeyError as e:
                # in some cases, invalid segre tree are sampled despite their very low prob
                logger.warning(f"Sampled an invalid segre tree and discard it.\t {str(e)}")
        return samples

    def cond_map_inference(self, batch_idx, num_segments, segre_sample):
        def _map_split(pcfg, start, end, num_seg, ref_tree):
            if num_seg == 1:
                return None  # leaf node

            parent_label, left_label, right_label, left_n, rule_idx, is_right_AB = ref_tree.get_rule_info()
            right_n = num_seg - left_n
            if parent_label == A:
                cur_left_n = left_n
            else:
                cur_left_n = right_n
            log_prob_mat = torch.log_softmax(pcfg[num_seg, start, end][cur_left_n - 1, :, batch_idx], dim=0)

            assert end - start - 1 == log_prob_mat.size(0)
            max_ind = log_prob_mat.argmax(dim=0)
            split_point = max_ind.item() + start + 1
            split_log_prob = log_prob_mat[max_ind]

            tree = SpliTree(start, split_point, end, A, split_log_prob)  # always use A as the label
            if parent_label == A:
                tree.l_child = _map_split(pcfg, start, split_point, left_n, ref_tree.l_child)
                tree.r_child = _map_split(pcfg, split_point, end, right_n, ref_tree.r_child)
            else:
                tree.l_child = _map_split(pcfg, start, split_point, right_n, ref_tree.r_child)
                tree.r_child = _map_split(pcfg, split_point, end, left_n, ref_tree.l_child)
            return tree

        sent_len = self.lengths[batch_idx]
        assert num_segments == segre_sample.num_segments()
        if self.pcfg_cache is None or (num_segments, 0, sent_len) not in self.pcfg_cache:
            pcfg = self._inside(num_segments)
        else:
            pcfg = self.pcfg_cache

        tree = _map_split(pcfg, 0, sent_len, num_segments, segre_sample)
        return tree

    def cond_entropy(self, batch_idx, num_segments, segre_sample):
        def _recur_compute(pcfg, start, end, num_seg, ref_tree):
            if num_seg == 1:
                return 0  # leaf node

            parent_label, left_label, right_label, left_n, rule_idx, is_right_AB = ref_tree.get_rule_info()
            right_n = num_seg - left_n
            log_prob_mat = torch.log_softmax(pcfg[num_seg, start, end][left_n - 1, :, batch_idx], dim=0)

            entropy_list = []
            for split_point in range(start + 1, end):
                split_log_prob = log_prob_mat[split_point - start - 1]

                try:
                    left_entropy = _recur_compute(pcfg, start, split_point, left_n, ref_tree.l_child)
                    right_entropy = _recur_compute(pcfg, split_point, end, right_n, ref_tree.r_child)
                    cur_entropy = (left_entropy + right_entropy - split_log_prob) * split_log_prob.exp()
                except KeyError:
                    # some splits are invalid, e.g., 2 segments in span (1, 2)
                    cur_entropy = 0
                entropy_list.append(cur_entropy)

            return sum(entropy_list)

        sent_len = self.lengths[batch_idx]
        assert num_segments == segre_sample.num_segments()
        if self.pcfg_cache is None or (num_segments, 0, sent_len) not in self.pcfg_cache:
            pcfg = self._inside(num_segments)
        else:
            pcfg = self.pcfg_cache

        entropy = _recur_compute(pcfg, 0, sent_len, num_segments, segre_sample)
        return entropy

if __name__ == "__main__":
    BS, N, H = 4, 10, 32
    device = "cuda:0"

    # span_scores = torch.randn(B, N, N, 2).to(device)
    # lengths = torch.LongTensor([6, 10, 8, 9]).to(device)
    # segre_crf = Segre_CRF()
    # segre_crf.sum(span_scores, num_segments=3, max_segment_len=4, lengths=lengths)

    lengths = [4, 10, 8, 9]

    span_hidden = torch.randn(BS, N + 2, H).to(device)  # with CLS and SEP
    span_rep = extract_span_features_with_minus(span_hidden)
    tree_rule = BTGRule(hidden_size=H).to(device)
    span_scores = tree_rule.span2score(span_rep, lengths)
    segre = Segre(span_scores, lengths, max_span_len=4, device=device)
    pcfg = segre._inside(num_segments=3)
    seg = Seg(span_scores, lengths, max_span_len=4, device=device)

    for bs_idx in range(BS):
        # num_segments = lengths[bs_idx]
        num_segments = 3
        samples = segre.sample(batch_idx=bs_idx, num_segments=num_segments, num_samples=3, allow_reordering=True)
        for sample in samples:
            print("Sample:", sample.obtain_reordered_segments(), sample.tree_log_prob().exp().item())

            # when num_segment is sent_length, conditional sample should be exactly the same with prob 1
            cond_sample = seg.cond_sample(batch_idx=bs_idx, num_segments=num_segments, num_samples=1, segre_sample=sample)[0]
            print("Conditional Sample", cond_sample.obtain_reordered_segments(), cond_sample.tree_log_prob().exp().item())
            print("Segmentations from conditional sample", cond_sample.obtain_orig_segments())
            cond_entropy = seg.cond_entropy(batch_idx=bs_idx, num_segments=num_segments, segre_sample=sample)
            print("Conditional entropy,", cond_entropy.item())
        
        argmax_sample = segre.map_inference(batch_idx=bs_idx, num_segments=num_segments)
        print("Argmax:", argmax_sample.obtain_reordered_segments(), argmax_sample.tree_log_prob().exp().item())
        cond_argmax = seg.cond_map_inference(batch_idx=bs_idx, num_segments=num_segments, segre_sample=argmax_sample)
        print("Conditional Argmax:", cond_argmax.obtain_reordered_segments(), cond_argmax.tree_log_prob().exp().item())
        print()

    # to check if it sums up to 1
    samples = segre.sample(batch_idx=0, num_segments=3, num_samples=200, allow_reordering=True)
    sample_set = set()
    prob_sum = 0
    for sample in samples:
        s_segs = tuple(sample.obtain_reordered_segments())
        if s_segs not in sample_set:
            sample_set.add(s_segs)
            prob_sum += sample.tree_log_prob().exp().item()
    print(f"prob_sum: {prob_sum}, sample_set: {len(sample_set)}")

    # test entropy and KL
    other_span_hidden = torch.randn(BS, N + 2, H).to(device)  # with CLS and SEP
    other_span_rep = extract_span_features_with_minus(other_span_hidden)
    other_btgrule = BTGRule(hidden_size=H).to(device)
    other_span_scores = other_btgrule.span2score(other_span_rep, lengths)
    other_segre = Segre(other_span_scores, lengths, max_span_len=4, device=device)
    other_segre._inside(num_segments=3)
    kl = segre.kl(num_segments=3, other=other_segre)
    print(f"KL: {kl}")

    self_kl = segre.kl(num_segments=3, other=segre)
    print(f"self KL: {self_kl}")

    entropy = segre.entropy(num_segments=3)
    print(f"Entropy: {entropy}")
