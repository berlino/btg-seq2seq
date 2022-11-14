import math
import itertools
import collections
import logging

import torch
import torch.nn as nn
import numpy as np

from neural_btg.structs.segre import Segre, Seg, BTGRule, _check_max_segment_len
from neural_btg.models.utils import extract_span_features_with_minus, gen_truncated_geometric
from neural_btg.utils import trim_transformer_input

from dataclasses import dataclass
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.file_utils import ModelOutput

logger = logging.getLogger("btg-seq2seq")

MAX_LENGTH = 64
USE_SRC_SEG_FM = True
CLIP_REWARD = True


class BTGSegWithSrcTgt:
    """
    Encoding schema for seg-to-seg translation:
        [CLS, PAD, src_seg, PAD, SEP, PAD] -> [SEG-CLS, tgt_seg, SEG-SEP, PAD, PAD]
    or 
        [SEG-CLS, PAD, src_seg, PAD, SEG-SEP, PAD] -> [SEG-CLS, tgt_seg, SEG-SEP, PAD, PAD]
    """
    def __init__(
        self,
        src_ids,
        src_mask,
        tgt_ids,
        tgt_mask,
        num_segments,
        max_src_seg_len,
        max_tgt_seg_len,
        seg_sos_id,
        seg_eos_id,
        pad_id,
    ) -> None:
        self.src_ids = src_ids
        self.src_mask = src_mask
        self.tgt_ids = tgt_ids
        self.tgt_mask = tgt_mask
        self.num_segments = num_segments
        self.max_src_seg_len = max_src_seg_len
        self.max_tgt_seg_len = max_tgt_seg_len

        self.seg_sos_id = seg_sos_id
        self.seg_eos_id = seg_eos_id
        self.pad_id = pad_id

        self.bs = src_ids.size(0)
        self.max_src_len = src_ids.size(1) - 2  # exclude CLS and SEP
        self.max_tgt_len = tgt_ids.size(1) - 2  # exclude CLS and SEP
        src_lengths_v = self.src_mask.sum(dim=1) - 2
        self.src_lengths = src_lengths_v.tolist()
        tgt_lengths_v = self.tgt_mask.sum(dim=1) - 2
        self.tgt_lengths = tgt_lengths_v.tolist()
        assert self.max_src_len == max(self.src_lengths)
        assert self.max_tgt_len == max(self.tgt_lengths)

        self.id2src_segre = collections.defaultdict(list)
        self.id2tgt_segre = collections.defaultdict(list)

    def add_src_segre(self, idx, segre_sample):
        self.id2src_segre[idx].append(segre_sample)

    def add_tgt_segre(self, idx, segre_sample):
        self.id2tgt_segre[idx].append(segre_sample)

    def to_transformer_inputs(self):
        seg_src_ids_l, seg_src_masks_l, seg_tgt_ids_l, seg_tgt_masks_l = [], [], [], []
        for ins_idx in range(self.bs):

            ins_src_len = self.src_lengths[ins_idx]
            ins_tgt_len = self.tgt_lengths[ins_idx]

            assert len(self.id2src_segre[ins_idx]) == len(self.id2tgt_segre[ins_idx])
            for src_segre_sample, tgt_segre_sample in zip(self.id2src_segre[ins_idx], self.id2tgt_segre[ins_idx]):
                src_segments = src_segre_sample.obtain_reordered_segments(internal_nodes=False)
                tgt_segments = tgt_segre_sample.obtain_orig_segments(internal_nodes=False)

                assert len(src_segments) == len(tgt_segments)
                assert len(src_segments) == self.num_segments

                # mask need to consider CLS and SEP token
                for (src_s, src_e), (tgt_s, tgt_e) in zip(src_segments, tgt_segments):
                    src_mask = (
                        [1]
                        + [0] * src_s
                        + (src_e - src_s) * [1]
                        + [0] * (ins_src_len - src_e)
                        + [1]
                        + (self.max_src_len - ins_src_len) * [0]
                    )
                    src_mask = torch.tensor(src_mask, dtype=torch.long)
                    seg_src_masks_l.append(src_mask)
                    _src_ids = self.src_ids[ins_idx].clone()
                    if USE_SRC_SEG_FM:
                        _src_ids[0] = self.seg_sos_id
                        assert src_mask[ins_src_len + 1] == 1
                        _src_ids[ins_src_len + 1] = self.seg_eos_id
                    seg_src_ids_l.append(_src_ids)

                    tgt_mask = (
                        [1]
                        + (tgt_e - tgt_s) * [1]
                        + [1]
                        + [0] * tgt_s
                        + [0] * (ins_tgt_len - tgt_e)
                        + (self.max_tgt_len - ins_tgt_len) * [0]
                    )
                    tgt_mask = torch.tensor(tgt_mask, dtype=torch.long)
                    seg_tgt_masks_l.append(tgt_mask)

                    _tgt_ids = torch.zeros_like(self.tgt_ids[ins_idx]).fill_(self.pad_id)
                    _tgt_ids[0] = self.seg_sos_id
                    _tgt_ids[1 : tgt_e - tgt_s + 1] = self.tgt_ids[ins_idx][tgt_s + 1 : tgt_e + 1]
                    _tgt_ids[tgt_e - tgt_s + 1] = self.seg_eos_id
                    seg_tgt_ids_l.append(_tgt_ids)

        seg_src_ids = torch.stack(seg_src_ids_l, dim=0)
        seg_src_mask = torch.stack(seg_src_masks_l, dim=0).to(seg_src_ids.device)
        seg_tgt_ids = torch.stack(seg_tgt_ids_l, dim=0)
        seg_tgt_mask = torch.stack(seg_tgt_masks_l, dim=0).to(seg_tgt_ids.device)

        avg_src_samples = (sum(len(self.id2src_segre[i]) for i in range(self.bs)) / self.bs)
        logger.debug(f"average {round(avg_src_samples, 2)} src samples per instance")

        avg_tgt_samples = (sum(len(self.id2tgt_segre[i]) for i in range(self.bs)) / self.bs)
        logger.debug(f"average {round(avg_tgt_samples, 2)} tgt samples per instance")

        return seg_src_ids, seg_src_mask, seg_tgt_ids, seg_tgt_mask

    @staticmethod
    def build_src_seg_id_and_mask(ins_src_ids, seg_sos_id, seg_eos_id, src_seg_s, src_seg_e, src_len):
        """
        Args:
            ins_src_ids: 1-d token indices of input
        """
        ins_src_seg_ids = ins_src_ids.clone()  # contain CLS and SEP, might also contain PAD
        if USE_SRC_SEG_FM:
            ins_src_seg_ids[0] = seg_sos_id
            ins_src_seg_ids[src_len + 1] = seg_eos_id

        ins_src_seg_mask = (
            [1]
            + [0] * src_seg_s 
            + (src_seg_e - src_seg_s) * [1]
            + [0] * (src_len - src_seg_e)
            + [1]
        )

        batch_src_len = ins_src_ids.size(0) - 2
        if batch_src_len > src_len:
            ins_src_seg_mask += (batch_src_len - src_len) * [0]
        return ins_src_seg_ids, ins_src_seg_mask

    def _obtain_tgt_neg_logprob(self, ins_idx, segre_offset, seg_loss_v, pair_num_cs):
        """
        Args:
            seg_loss_mat: all seg2seg loss matrices
            pair_num_cs: pair_num_cs[ins_idx][seg_idx] is the index of seg_idx in seg_loss_v
        """
        ins_tgt_len = self.tgt_lengths[ins_idx]

        src_segre_sample, tgt_segre_sample = self.id2src_segre[ins_idx][segre_offset], self.id2tgt_segre[ins_idx][segre_offset]
        single_pair_loss = []
        for seg_idx, (src_segment, tgt_segment) in enumerate(zip(
            src_segre_sample.obtain_reordered_segments(internal_nodes=False),
            tgt_segre_sample.obtain_orig_segments(internal_nodes=False)),
        ):
            tgt_s, tgt_e = tgt_segment
            assert tgt_e <= ins_tgt_len
            offset = pair_num_cs[ins_idx] + segre_offset * self.num_segments + seg_idx
            single_pair_loss.append(seg_loss_v[offset])

        return - sum(single_pair_loss)

    def aggregate_loss_elbo(self, seg_loss_v):
        """
        Retrieve tree-level loss and return an EBLO objective (KL is added to the loss later)
        """
        pair_num_l = [len(self.id2src_segre[i]) * self.num_segments for i in range(self.bs)]
        pair_num_cs = [0] + np.cumsum(pair_num_l).tolist()
        assert sum(pair_num_l) == seg_loss_v.size(0)

        batch_loss_l = []
        for ins_idx in range(self.bs):
            if len(self.id2src_segre[ins_idx]) < 2:
                continue
                
            with torch.no_grad():
                baseline_segre_offset = len(self.id2src_segre[ins_idx]) - 1
                baseline_src_segre_sample = self.id2src_segre[ins_idx][baseline_segre_offset]
                baseline_tgt_segre_sample = self.id2tgt_segre[ins_idx][baseline_segre_offset]
                baseline_logprob = self._obtain_tgt_neg_logprob(ins_idx, baseline_segre_offset, seg_loss_v, pair_num_cs)

            for segre_offset in range(len(self.id2src_segre[ins_idx]) - 1):
                src_segre_sample, tgt_segre_sample = self.id2src_segre[ins_idx][segre_offset], self.id2tgt_segre[ins_idx][segre_offset]
                src_seg_reorder_logprob = src_segre_sample.tree_log_prob()
                tgt_seg_logprob = tgt_segre_sample.tree_log_prob()
                tgt_logprob = self._obtain_tgt_neg_logprob(ins_idx, segre_offset, seg_loss_v, pair_num_cs)
                
                btg_reward = (tgt_logprob - baseline_logprob).detach().item()
                btg_log_prob = src_seg_reorder_logprob + tgt_seg_logprob
                entropy_reward = (tgt_segre_sample.cond_entropy - baseline_tgt_segre_sample.cond_entropy).detach().item()

                if CLIP_REWARD:
                    btg_reward = max(min(1, btg_reward), -1)
                    entropy_reward = max(min(1, entropy_reward), -1)
                logger.debug(f"instance {ins_idx} got reward {btg_reward} from {segre_offset}th segre with tgt logprob {tgt_logprob.item()}, btg logprob {btg_log_prob.item()}")
                logger.debug(f"instance {ins_idx} got entropy reward: {entropy_reward}")

                elbo_obj = btg_reward * btg_log_prob + tgt_logprob + entropy_reward * src_seg_reorder_logprob + tgt_segre_sample.cond_entropy

                elbo_loss = -elbo_obj
                batch_loss_l.append(elbo_loss)

        elbo_loss = sum(batch_loss_l) / len(batch_loss_l)
        return elbo_loss


class BTGSeg2Seg(torch.nn.Module):
    BTG_class = BTGSegWithSrcTgt

    def __init__(self, seq2seq, config, sos_id, eos_id, seg_sos_id, seg_eos_id, pad_id, beam_size, use_posterior=False):
        """
        Args:
            seq2seq could be a BART or T5 model from huggingface 
        """
        super().__init__()
        self.seq2seq = seq2seq
        self.device = config.device

        self.tree_rule = BTGRule(self.seq2seq.get_encoder().config.hidden_size)
        self.leaf_rule = BTGRule(self.seq2seq.get_encoder().config.hidden_size)

        # config for segre
        self.num_segre_samples = config.num_segre_samples
        self.monotone_align = config.monotone_align

        self.geo_p = config.geo_p
        self.use_posterior = use_posterior
        if self.use_posterior:
            logger.debug(f"Using posterior segre with ELBO objective with {self.num_segre_samples} samples")
        else:
            assert self.num_segre_samples > 1, "VIMCO needs more than one sample"
            logger.debug(f"Using prior segre with VIMCO objective with {self.num_segre_samples} samples")

        logger.debug("Using shared syntax and semantic encoder")
        self.syntax_encoder = self.seq2seq.get_encoder()

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.seg_sos_id = seg_sos_id
        self.seg_eos_id = seg_eos_id
        self.pad_id = pad_id
        self.beam_size = beam_size

        # initalize parser parameters
        for p in itertools.chain(self.tree_rule.parameters(), self.leaf_rule.parameters()):
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @trim_transformer_input 
    def seq2seq_loss(self, src_ids, src_mask, tgt_ids, tgt_mask):
        loss_v = self._seq2seq_forward(src_ids, src_mask, tgt_ids, tgt_mask)
        return loss_v.mean()

    @trim_transformer_input
    def btg_loss(
        self,
        src_ids,
        src_mask,
        tgt_ids,
        tgt_mask,
        num_segments=None,
        max_src_seg_len=None,
        max_tgt_seg_len=None,
        max_src_len=None,
        max_tgt_len=None,
    ):
        """
        Args:
            num_segments: the maximal number of segments. If None, use max value possible

            Seg2seg is reduced to seq2seq under following conditions
                1) num_segments is 1,
                3) examples are too short for btg decoding
            Otherwise, use btg training.
        """
        assert self.training

        # handle corner cases of short examples
        src_lengths = (torch.sum(src_mask, dim=1) - 2).tolist()
        min_src_len = min(src_lengths)
        tgt_lengths = (torch.sum(tgt_mask, dim=1) - 2).tolist()
        min_tgt_len = min(tgt_lengths)
        min_len = min(min_src_len, min_tgt_len)
        logger.debug(f"pre-specified max_num_segments: {num_segments}")
        if num_segments is None:
            num_segments = min_len
        else:
            num_segments = min(min_len, num_segments)
        logger.debug(f"adjusted max_num_segments: {num_segments}")

        seq_loss = self._seq2seq_forward(src_ids, src_mask, tgt_ids, tgt_mask).mean()

        # long sents are discarded for efficiency
        seg_max_src_len, seg_max_tgt_len = -1, -1
        seg_src_id_l, seg_src_mask_l, seg_tgt_id_l, seg_tgt_mask_l = [], [], [], []
        for ex_idx in range(len(src_lengths)):
            ex_src_len, ex_tgt_len = src_lengths[ex_idx], tgt_lengths[ex_idx]
            if ex_src_len <= max_src_len and ex_tgt_len <= max_tgt_len:
                seg_max_src_len = max(seg_max_src_len, ex_src_len)
                seg_max_tgt_len = max(seg_max_tgt_len, ex_tgt_len)
                seg_src_id_l.append(src_ids[ex_idx])
                seg_src_mask_l.append(src_mask[ex_idx])
                seg_tgt_id_l.append(tgt_ids[ex_idx])
                seg_tgt_mask_l.append(tgt_mask[ex_idx])

        # no need to do btg training
        if num_segments == 1 or seg_max_src_len == -1 or seg_max_tgt_len == -1:
            return seq_loss

        src_ids = torch.stack(seg_src_id_l, dim=0)[:, :seg_max_src_len + 2]
        src_mask = torch.stack(seg_src_mask_l, dim=0)[:, :seg_max_src_len + 2]
        tgt_ids = torch.stack(seg_tgt_id_l, dim=0)[:, :seg_max_tgt_len + 2]
        tgt_mask = torch.stack(seg_tgt_mask_l, dim=0)[:, :seg_max_tgt_len + 2]

        # sample N
        k_dist = gen_truncated_geometric(n=num_segments, p=self.geo_p, remove_first=True)
        sampled_num_seg = np.random.choice(range(1, num_segments + 1), p=k_dist, size=1)[0]
        logger.debug(f"Sampled number of segments: {sampled_num_seg}")

        # compute loss
        btg_loss = self.btg_train(
            src_ids,
            src_mask,
            tgt_ids,
            tgt_mask,
            sampled_num_seg,
            max_src_seg_len,
            max_tgt_seg_len,
        )
        return self.geo_p * seq_loss + (1 - self.geo_p) * btg_loss
    
    @trim_transformer_input
    def seq2seq_decode(self, src_ids, src_mask, seg_gen=False):
        if seg_gen:
            bos_token_id = self.seg_sos_id
            eos_token_id = self.seg_eos_id
            bad_words_ids = [[self.eos_id] for _ in range(src_ids.size(0))]
        else:
            bos_token_id = self.sos_id
            eos_token_id = self.eos_id
            bad_words_ids = [[self.seg_eos_id] for _ in range(src_ids.size(0))]

        pred_ids = self.seq2seq.generate(
            inputs=src_ids,
            attention_mask=src_mask,
            bos_token_id=bos_token_id,
            decoder_start_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=self.pad_id,
            bad_words_ids=bad_words_ids,
            num_beams=self.beam_size,
            max_length=MAX_LENGTH,
        )
        return pred_ids
    
    @trim_transformer_input
    def btg_decode(self, src_ids, src_mask, num_segments=None, max_src_seg_len=8, use_argmax=False):
        src_lengths = (torch.sum(src_mask, dim=1) - 2).tolist()
        min_src_len = min(src_lengths)
        if min_src_len == 1 or num_segments == 1:
            return self.seq2seq_decode(src_ids, src_mask), None

        return self.btg_marginalize_decode(src_ids, src_mask, num_segments, max_src_seg_len, use_argmax=use_argmax)

    def _encode_src_seg(self, src_seg_ids, src_seg_mask):
        """
        Representations of src segments are contextualized and they're identical to those from sequence-level representations 
        """
        bs = src_seg_ids.size(0)
        cum_src_seg_mask = src_seg_mask.cumsum(dim=1)
        src_seq_mask = cum_src_seg_mask < src_seg_mask.sum(dim=1)[:, None]
        last_one_idx = src_seq_mask.sum(dim=1)
        src_seq_mask[torch.arange(bs), last_one_idx] = 1
        return self.seq2seq.get_encoder()(input_ids=src_seg_ids, attention_mask=src_seq_mask)["last_hidden_state"]
    
    def _decode_tgt_rep(self, tgt_ids, tgt_mask, encoder_output, src_mask):
        """
        Note that encoder_output is generated using a potentially different attention mask than src_mask
        """
        dec_output = self.seq2seq(attention_mask=src_mask, decoder_input_ids=tgt_ids, encoder_outputs=[encoder_output,], output_hidden_states=True, return_dict=True)
        return dec_output["decoder_hidden_states"][-1]

    def obtain_seq2seq_logits(self, src_ids, src_mask, tgt_ids, tgt_mask):
        """
        Return: unnormalized logits, which are expected to be paired with crossEntropyloss.
        """
        encoder_output = self._encode_src_seg(src_ids, src_mask)
        dec_rep = self._decode_tgt_rep(tgt_ids, tgt_mask, encoder_output, src_mask)

        if self.seq2seq._get_name().startswith("Bart") or self.seq2seq._get_name().startswith("MBart"):
            lm_logits = self.seq2seq.lm_head(dec_rep) + self.seq2seq.final_logits_bias
        else:
            assert self.seq2seq._get_name().startswith("T5") or self.seq2seq._get_name().startswith("MT5")
            dec_rep = dec_rep * (self.seq2seq.model_dim**-0.5)
            lm_logits = self.seq2seq.lm_head(dec_rep)
        return lm_logits

    def _obtain_span_representation(self, src_ids, src_mask):
        token_rep = self.syntax_encoder(input_ids=src_ids, attention_mask=src_mask)["last_hidden_state"]
        span_rep = extract_span_features_with_minus(token_rep)
        return span_rep

    def _obtain_conditioned_span_representation(self, src_ids, src_mask, tgt_ids, tgt_mask):
        src_enc = self._encode_src_seg(src_ids, src_mask)
        cond_tgt_rep = self._decode_tgt_rep(tgt_ids, tgt_mask, src_enc, src_mask)
        cond_tgt_span_rep = extract_span_features_with_minus(cond_tgt_rep)

        tgt_enc = self._encode_src_seg(tgt_ids, tgt_mask)
        cond_src_rep = self._decode_tgt_rep(src_ids, src_mask, tgt_enc, tgt_mask)
        cond_src_span_rep = extract_span_features_with_minus(cond_src_rep)
        return cond_src_span_rep, cond_tgt_span_rep

    def _seq2seq_forward(self, src_ids, src_mask, tgt_ids, tgt_mask):
        lm_logits = self.obtain_seq2seq_logits(src_ids, src_mask, tgt_ids, tgt_mask)
        lm_log_probs = torch.log_softmax(lm_logits, dim=-1)

        shift_log_probs = lm_log_probs[:, :-1]
        shift_labels = tgt_ids[:, 1:]
        shift_mask = tgt_mask[:, 1:]

        label_log_probs = torch.gather(shift_log_probs, dim=2, index=shift_labels[:, :, None])
        label_log_probs = label_log_probs.squeeze(2)
        log_likelihood_mat = label_log_probs * shift_mask
        log_likelihood_v = log_likelihood_mat.sum(dim=1)
        return -log_likelihood_v
    
    def _seq2seq_inference(self, src_seg_ids, src_seg_mask, normalize_score=True, seg_gen=False):
        """
        Args:
            normalize_score: whether to use the normalized score returned by huggingface
            seg_gen: which tag set of bos and eos to use, depending on whether generating segments or not
        """
        bs = src_seg_ids.shape[0]
        max_src_len = src_seg_ids.shape[1]
        encoder_output = self._encode_src_seg(src_seg_ids, src_seg_mask)

        if seg_gen:
            bos_token_id = self.seg_sos_id
            eos_token_id = self.seg_eos_id
            bad_words_ids = [[self.eos_id] for _ in range(src_seg_ids.size(0))]
        else:
            bos_token_id = self.sos_id
            eos_token_id = self.eos_id
            bad_words_ids = [[self.seg_eos_id] for _ in range(src_seg_ids.size(0))]

        @dataclass
        class EncoderState(ModelOutput):
            last_hidden_state: torch.Tensor

        self.eval()
        with torch.no_grad():
            seq_bs = 128
            num_batch = math.ceil(bs / seq_bs)
            bart_seq_l = []
            bart_seq_score_l = []
            for k in range(num_batch):
                model_kwargs = {
                    "encoder_outputs": EncoderState(encoder_output[k*seq_bs:(k+1)*seq_bs]),
                    "attention_mask": src_seg_mask[k*seq_bs:(k+1)*seq_bs],
                } 

                num_example = src_seg_ids[k*seq_bs:(k+1)*seq_bs].shape[0]
                bart_ret = self.seq2seq.generate(
                    num_beams=self.beam_size,
                    decoder_start_token_id=bos_token_id,
                    bos_token_id=bos_token_id,
                    bad_words_ids=bad_words_ids,
                    eos_token_id=eos_token_id,
                    pad_token_id=self.pad_id,
                    forced_eos_token_id=eos_token_id,
                    return_dict_in_generate=True,
                    renormalize_logits=True,
                    output_scores=True,
                    max_length=MAX_LENGTH,
                    num_return_sequences=self.beam_size,
                    **model_kwargs)

                beam_seq = bart_ret.sequences.reshape(num_example, self.beam_size, -1)
                beam_seq = torch.nn.functional.pad(beam_seq, (0, MAX_LENGTH - beam_seq.size(2)), value=self.pad_id)
                bart_seq_l.append(beam_seq)

                if normalize_score:
                    # use hg scores which are normalized
                    bart_seq_score_l.append(bart_ret.sequences_scores.reshape(num_example, self.beam_size))
                else: 
                    # note that bart_ret.sequence_scores is normalized and we need to recover seg-level log-porb
                    trs_bs = self.seq2seq.compute_transition_beam_scores(
                            sequences=bart_ret.sequences,
                            scores=bart_ret.scores,
                            beam_indices=bart_ret.beam_indices
                    )
                    trs_bs = trs_bs.sum(dim=1).reshape(num_example, self.beam_size)
                    bart_seq_score_l.append(trs_bs)

        bart_seq = torch.cat(bart_seq_l, dim=0)
        bart_seq_score = torch.cat(bart_seq_score_l, dim=0)
        return bart_seq, bart_seq_score

    def btg_train(
        self,
        src_ids,
        src_mask,
        tgt_ids,
        tgt_mask,
        num_segments,
        max_src_seg_len,
        max_tgt_seg_len,
    ):
        assert self.use_posterior
        bs, max_src_len = src_ids.size()
        max_src_len = max_src_len - 2  # remove CLS and SEP

        src_lengths = torch.sum(src_mask, dim=1) - 2
        src_lengths_l = src_lengths.tolist()
        tgt_lengths = torch.sum(tgt_mask, dim=1) - 2
        tgt_lengths_l = tgt_lengths.tolist()

        # create parsers
        span_rep, tgt_span_rep = self._obtain_conditioned_span_representation(src_ids, src_mask, tgt_ids, tgt_mask)
        prior_span_rep = self._obtain_span_representation(src_ids, src_mask)
        assert max_src_len == span_rep.size(0) - 1  # span_rep is left-closed, right-open

        span_scores = self.tree_rule.span2score(span_rep)
        segre = Segre(span_scores, src_lengths_l, max_src_seg_len, self.device)

        # add KL and entroy
        prior_span_scores = self.tree_rule.span2score(prior_span_rep)
        prior_segre = Segre(prior_span_scores, src_lengths_l, max_src_seg_len, self.device)
        prior_segre._inside(num_segments)  # update the internal pcfg
        kl = segre.kl(num_segments, prior_segre)
        kl_loss = kl.mean()
        logger.debug("kl: {}".format(kl.mean().item()))

        tgt_span_score = self.leaf_rule.span2score(tgt_span_rep)
        tgt_seg = Seg(tgt_span_score, tgt_lengths_l, max_tgt_seg_len, self.device)

        # create seq2seq inputs
        src_seg_ids_l, src_seg_masks_l, tgt_seg_ids_l, tgt_seg_masks_l = [], [], [], []
        btg_seg = self.BTG_class(
            src_ids,
            src_mask,
            tgt_ids,
            tgt_mask,
            num_segments,
            max_src_seg_len,
            max_tgt_seg_len,
            seg_sos_id=self.seg_sos_id,
            seg_eos_id=self.seg_eos_id,
            pad_id=self.pad_id,
        )

        for ins_idx in range(bs):
            # short examples
            if btg_seg.src_lengths[ins_idx] < num_segments or btg_seg.tgt_lengths[ins_idx] < num_segments:
                continue

            # sample src segre
            segre_samples = segre.sample(
                ins_idx,
                num_segments,
                self.num_segre_samples,
                not self.monotone_align,
            )
            for segre_sample in segre_samples:
                btg_seg.add_src_segre(ins_idx, segre_sample)
                logger.debug(f"instance {ins_idx}: sampled reordered src segments {segre_sample.obtain_reordered_segments()}, prob {segre_sample.tree_log_prob().exp().item()}")

                tgt_segre_sample = tgt_seg.cond_sample(ins_idx, num_segments, 1, segre_sample,)[0]

                btg_seg.add_tgt_segre(ins_idx, tgt_segre_sample)
                logger.debug(f"instance {ins_idx}: sampled tgt segments {tgt_segre_sample.obtain_orig_segments()}, prob {tgt_segre_sample.tree_log_prob().exp().item()}")

                cond_entropy = tgt_seg.cond_entropy(ins_idx, num_segments, segre_sample)
                tgt_segre_sample.cond_entropy = cond_entropy
                logger.debug(f"instance {ins_idx} has conditional entropy: {cond_entropy.item()}")

            # baseline
            argmax_src_segre_sample = segre.map_inference(ins_idx, num_segments, not self.monotone_align)
            btg_seg.add_src_segre(ins_idx, argmax_src_segre_sample)
            logger.debug(f"instance {ins_idx}: argmax src segments {argmax_src_segre_sample.obtain_reordered_segments()}, prob {argmax_src_segre_sample.tree_log_prob().exp().item()}")

            argmax_tgt_segre_sample = tgt_seg.cond_map_inference(ins_idx, num_segments, argmax_src_segre_sample)
            btg_seg.add_tgt_segre(ins_idx, argmax_tgt_segre_sample)
            cond_entropy = tgt_seg.cond_entropy(ins_idx, num_segments, argmax_src_segre_sample)
            argmax_tgt_segre_sample.cond_entropy = cond_entropy
            logger.debug(f"instance {ins_idx}: argmax tgt segments {argmax_tgt_segre_sample.obtain_orig_segments()}, prob {argmax_tgt_segre_sample.tree_log_prob().exp().item()}")
                    
        (
            src_seg_ids,
            src_seg_masks,
            tgt_seg_ids,
            tgt_seg_masks,
        ) = btg_seg.to_transformer_inputs()

        src_seg_ids_l.append(src_seg_ids)
        src_seg_masks_l.append(src_seg_masks)
        tgt_seg_ids_l.append(tgt_seg_ids)
        tgt_seg_masks_l.append(tgt_seg_masks)

        src_seg_ids_t = torch.cat(src_seg_ids_l, dim=0)
        src_seg_masks_t = torch.cat(src_seg_masks_l, dim=0)
        tgt_seg_ids_t = torch.cat(tgt_seg_ids_l, dim=0)
        tgt_seg_masks_t = torch.cat(tgt_seg_masks_l, dim=0)

        seg_loss_mat = self._seq2seq_forward(
            src_seg_ids_t,
            src_seg_masks_t,
            tgt_seg_ids_t,
            tgt_seg_masks_t,
        )

        # aggregate loss
        seg2seg_loss = btg_seg.aggregate_loss_elbo(seg_loss_mat)
        
        final_seg_loss = seg2seg_loss + kl_loss
        return final_seg_loss

    def btg_marginalize_decode(
        self,
        src_ids,
        src_mask,
        num_segments=-1,
        max_src_seg_len=8,
        constraints=None,
        use_argmax=False,
    ):
        """
        Configurations:
            1. num_segments: if -1, use src_len as the number fo segments (i.e., full marginalization)
            2. use_argmax will affect how many source segments are considered 
            3. constraints: a dictionary that maps from src segments (in the form of start and end index) to target segments 
                (in the form of a sequence of token ids).
        """
        EPSILON = 1e-6  # for numerical stability
        CONSTRAIN_VAL = 1  # 1 or 1e2

        def merge_candidate(left_candidates, right_candidates):
            # [5, 0, 1] + [0, 1, 2] = [5, 0, 1, 2]
            left_pointer = len(left_candidates) - 1
            right_pointer = 1
            ret_l = left_pointer
            match_flag = False
            while left_pointer >= 0 and right_pointer <= len(right_candidates):
                if left_candidates[left_pointer:] == right_candidates[:right_pointer]:
                    match_flag = True
                    ret_l = left_pointer
                left_pointer -= 1
                right_pointer += 1

            if not match_flag:
                return left_candidates[: ret_l + 1] + right_candidates
            else:
                return left_candidates[: ret_l] + right_candidates
                
        def merge_score(cur_l, candidate_seq, candidate_score):
            # logsumexp to merge the scores of the same candidate sequence
            flag = False
            for idx, (cand, cur_score) in enumerate(cur_l):
                if cand == candidate_seq:
                    try:
                        cur_l[idx] = (candidate_seq, math.log(math.exp(cur_score) + math.exp(candidate_score) + EPSILON))
                    except OverflowError:
                        logger.warn("OverflowError in merge_score")
                        cur_l[idx] = (candidate_seq, max(cur_score, candidate_score))
                    flag = True
                    break

            if not flag:
                cur_l.append((candidate_seq, candidate_score))
        
        # only supports batch size 1 for now
        bs, _ = src_ids.size()
        assert bs == 1
        bs_idx, A, B = 0, 0, 1  # use to indicate batch index, straight and inversion
        src_lengths = torch.sum(src_mask, dim=1) - 2
        src_lengths_l = src_lengths.tolist()
        src_len = src_lengths_l[0]

        # create the segre parser
        max_src_seg_len = min(max_src_seg_len, src_len)
        span_rep = self._obtain_span_representation(src_ids, src_mask)
        assert src_len == span_rep.size(0) - 1
        span_scores = self.tree_rule.span2score(span_rep, src_lengths_l)

        # choose the number of segments
        if num_segments is None or num_segments > min(src_lengths_l):
            logger.debug(f"Adapting to {min(src_lengths_l)} segments")
            num_segments = min(src_lengths_l)
        logger.debug(f"BTG CKY decoding with {num_segments} segments")


        if num_segments == -1:
            num_segments = src_len
        
        segre = Segre(span_scores, src_lengths_l, max_src_seg_len, device=self.device)
        segre._inside(num_segments)

        # generate src segments
        src_seg_ids_l = []
        src_seg_masks_l = []
        src_seg2idx = {}
        for seg_idx, (src_seg_s, src_seg_e) in enumerate(segre.obtain_effective_segments(num_segments, argmax=use_argmax)):
            if src_seg_s == 0 and src_seg_e == src_len:
                # seq2seq decoding will be done later
                continue
            else:
                seg_src_ids, seg_src_mask = self.BTG_class.build_src_seg_id_and_mask(src_ids[bs_idx], self.seg_sos_id, self.seg_eos_id, src_seg_s, src_seg_e, src_len)
                seg_src_mask = torch.LongTensor(seg_src_mask).to(src_ids.device)
            src_seg_ids_l.append(seg_src_ids)
            src_seg_masks_l.append(seg_src_mask)

            src_seg2idx[(src_seg_s, src_seg_e)] = seg_idx

        src_seg_ids = torch.stack(src_seg_ids_l, dim=0)
        src_seg_masks = torch.stack(src_seg_masks_l, dim=0)

        # seg2seg translation, beam is a list of BEAM objects
        seg_beams, seg_beams_scores = self._seq2seq_inference(src_seg_ids, src_seg_masks, normalize_score=True, seg_gen=True)

        # seq2seq translation
        seq_beams, seq_beams_scores = self._seq2seq_inference(src_ids, src_mask, normalize_score=True, seg_gen=False)
        ## merge
        src_seg2idx[(0, src_len)] = seg_beams.size(0)
        seg_beams = torch.cat([seg_beams, seq_beams], dim=0)
        seg_beams_scores = torch.cat([seg_beams_scores, seq_beams_scores], dim=0)


        chart = collections.defaultdict(list)

        # leaf translation
        n = 1
        for s in range(src_len):
            for e in range(s + 1, src_len + 1):
                if constraints and (s, e) in constraints:
                    logger.debug(f"Triggered constraint in span {s}-{e}")
                    chart[n, s, e, A] = [(constraints[(s, e)], CONSTRAIN_VAL)]
                    chart[n, s, e, B] = [(constraints[(s, e)], CONSTRAIN_VAL)]
                    continue
                
                if (s, e) not in src_seg2idx:
                    continue

                seg_idx = src_seg2idx[(s, e)]
                raw_candidates = seg_beams[seg_idx] 
                candidates_scores = seg_beams_scores[seg_idx]
                candidate_set = set()
                for raw_candidate, candidate_score in zip(raw_candidates, candidates_scores):
                    candidate = [c.item() for c in raw_candidate]
                    seg_eos_idx = candidate.index(self.seg_eos_id) if self.seg_eos_id in candidate else len(candidate)
                    seq_eos_idx = candidate.index(self.eos_id) if self.eos_id in candidate else len(candidate)
                    eos_idx = min(seg_eos_idx, seq_eos_idx)
                    candidate = candidate[0: eos_idx]
                    candidate_len = len(candidate)

                    special_tokens = [self.sos_id, self.eos_id, self.seg_sos_id, self.seg_eos_id, self.pad_id]
                    candidate = [c for c in candidate if c not in special_tokens]

                    if len(candidate) == 0:  # discourage empty translation
                        logger.info("catch empty translation!")
                        continue
                    if tuple(candidate) in candidate_set:
                        logger.info("catch duplicate translation!")
                        continue

                    candidate_set.add(tuple(candidate))
                    # logger.debug(f"leaf node {s}-{e}: {detok_f(candidate)}")
                    candidate_score = candidate_score.item()
                    norm_candidate_score = candidate_score
                    chart[n, s, e, A].append((candidate, norm_candidate_score))
                    chart[n, s, e, B].append((candidate, norm_candidate_score))

        for n in range(2, num_segments + 1):
            for w in range(1, src_len + 1):
                for s in range(0, src_len - w + 1):
                    e = s + w

                    # does not need to consider this path
                    if constraints and (s, e) in constraints:
                        continue

                    for left_n in range(1, n):
                        right_n = n - left_n
                    
                        for m in range(s + 1, e):
                            # BA -> A
                            BA_A_score = segre.pcfg_cache[n, s, e][left_n - 1, m - s - 1, 0, bs_idx, A].item()
                            # BB -> A
                            BB_A_score = segre.pcfg_cache[n, s, e][left_n - 1, m - s - 1, 1, bs_idx, A].item()
                            # AA -> B
                            AA_B_score = segre.pcfg_cache[n, s, e][left_n - 1, m - s - 1, 0, bs_idx, B].item()
                            # AB -> B
                            AB_B_score = segre.pcfg_cache[n, s, e][left_n - 1, m - s - 1, 1, bs_idx, B].item()

                            rules = [[B, A, A], [B, B, A], [A, A, B], [A, B, B]]
                            rule_scores = [BA_A_score, BB_A_score, AA_B_score, AB_B_score]

                            for rule, rule_score in zip(rules, rule_scores):
                                left_nt, right_nt, parent_nt = rule

                                left_candidates = chart[left_n, s, m, left_nt]
                                right_candidates = chart[right_n, m, e, right_nt]
                                for left_candidate, left_candidate_score in left_candidates:
                                    for right_candidate, right_candidate_score in right_candidates:
                                        if parent_nt == A:
                                            straight_candidate = merge_candidate(left_candidate, right_candidate)
                                            straight_candidate_score = left_candidate_score + right_candidate_score + rule_score
                                            merge_score(chart[n, s, e, A], straight_candidate, straight_candidate_score)
                                        else:
                                            assert parent_nt == B
                                            invert_candidate = merge_candidate(right_candidate, left_candidate)
                                            invert_candidate_score = left_candidate_score + right_candidate_score + rule_score
                                            merge_score(chart[n, s, e, B], invert_candidate, invert_candidate_score)
                            
                            sorted_l = sorted(chart[n, s, e, A], key=lambda x: x[1], reverse=True)
                            chart[n, s, e, A] = list(sorted_l)[:self.beam_size]

                            sorted_l = sorted(chart[n, s, e, B], key=lambda x: x[1], reverse=True)
                            chart[n, s, e, B] = list(sorted_l)[:self.beam_size]

        final_candidates = []
        final_candidate_scores = []

        k_dist = gen_truncated_geometric(n=num_segments, p=self.geo_p, remove_first=False)
        for n in range(1, num_segments + 1):
            p_n = k_dist[n - 1]

            if n == 1:
                candidates = chart[n, 0, src_len, A]
            else:
                A_candidates = [(cand, cand_score + segre.pcfg_cache["root", n][bs_idx, A]) for cand, cand_score in chart[n, 0, src_len, A]]
                B_candidates = [(cand, cand_score + segre.pcfg_cache["root", n][bs_idx, B]) for cand, cand_score in chart[n, 0, src_len, B]]
                candidates = A_candidates + B_candidates
            
            # merge candidate across num_segments
            for candidate, candidate_score in candidates:
                _score = math.log(p_n * math.exp(candidate_score) + EPSILON)
                if candidate in final_candidates:
                    idx = final_candidates.index(candidate)
                    final_candidate_scores[idx] = math.log(math.exp(final_candidate_scores[idx]) + math.exp(_score))
                else:
                    final_candidates.append(candidate)
                    final_candidate_scores.append(_score)

        sorted_final_l = sorted(zip(final_candidates, final_candidate_scores), key=lambda x: x[1], reverse=True)

        best_candidate, best_candidate_score = list(sorted_final_l)[0]
        return [best_candidate], None
