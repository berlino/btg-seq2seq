import sys
import os

import torch
import logging

from sacrebleu.metrics import BLEU


def setup_logger():
    logger = logging.getLogger("btg-seq2seq")
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)
    return logger


def setup_logger_file(logger, log_dir):
    """
    Send info to console, and detailed debug information in logfile
    (Seems not working)
    """
    logfile_path = os.path.join(log_dir, "log.txt")
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    # sh.setLevel(logging.INFO)

    fh = logging.FileHandler(logfile_path)
    fh.setFormatter(formatter)
    # fh.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(sh)  # log to console as well

    logger.info("Logging to {}".format(logfile_path))

def compute_bleu_score(hypotheses, references):
    return (BLEU().corpus_score(hypotheses=hypotheses, references=[references]).score)

class Example(object):
    """A single training/test example."""

    def __init__(
        self,
        idx,
        source,
        target,
    ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(",")) == 2
    src_filename = filename.split(",")[0]
    trg_filename = filename.split(",")[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
    return examples


def trim_transformer_input(forward_f):
    def wrapper(*args, **kwargs):
        if len(args) == 3:
            self, src_ids, src_mask = args

            max_src_len = torch.cumsum(src_mask, dim=1).max(dim=1)[1].max() + 1
            src_ids = src_ids[:, :max_src_len].contiguous()
            src_mask = src_mask[:, :max_src_len].contiguous()
            return forward_f(self, src_ids, src_mask, **kwargs)
        else:
            self, src_ids, src_mask, tgt_ids, tgt_mask = args

            # find the index of the right-most 1
            # 100100(1)0000  Method: find the right-most 1 using cumsum
            # max will return the index of SEP token 
            max_src_len = torch.cumsum(src_mask, dim=1).max(dim=1)[1].max() + 1
            src_ids = src_ids[:, :max_src_len].contiguous()
            src_mask = src_mask[:, :max_src_len].contiguous()

            assert tgt_mask is not None
            max_tgt_len = torch.cumsum(tgt_mask, dim=1).max(dim=1)[1].max() + 1
            tgt_ids = tgt_ids[:, :max_tgt_len].contiguous()
            tgt_mask = tgt_mask[:, :max_tgt_len].contiguous()
            return forward_f(
                self,
                src_ids,
                src_mask,
                tgt_ids,
                tgt_mask,
                **kwargs,
            )

    return wrapper
