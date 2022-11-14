import os
import math
import random
import logging
import argparse
import numpy as np

from tokenizers import Tokenizer
import tokenizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.models import BPE
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

import torch
import wandb

from neural_btg.utils import setup_logger_file, setup_logger, read_examples

logger = setup_logger()

class InputFeatures(object):
    def __init__(self, example_id, source_ids, target_ids, source_mask, target_mask,):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    def tokenize(input_str):
        _encoded = tokenizer.encode(input_str)
        return _encoded.tokens, _encoded.ids

    num_src_tokens, num_tgt_tokens = 0, 0
    coverage = 0
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens, source_ids = tokenize(example.source)
        source_tokens, source_ids = (source_tokens[: max_source_length - 2], source_ids[: max_source_length - 2])
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = [tokenizer.cls_token_id] + source_ids + [tokenizer.sep_token_id]
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        num_src_tokens += len(source_tokens)

        # target
        if stage == "test":
            target_tokens, target_ids = tokenize("[PAD]")
        else:
            target_tokens, target_ids = tokenize(example.target)
        target_tokens, target_ids = (target_tokens[: max_target_length - 2], target_ids[: max_source_length - 2])
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = [tokenizer.cls_token_id] + target_ids + [tokenizer.sep_token_id]
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        num_tgt_tokens += len(target_tokens)

        # if example_index < 1:
        if False:
            if stage == "train":
                logger.info("*** Sample Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format( [x.replace("\u0120", "_") for x in source_tokens]))
                logger.info("source_ids: {}".format(" ".join(map(str, source_ids))))
                logger.info("source_mask: {}".format(" ".join(map(str, source_mask))))

                logger.info("target_tokens: {}".format( [x.replace("\u0120", "_") for x in target_tokens]))
                logger.info("target_ids: {}".format(" ".join(map(str, target_ids))))
                logger.info("target_mask: {}".format(" ".join(map(str, target_mask))))

        features.append(InputFeatures(example_index, source_ids, target_ids, source_mask, target_mask,))

    logger.debug(f"average src tokens {num_src_tokens / len(examples)}, average tgt tokens {num_tgt_tokens / len(examples)}")
    return features

def train_ws_bpe_tokenizer(files, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[ "[CLS]", "[SEP]", "[PAD]", "[SEG-CLS]", "[SEG-SEP]", "[UNK]", "[MASK]", ],)
    tokenizer.pre_tokenizer = tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files, trainer=trainer)

    # make it compatible with transformers tokenizer
    tokenizer.cls_token = "[CLS]"
    tokenizer.cls_token_id = tokenizer.token_to_id("[CLS]")
    tokenizer.sep_token = "[SEP]"
    tokenizer.sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.seg_cls_token = "[SEG-CLS]"
    tokenizer.seg_cls_token_id = tokenizer.token_to_id("[SEG-CLS]")
    tokenizer.seg_sep_token = "[SEG-SEP]"
    tokenizer.seg_sep_token_id = tokenizer.token_to_id("[SEG-SEP]")
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.token_to_id("[PAD]")
    assert (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.seg_cls_token_id, tokenizer.seg_sep_token_id,) == ( 0, 1, 2, 3, 4,)
    return tokenizer

def train_byte_tokenizer(files, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[ "[CLS]", "[SEP]", "[PAD]", "[SEG-CLS]", "[SEG-SEP]", "[UNK]", "[MASK]", ],)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train(files, trainer=trainer)

    # make it compatible with transformers tokenizer
    tokenizer.cls_token = "[CLS]"
    tokenizer.cls_token_id = tokenizer.token_to_id("[CLS]")
    tokenizer.sep_token = "[SEP]"
    tokenizer.sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.seg_cls_token = "[SEG-CLS]"
    tokenizer.seg_cls_token_id = tokenizer.token_to_id("[SEG-CLS]")
    tokenizer.seg_sep_token = "[SEG-SEP]"
    tokenizer.seg_sep_token_id = tokenizer.token_to_id("[SEG-SEP]")
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.token_to_id("[PAD]")
    assert (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.seg_cls_token_id, tokenizer.seg_sep_token_id,) == ( 0, 1, 2, 3, 4,)
    return tokenizer


def load_a_tokenizer(filepath):
    tokenizer = Tokenizer.from_file(filepath)
    # make it compatible with transformers tokenizer
    tokenizer.cls_token = "[CLS]"
    tokenizer.cls_token_id = tokenizer.token_to_id("[CLS]")
    tokenizer.sep_token = "[SEP]"
    tokenizer.sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.seg_cls_token = "[SEG-CLS]"
    tokenizer.seg_cls_token_id = tokenizer.token_to_id("[SEG-CLS]")
    tokenizer.seg_sep_token = "[SEG-SEP]"
    tokenizer.seg_sep_token_id = tokenizer.token_to_id("[SEG-SEP]")
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.token_to_id("[PAD]")
    assert ( tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.seg_cls_token_id, tokenizer.seg_sep_token_id,) == (0, 1, 2, 3, 4,)
    return tokenizer


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup(parser, project_name):
    ## Required parameters
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--load_model_path",
        default=None,
        type=str,
        help="Path to trained model: Should contain the .bin files",
    )
    ## Other parameters
    parser.add_argument(
        "--train_filename",
        default=None,
        type=str,
        help="The train filenames (source and target files).",
    )
    parser.add_argument(
        "--dev_filename",
        default=None,
        type=str,
        help="The dev filename. (source and target files).",
    )
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. (source and target files).",
    )
    parser.add_argument(
        "--gen_filename",
        default=None,
        type=str,
        help="The gen filename. (source and target files).",
    )

    parser.add_argument(
        "--encoder_num_hidden_layers",
        default=6,
        type=int,
        help="number of transformer encoder layers",
    )

    parser.add_argument(
        "--encoder_hidden_size",
        default=512,
        type=int,
        help="hidden size for transformer encoder",
    )

    parser.add_argument(
        "--encoder_num_attention_heads",
        default=8,
        type=int,
        help="number of attention heads for transformer encoder layers",
    )

    parser.add_argument(
        "--decoder_num_hidden_layers",
        default=6,
        type=int,
        help="number of transformer decoder layers",
    )

    parser.add_argument(
        "--max_source_length",
        default=64,
        type=int,
        help="The maximum total source sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=64,
        type=int,
        help="The maximum total target sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--use_ws_tokenizer", action="store_true", help="use whitespace for tokenization"
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--vocab_size",
        default=6000,
        type=int,
        help="vocabulary size",
    )

    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--beam_size", default=10, type=int, help="beam size for beam search"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--eval_steps", default=-1, type=int, help="")
    parser.add_argument("--train_steps", default=-1, type=int, help="")
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument('--eval_step_list', nargs='+', help='list of steps for evaluation')

    # print arguments
    args = parser.parse_args()

    # Setup a single training
    device = torch.device("cuda:0")
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # setup wandb
    exp_name = args.output_dir.split("/")[-1]  # convention
    job_types = []
    if args.do_train:
        job_types.append("train")
    if args.do_eval:
        job_types.append("valid")
    if args.do_test:
        job_types.append("test")
    wandb.init(project=project_name, group=exp_name, job_type="-".join(job_types))
    wandb.config.update(args)
    return args


def adapt_arg_parser_for_seg2seg(parser):
    parser.add_argument(
        "--num_segments", default=-1, type=int, help="Number of latent segments"
    )
    parser.add_argument(
        "--max_source_segment_length",
        default=None,
        type=int,
        help="max length for a single source segment. If 0, a heuristic will be used.",
    )
    parser.add_argument(
        "--max_target_segment_length",
        default=0,
        type=int,
        help="max length for a single target segment. If 0, a heuristic will be used.",
    )
    parser.add_argument(
        "--num_segre_samples",
        default=8,
        type=int,
        help="number of samples used for VIMCO estimation",
    )
    parser.add_argument(
        "--monotone_align",
        action="store_true",
        help="if true, reordering will be disabled",
    )
    parser.add_argument(
        "--use_btg_decode",
        action="store_true",
        help="if true, use btg for decoding, instead of original seq2seq",
    )
    parser.add_argument(
        "--use_posterior",
        action="store_true",
        help="if true, a posterior of btg conditioned on the target will be used",
    )
    parser.add_argument(
        "--geo_p",
        default=0.5,
        type=float,
        help="parameter for geometric distribution over the number of segments",
    )
    parser.add_argument(
        "--use_argmax", action="store_true", help="whether to use argmax"
    )
    return parser
