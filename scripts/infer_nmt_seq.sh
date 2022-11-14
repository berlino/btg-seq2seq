#!/bin/bash

exp_id=341
output_dir="log/neural_btg_nmt_chr_en_seg2seg_$exp_id"

CUDA_VISIBLE_DEVICES=1 python neural_btg/commands/train_nmt_seg2seg.py \
        --do_test \
        --dev_filename data/chr-en/dev.chr,data/chr-en/dev.en \
        --test_filename data/chr-en/test.chr,data/chr-en/test.en \
        --output_dir $output_dir \
        --beam_size 5 \
        --eval_batch_size 200 \
	--eval_step_list 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000
