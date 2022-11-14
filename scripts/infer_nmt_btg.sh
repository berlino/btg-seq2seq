#!/bin/bash

exp_id=341
output_dir="log/neural_btg_nmt_chr_en_seg2seg_$exp_id"

CUDA_VISIBLE_DEVICES=2 python neural_btg/commands/train_nmt_seg2seg.py \
        --do_test \
        --dev_filename data/chr-en/dev.chr,data/chr-en/dev.en \
        --test_filename data/chr-en/test.chr,data/chr-en/test.en \
        --output_dir $output_dir \
	--use_btg_decode \
	--use_argmax \
	--num_segments -1 \
        --geo_p 0.6 \
        --max_source_segment_length 8 \
        --max_target_segment_length 8 \
        --beam_size 15 \
        --eval_batch_size 1 \
	--eval_step_list 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000
