#!/bin/bash

exp_id=0
output_dir="log/neural_btg_nmt_toy_chr_en_seg2seg_$exp_id"

CUDA_VISIBLE_DEVICES=0	 python neural_btg/commands/train_nmt_seg2seg.py \
	--do_train \
	--do_eval \
	--do_test \
	--use_argmax \
	--use_btg_decode \
	--use_posterior \
	--train_filename data/toy-chr-en/train.chr,data/toy-chr-en/train.en \
	--dev_filename data/toy-chr-en/dev.chr,data/toy-chr-en/dev.en \
	--output_dir $output_dir \
	--max_source_length 64 \
	--max_target_length 64 \
	--num_segments 3 \
	--geo_p 0.6 \
	--num_segre_samples 3 \
	--max_source_segment_length 8 \
	--max_target_segment_length 8 \
	--gradient_accumulation_steps 1 \
	--beam_size 5 \
	--train_batch_size 32 \
	--eval_batch_size 1 \
	--learning_rate 1e-3 \
	--warmup_steps 20 \
	--train_steps 300 \
	--eval_steps 50
