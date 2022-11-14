#!/bin/bash

exp_id=351
output_dir="log/neural_btg_nmt_chr_en_seg2seg_$exp_id"

CUDA_VISIBLE_DEVICES=0 python neural_btg/commands/train_nmt_seg2seg.py \
        --do_train \
        --do_eval \
        --use_posterior \
        --train_filename data/chr-en/train.chr,data/chr-en/train.en \
        --dev_filename data/chr-en/dev.chr,data/chr-en/dev.en \
        --output_dir $output_dir \
        --max_source_length 64 \
        --max_target_length 64 \
        --vocab_size 8000 \
        --num_segments 3 \
        --geo_p 0.6 \
        --num_segre_samples 1 \
        --max_source_segment_length 8 \
        --max_target_segment_length 8 \
        --gradient_accumulation_steps 2  \
        --beam_size 5 \
        --train_batch_size 200 \
        --eval_batch_size 200 \
        --learning_rate 3e-4 \
        --warmup_steps 0 \
        --train_steps 30000 \
        --eval_steps 1000
