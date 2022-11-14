import os
import wandb
import argparse
import random
import numpy as np
from itertools import cycle
from tqdm import tqdm, trange

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup

from neural_btg.models.btg_seg2seg import BTGSeg2Seg
from neural_btg.utils import setup_logger_file, setup_logger, read_examples, compute_bleu_score
from neural_btg.commands.train_utils import train_byte_tokenizer, train_ws_bpe_tokenizer, load_a_tokenizer, setup, convert_examples_to_features, adapt_arg_parser_for_seg2seg

logger = setup_logger()

def train_seg2seg(project_name):
    # 1. setup args
    parser = argparse.ArgumentParser()
    args = setup(adapt_arg_parser_for_seg2seg(parser), project_name=project_name)
    setup_logger_file(logger, args.output_dir)
    logger.info(args)

    # 2. build model and tokenizer
    tokenizer_file_path = os.path.join(args.output_dir, "tokenizer.json")
    if args.do_train:
        if args.use_ws_tokenizer:
            tokenizer = train_ws_bpe_tokenizer(args.train_filename.split(","), vocab_size=args.vocab_size)
        else:
            tokenizer = train_byte_tokenizer(args.train_filename.split(","), vocab_size=args.vocab_size)
        tokenizer.save(tokenizer_file_path)
    else:
        tokenizer = load_a_tokenizer(tokenizer_file_path)

    # 3. build seq2seq
    config = T5Config.from_pretrained("t5-small")
    config.decoder_start_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.sep_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.vocab_size = tokenizer.get_vocab_size()
    seq2seq = T5ForConditionalGeneration(config=config)

    # 4. build seg2seg
    model = BTGSeg2Seg(
        seq2seq,
        args,
        sos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id,
        seg_sos_id=tokenizer.seg_cls_token_id, 
        seg_eos_id=tokenizer.seg_sep_token_id, 
        pad_id=tokenizer.pad_token_id, 
        use_posterior=args.use_posterior,
        beam_size=args.beam_size,
    )

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(args.device)

    if args.do_train:
        # prepare training data loader
        train_examples = read_examples(args.train_filename)

        seg_train_features = convert_examples_to_features(train_examples, tokenizer, 100, 100, stage="train")
        seg_source_ids = torch.tensor([f.source_ids for f in seg_train_features], dtype=torch.long)
        seg_source_mask = torch.tensor([f.source_mask for f in seg_train_features], dtype=torch.long)
        seg_target_ids = torch.tensor([f.target_ids for f in seg_train_features], dtype=torch.long)
        seg_target_mask = torch.tensor([f.target_mask for f in seg_train_features], dtype=torch.long)
        seg_train_data = TensorDataset(seg_source_ids, seg_source_mask, seg_target_ids, seg_target_mask)
        seg2seg_train_sampler = RandomSampler(seg_train_data)
        seg2seg_train_dataloader = DataLoader(seg_train_data, sampler=seg2seg_train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        seq_train_features = convert_examples_to_features(train_examples, tokenizer, 100, 100, stage="train")
        seq_source_ids = torch.tensor([f.source_ids for f in seq_train_features], dtype=torch.long)
        seq_source_mask = torch.tensor([f.source_mask for f in seq_train_features], dtype=torch.long)
        seq_target_ids = torch.tensor([f.target_ids for f in seq_train_features], dtype=torch.long)
        seq_target_mask = torch.tensor([f.target_mask for f in seq_train_features], dtype=torch.long)
        seq_train_data = TensorDataset(seq_source_ids, seq_source_mask, seq_target_ids, seq_target_mask)
        seq2seq_train_sampler = RandomSampler(seq_train_data)
        seq2seq_train_dataloader = DataLoader(seq_train_data, sampler=seq2seq_train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.train_steps

        # optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=num_train_optimization_steps)

        # training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps * args.train_batch_size // len(train_examples))

        ## prepare eval data
        eval_examples = read_examples(args.dev_filename)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, 100, 100, stage="dev")
        eval_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        eval_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
        eval_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
        eval_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(eval_source_ids, eval_source_mask, eval_target_ids, eval_target_mask)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # start to train
        model.train()
        best_bleu, best_loss = 0, 1e6
        bar = range(1, num_train_optimization_steps + 1)
        seg2seg_train_dataloader = cycle(seg2seg_train_dataloader)
        seq2seq_train_dataloader = cycle(seq2seq_train_dataloader)
        num_seg_update = 0
        for global_step in bar:
            if global_step <= args.warmup_steps: # seq2seq warmup
                train_num_segments = 1
                train_gradient_accumulation_steps = 1
                train_dataloader = seq2seq_train_dataloader
            else:   # seg2seg training
                train_num_segments = args.num_segments
                train_gradient_accumulation_steps = args.gradient_accumulation_steps
                train_dataloader = seg2seg_train_dataloader
                num_seg_update += 1

            for _ in range(train_gradient_accumulation_steps):
                batch = next(train_dataloader)
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch

                try:
                    loss = model.btg_loss(
                        source_ids,
                        source_mask,
                        target_ids,
                        target_mask,
                        num_segments=train_num_segments,
                        max_src_seg_len=args.max_source_segment_length,
                        max_tgt_seg_len=args.max_target_segment_length,
                        max_src_len=args.max_source_length,
                        max_tgt_len=args.max_target_length,
                    )
                except (KeyError, RuntimeError) as e:
                    logger.warning(f"OOM error, skip one batch")
                    torch.cuda.empty_cache()
                    loss = torch.tensor(0.0, requires_grad=True).to(args.device)

                loss = loss / args.gradient_accumulation_steps
                loss.backward()

            #  gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if global_step  % args.eval_steps == 0:
                logger.info(" step {} num_seg_update {}".format(global_step, num_seg_update))
                wandb.log({"train_loss": loss.item()}, step=global_step)

            if global_step > args.warmup_steps:
                logger.info("  step {} loss {}".format(global_step, loss.item()))

            # if (args.do_eval and (global_step % args.eval_steps == 0)) and global_step > args.warmup_steps:
            if (args.do_eval and (global_step % args.eval_steps == 0)):
                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # eval-part1: seq2seq loss
                model.eval()
                eval_loss_l = []
                for batch in tqdm(eval_dataloader, desc="compute dev loss"):
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        seq_loss = model.seq2seq_loss(source_ids, source_mask, target_ids, target_mask)
                    eval_loss_l.append(seq_loss.item())

                model.train()
                eval_loss = sum(eval_loss_l) / len(eval_loss_l)
                logger.info("  %s = %s", "eval_loss", str(round(eval_loss, 5)))
                logger.info("  %s = %s", "gloabl_step", str(global_step))
                logger.info("  " + "*" * 20)
                wandb.log({"eval_loss": eval_loss}, step=global_step)

                # save checkpoint
                step_output_dir = os.path.join(args.output_dir, "step_checkpoints")
                if not os.path.exists(step_output_dir):
                    os.makedirs(step_output_dir)
                output_model_file = os.path.join(step_output_dir, f"{global_step}.bin")
                torch.save(model.state_dict(), output_model_file)
                
                # save the checkponit with best loss
                if eval_loss < best_loss:
                    logger.info("  Best loss:%s", round(eval_loss, 5))
                    logger.info("  " + "*" * 20)
                    best_loss = eval_loss

                # eval-part2: calculate bleu on dev, choose one of the decoding strategy depending on args.use_btg_decode
                model.eval()
                all_predictions = []
                for batch in tqdm(eval_dataloader, desc="compute dev bleu"):
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, source_mask, _, _ = batch
                    with torch.no_grad():
                        pred_ids = model.seq2seq_decode(source_ids, source_mask, seg_gen=False).tolist()

                        if args.use_btg_decode:
                            pred_ids, _ = model.btg_decode(
                                source_ids,
                                source_mask,
                                num_segments=args.num_segments,
                                max_src_seg_len=args.max_source_segment_length,
                                use_argmax=args.use_argmax,
                            )
                        else:
                            pred_ids = model.seq2seq_decode(source_ids, source_mask, seg_gen=False).tolist()

                        pred_txts = tokenizer.decode_batch(pred_ids, skip_special_tokens=True)
                        all_predictions += pred_txts

                model.train()
                refs = []
                accs = []
                with open(os.path.join(args.output_dir, f"dev_{global_step}.output"), "w") as f1, open( os.path.join(args.output_dir, f"dev_{global_step}.gold"), "w") as f2: 
                    for hyp, gold in zip(all_predictions, eval_examples):
                        f1.write(hyp + "\n")
                        f2.write(gold.target + "\n")
                        refs.append(gold.target)
                        accs.append(hyp == gold.target)

                dev_bleu = compute_bleu_score(all_predictions, refs)
                dev_acc = np.mean(accs) * 100
                logger.info("  %s = %s " % ("sacre bleu", str(dev_bleu)))
                logger.info("  %s = %s " % ("xMatch", str(round(dev_acc, 4))))
                logger.info("  " + "*" * 20)
                wandb.log({"dev-sacre-bleu": dev_bleu, "dev-acc": dev_acc}, step=global_step,)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    output_dir = os.path.join(args.output_dir, "checkpoint-best-bleu")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model.state_dict(), output_model_file)

    if args.do_test:
        dev_examples = read_examples(args.dev_filename)
        dev_features = convert_examples_to_features(dev_examples, tokenizer, 100, 100, stage="test")
        dev_source_ids = torch.tensor([f.source_ids for f in dev_features], dtype=torch.long)
        dev_source_mask = torch.tensor([f.source_mask for f in dev_features], dtype=torch.long)
        dev_data = TensorDataset(dev_source_ids, dev_source_mask)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)

        test_examples = read_examples(args.test_filename)
        test_features = convert_examples_to_features(test_examples, tokenizer, 100, 100, stage="test")
        test_source_ids = torch.tensor([f.source_ids for f in test_features], dtype=torch.long)
        test_source_mask = torch.tensor([f.source_mask for f in test_features], dtype=torch.long)
        test_data = TensorDataset(test_source_ids, test_source_mask)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        best_step, best_dev_bleu = -1, -1
        logger.info(f"evaluating checkpoints from {args.eval_step_list}")
        for step in args.eval_step_list:
            model_path = os.path.join(args.output_dir, f"step_checkpoints/{step}.bin")
            if os.path.exists(model_path):
                logger.info("loading model from {}".format(model_path))
                model.load_state_dict(torch.load(model_path))
            else:
                continue

            # model selection of btg model
            model.eval()
            btg_predictions = []
            for batch in tqdm(dev_dataloader, total=len(dev_dataloader), desc=f"eval step {step}"):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    if args.use_btg_decode:
                        btg_pred_ids, _ = model.btg_decode(
                            source_ids,
                            source_mask,
                            num_segments=args.num_segments,
                            max_src_seg_len=args.max_source_segment_length,
                            use_argmax=args.use_argmax,
                        )
                    else:
                        btg_pred_ids = model.seq2seq_decode(source_ids, source_mask, seg_gen=False).tolist()

                    btg_pred_txts = tokenizer.decode_batch(btg_pred_ids, skip_special_tokens=True)
                    btg_predictions += btg_pred_txts

            btg_accs = []
            refs = []
            for btg_hyp, gold in zip(btg_predictions, dev_examples):
                refs.append(gold.target)
                btg_accs.append(btg_hyp == gold.target)

            btg_bleu = compute_bleu_score(btg_predictions, refs)
            btg_acc = np.mean(btg_accs) * 100
            logger.info(f"Step {step}, dev bleu {btg_bleu:.4f}, accuracy {btg_acc:.4f}")

            if btg_bleu > best_dev_bleu:
                best_step = step
                best_dev_bleu = btg_bleu
        logger.info(f"best dev bleu: {best_dev_bleu} from step {best_step}")

        # final test
        model_path = os.path.join(args.output_dir, f"step_checkpoints/{best_step}.bin")
        logger.info("loading best model from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
        model.eval()

        btg_predictions = []
        refs = [ex.target for ex in test_examples]
        for batch in tqdm(test_dataloader, total=len(test_dataloader), desc=f"test with step {best_step}"):
            batch = tuple(t.to(args.device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                if args.use_btg_decode:
                    btg_pred_ids, _ = model.btg_decode(
                        source_ids,
                        source_mask,
                        num_segments=args.num_segments,
                        max_src_seg_len=args.max_source_segment_length,
                        use_argmax=args.use_argmax,
                    )
                else:
                    btg_pred_ids = model.seq2seq_decode(source_ids, source_mask, seg_gen=False).tolist()

                btg_pred_txts = tokenizer.decode_batch(btg_pred_ids, skip_special_tokens=True)
                btg_predictions += btg_pred_txts

        btg_bleu = compute_bleu_score(btg_predictions, refs)
        logger.info(f"test bleu: {btg_bleu}")


if __name__ == "__main__":
    train_seg2seg("nmt-chr-en-seg2seg")
