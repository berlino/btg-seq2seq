# BTG Seq2Seq

> BTG Seq2Seq is a neural transducer that maintains the flexibility of standard sequence-to-sequence (seq2seq) models while incorporating hierarchical phrases as a source of inductive bias during training and as explicit constraints during inference. 

[Hierarchical Phrase-based Sequence-to-Sequence Learning]()
EMNLP 2022


## Setup

It's recommended to use conda to create a virtual environment.

    conda create --name btg-seq2seq python==3.8
    conda activate btg-seq2seq
    conda install pytorch cudatoolkit=11.7 -c pytorch-nightly -c nvidia
    pip install -e .

wandb is optional for logging.

## Code Structure

* `structs/segre.py`: parsers (segre is short for "segmentation and reordering")
* `models/btg_seg2seg.py`: main model file of btg-seq2seq
* `commands/*`: trainer for  btg-seq2seq (also named "seg2seg")

## Experiments

### Sanity Check 

To start with, you can run btg-seq2seq on the toy chr-en traslation dataset, which only contains 32 translation pairs as both training and test.
Run with:

```
./scripts/train_nmt_toy_chr_en_seg2seg.sh
```

You're expected to get 100 BLEU score.

### Machine Translation

Run the following script to train the btg-seq2seq for chr-en translation.

```
./scripts/train_nmt_chr_en_seg2seg.sh
```

Run inference with seq2seq

```
./scripts/infer_nmt_seq.sh
```

Run inference with btg

```
./scripts/infer_nmt_btg.sh
```

### Practical notes

* Memory: If OOM error pops up, consider decrease `batch_size` (e.g., to 100), `max_source/target_length` (e.g., 36 or 16) or `num_segments` (e.g., 2). 
* Training time: Consider increase `warmup_steps` (e.g., to 6k) to speedup the training, and decrease `train_steps` During warmup, the underlying seq2seq is pretrained (i.e., num\_segments is set to 1)
* Adapt to a new language pair: replace the corresponding train/dev/test filenames in the script; tune `vocab_size` (the size of BPE tokens) and `train\_steps` depending on your data size. 

## License
MIT
