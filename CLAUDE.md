# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A from-scratch PyTorch implementation of the Transformer ("Attention is All You Need") for
de→en machine translation. Three entry-point scripts (`preprocess.py` → `train.py` →
`translate.py`) wrap a self-contained model package in `transformer/`.

There is no test suite, linter config, or build step. Verification is done by running the
scripts end-to-end.

## Environment

Dependencies are pinned in `requirements.txt` to an **old stack**: `python 3.6`, `pytorch 1.3.1`,
`spacy 2.3.5`, plus `dill`, `tqdm`, `tensorboard`.

The code uses the **legacy torchtext API** (`torchtext.data.Field`, `BucketIterator`,
`torchtext.datasets.Multi30k.splits`, `TranslationDataset`), which was removed in
torchtext 0.9. Any torchtext ≥ 0.9 requires `torchtext.legacy.*` imports or a rewrite of
`prepare_dataloaders*` in `train.py` and both `main*` functions in `preprocess.py`.

spaCy models must be downloaded before preprocessing: `python -m spacy download en` / `de`.

## Commands

```bash
# 1. Preprocess (Multi30k + spaCy tokenizer) -> single pickle
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl

# 2. Train
python train.py -data_pkl m30k_deen_shr.pkl -embs_share_weight -proj_share_weight \
  -label_smoothing -output_dir output -b 256 -warmup 128000 -epoch 400

# 2b. Reference training config (env-var driven)
gpu=0 lr_mul=0.5 scale_emb_or_prj=emb bash train_multi30k_de_en.sh

# 3. Decode the test split with beam search
python translate.py -data_pkl m30k_deen_shr.pkl -model output/model.chkpt -output prediction.txt
```

`-output_dir` is effectively required — `train.py` hits a bare `raise` without it.
`-no_cuda` on `train.py`/`translate.py` forces CPU.

## Two data pipelines (mutually exclusive)

`preprocess.py` contains two independent `main` functions and **switching between them means
editing the `__main__` block at the bottom of the file** — there is no flag.

- `main_wo_bpe()` (currently active, and the tested path): downloads Multi30k via torchtext,
  tokenizes with spaCy, builds separate `SRC`/`TRG` Fields (optionally merged by `-share_vocab`),
  and pickles `{settings, vocab: {src, trg}, train, valid, test}` — examples included.
- `main()` (BPE / WMT'17, marked WIP and not fully tested): downloads raw WMT corpora, learns BPE
  codes via `learn_bpe.py`, writes `.src`/`.trg` text files, and pickles only
  `{settings, vocab: <single shared Field>}`. Training then reads the text files through
  `-train_path`/`-val_path`; this path asserts `-embs_share_weight`. `translate.py` does **not**
  support BPE data.

Pickles are written with `dill` (needed to serialize the Field tokenizer closures).

## Architecture

`transformer/` is layered bottom-up; each file depends only on the one below it:

- `Modules.py` — `ScaledDotProductAttention` (masking via `masked_fill(mask == 0, -1e9)`).
- `SubLayers.py` — `MultiHeadAttention`, `PositionwiseFeedForward`. Both are **post-LN**:
  residual is added first, then `LayerNorm`.
- `Layers.py` — `EncoderLayer` (self-attn + FFN), `DecoderLayer` (self-attn + cross-attn + FFN).
- `Models.py` — `PositionalEncoding`, `Encoder`, `Decoder`, `Transformer`, and the mask helpers
  `get_pad_mask` / `get_subsequent_mask`.
- `Translator.py` — beam search over a trained `Transformer`.
- `Optim.py` — `ScheduledOptim`, the Noam warmup schedule.

### Conventions that cut across files

**Tensor layout.** torchtext's `BucketIterator` yields time-major batches; `patch_src`/`patch_trg`
in `train.py` transpose them to batch-major `(batch, seq_len)` and split the target into
teacher-forcing input `trg[:, :-1]` and flattened `gold = trg[:, 1:]`. Accordingly
`Transformer.forward` returns **flattened logits** `(batch * seq_len, vocab)`, not 3-D — loss
functions in `train.py` assume this.

**Positional encoding cap.** `n_position=200` is a hard ceiling baked into the sinusoid table;
longer sequences index out of bounds. `preprocess.py -max_len` defaults to 100.

**Weight sharing and scaling.** Per §3.4 of the paper, the `√d_model` factor is applied either to
the embedding output (`-scale_emb_or_prj emb`) or inversely to the projection output (`prj`,
default) or not at all (`none`). Critically, **scaling is only applied when
`trg_emb_prj_weight_sharing` is on** (`-proj_share_weight`); otherwise both flags are forced off.

**Learning rate.** `ScheduledOptim` owns the LR — call `optimizer.step_and_update_lr()`, never
`optimizer.step()`. The formula is `lr_mul * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)`,
so `-lr_mul` rescales the whole Noam curve. Small batch with short warmup triggers a warning in
`train.py`; the README's best-known setting is `lr_mul 0.5`, `scale_emb_or_prj emb`, `warmup 4000`,
`b 256`.

**Checkpoints carry their own config.** `train.py` saves `{epoch, settings: opt, model: state_dict}`
and `translate.py` rebuilds the `Transformer` from `checkpoint['settings']`. Renaming or removing
a `train.py` argparse option therefore breaks loading of every existing checkpoint.

## Known rough edges

These are pre-existing; don't assume they're intentional if you touch the surrounding code.

- `translate.py:load_model` does **not** pass `scale_emb_or_prj` when reconstructing the model, so
  it always defaults to `'prj'`. A model trained with `-scale_emb_or_prj emb` is rebuilt with the
  wrong scaling.
- `-save_mode all` writes checkpoints to the **current working directory**, while `-save_mode best`
  (the default) writes into `-output_dir`.
- `Translator.translate_sentence` asserts batch size 1; there is no batched decoding.
- `learn_bpe.py` / `apply_bpe.py` are vendored verbatim from
  [subword-nmt](https://github.com/rsennrich/subword-nmt/) — keep changes there minimal and
  upstream-compatible.
