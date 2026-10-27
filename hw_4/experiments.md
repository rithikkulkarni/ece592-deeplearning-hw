# HW4 Experiments

Running log of every training run for HW4 (Pokémon next-token prediction). One entry per run,
with the exact config, what changed, and what we observed. Roadmap lives in `../ROADMAP.md`.

**Important:** we get **no FID/PDR locally** — the TAs compute them on the submitted `.txt`.
Targets: Simple FID ≤ 86 / PDR ≥ 0.1, Medium ≤ 80 / ≥ 0.5, Strong ≤ 73 / ≥ 0.85 (both must pass).
So "result" below means the signals we *can* see: training loss, dev reconstruction accuracy
(a weak proxy per the slides), generated-image diversity stats, and the visual grid. None of
these confirm that a tier passes.

**Signals tracked per run**
- Final / best training loss (checkpoint is saved at the lowest *training* loss).
- Dev reconstruction accuracy (greedy `generate`, last 160 tokens vs. labels).
- Tail stats on the generated `.txt` (last 160 of 400 tokens): avg distinct colors per tail vs. the
  given prefix, share of token 0 (white), count of single-token tails, avg longest repeated run.
- Visual look at the generated grid (cell 38) — *add your impression per run*.
- **Proxy PDR / proxy FID** (from exp_6 onward; the notebook now ends with a "Proxy Metrics"
  section, cells 41-45, that prints them on every run-all). These are *local stand-ins*, not the
  TAs' numbers: PDR comes from our own small CNN trained on real vs. bottom-corrupted images, FID
  from Inception features vs. 712 real train+dev images (80 generated images → noisy; a real-vs-
  real floor and a blank-bottom control are printed alongside). Use them to compare our runs with
  each other only — they can't tell us whether Strong's FID ≤ 73 / PDR ≥ 0.85 is met. Exp_1-5
  were run before this existed; their `exp_N_results.txt` files could be scored retroactively.

## Summary

| Exp | Model | Params | Epochs | LR | Final train loss | Best dev acc | Status |
|---|---|---|---|---|---|---|---|
| exp_1 | GPT-2, 2L / 64d / 2h (sample defaults) | 136,384 | 50 | 1e-3 | 1.4746 | 0.4124 (ep 4) | done |
| exp_2 | GPT-2, 6L / 256d / 8h ("config C") | 4,884,224 | 100 | 3e-4 | 0.4366 | 0.4115 (ep 3) | done |
| exp_3 | exp_2 + dropout 0.3 + dev-loss checkpointing | 4,884,224 | 100 | 3e-4 | 0.9507 (ep 100, dropout on) | 0.4134 (ep 4) | done — saved ckpt = ep 52, dev loss 1.566 |
| exp_4 | Llama, 8L / 384d / 6h, attn dropout 0.3 | 14,290,560 | 100 | 3e-4 | 0.0783 (ep 100) | 0.3916 (ep 85) | done — saved ckpt = ep 12, dev loss **1.4533** |
| exp_5 | exp_4 + warmup/cosine LR schedule, 30 epochs | 14,290,560 | 30 | 3e-4 peak | 0.7470 (ep 30, dropout on) | 0.4010 (ep 15) | done — saved ckpt = ep 12, dev loss 1.4652 |

---

## exp_1 — Sample defaults (baseline run)

First run of the provided notebook, completely unchanged, to confirm the pipeline works and get a
reference point.

| Variable | Value |
|---|---|
| Architecture | `GPT2Config`: `n_layer`=2, `n_embd`=64, `n_head`=2, `n_positions`=400 (`n_ctx`=128 in config, likely inert — unverified) |
| Trainable params | 136,384 |
| Dropout | attn/embd/resid all 0.1 |
| Epochs | 50 |
| Batch size | 16 |
| Optimizer | AdamW, lr=1e-3, weight_decay=0.1 |
| Loss | CrossEntropyLoss over all 399 positions |
| Seed | `set_seed(0)` |
| Decoding (dev + test) | `model.generate`, defaults (greedy), `max_length=400` |
| Checkpoint rule | lowest training loss (saved every time loss improves) |
| Platform | Colab T4 via VS Code kernel |

**Result**
- Train loss fell steadily: 3.7601 (ep 1) → 2.0952 (ep 6) → 1.7103 (ep 21) → 1.4746 (ep 50). Still
  decreasing at epoch 50, so the model hadn't plateaued — the best-loss checkpoint is the last epoch.
- **Dev reconstruction accuracy went *down* as training loss went down:** 0.4106 (ep 1) → 0.4124
  best at ep 4 → 0.3502 (ep 26) → 0.3145 (ep 50). Consistent with the slides' warning that this
  metric doesn't track generation quality (possible explanations, not verified: greedy decoding
  matches blank/background pixels early, and as the model learns finer detail it commits to
  detail that doesn't match the true pixels exactly; or overfitting). We do not read it as "the
  model got worse" — the slide's guidance is to ignore it for checkpointing.
- Speed: ~1s per training epoch, ~1s per dev batch (5 batches) — the whole run is short.
- Generated `.txt` (80 lines × 400 ints, validated OK). Tail stats (last 160 tokens, 80 images):
  - Avg distinct colors per tail 4.6 vs. 22.8 in the given prefix.
  - 6019 / 12800 tail tokens (47%) are token 0 (white).
  - 7 of 80 tails are a single repeated token; avg longest repeated run 47 tokens (max 160).
  - Colors used in tails: 87 distinct, vs. 162 in prefixes.
  → Completions are much blander than the real prefixes (typical greedy small-model collapse),
  though some of the white is legitimately background.
- Visual impression of the grid: _(add yours)_
- FID / PDR: **unknown** (TA-computed). Assumed (not confirmed) to be around the Simple tier.
- Not yet verified: that the first 240 generated tokens equal the test input prefix.
- Output file: saved as `hw_4/exp_1_results.txt` (copy of `reconstructed_results.txt`, which also
  still holds exp_1's output). Later runs are extracted to `exp_N_results.txt` via
  `python extract_submission.py --out exp_N_results.txt` so nothing gets overwritten.

---

## exp_2 — Config C: bigger model, lower LR

First real attempt to improve on exp_1. exp_1's model is tiny (136k params) and its loss was still
falling at epoch 50, so capacity/training budget looks like the limit. Chosen by the user from the
options presented (A: longer only, B: moderate scale-up, C: larger jump). Several variables change
at once, deliberately — a big jump is easier to read through loss / tail stats / visuals than a
small one when we have no FID/PDR. Lower LR because larger models usually need it (rule of thumb,
not verified here).

| Variable | Value | Changed from exp_1? |
|---|---|---|
| Architecture | GPT-2, `n_layer`=6, `n_embd`=256, `n_head`=8 | **yes** (was 2 / 64 / 2) |
| Trainable params | 4,884,224 (printed in cell 28; my ~5M estimate was close) | **yes** (was 136,384) |
| Dropout | 0.1 / 0.1 / 0.1 | no |
| Epochs | 100 | **yes** (was 50) |
| Batch size | 16 | no |
| Optimizer | AdamW, lr=3e-4, weight_decay=0.1 | **yes** (lr was 1e-3) |
| Loss | CrossEntropyLoss | no |
| Seed | `set_seed(0)` | no |
| Decoding | greedy (defaults) | no |
| Checkpoint rule | lowest training loss | no |
| Platform | Colab T4 via VS Code kernel | no |

**What we'd watch for:** train loss clearly below exp_1's 1.47; tail diversity (colors per tail,
share of white, single-token tails) improving vs. exp_1; signs of overfitting (only 632 images,
no validation loss in the notebook — dev accuracy is our only overfitting signal, and it behaved
oddly in exp_1); per-epoch time, since the dev `generate` pass gets slower with a bigger model.

**Result**
- Train loss fell all the way to **0.4366** at epoch 100 (exp_1 ended at 1.4746): 3.26 (ep 1) →
  1.71 (ep 11) → 1.20 (ep 41) → 0.80 (ep 71) → 0.44 (ep 100). Never plateaued; it was still dropping
  ~0.01-0.02 per epoch at the end. The saved checkpoint is again the last epoch.
- **Loss this low on only 632 images strongly suggests memorization** (exp_1 at 1.47 was far from
  that). We have no validation loss, so we can't measure the generalization gap directly.
- Dev reconstruction accuracy: best 0.4115 at epoch 3, ending at 0.3848 (ep 100), wandering
  0.33-0.39 in between. Same pattern as exp_1: it does not improve as training loss falls, so it is
  uninformative about quality (per the slides). Final value is slightly higher than exp_1's 0.3145
  but within the noise of this metric.
- Speed: ~5s per training epoch, ~4s per dev batch (vs ~1s / ~1s in exp_1) — still short overall.
- Generated `.txt` (`hw_4/exp_2_results.txt`, validated 80 × 400, values 0-166). Tail stats vs. exp_1:

  | Stat (last 160 tokens, 80 images) | exp_1 | exp_2 |
  |---|---|---|
  | Avg distinct colors per tail (prefix avg = 22.8) | 4.59 | **14.19** |
  | Distinct colors used across all tails | 87 | **154** |
  | Single-token tails | 7 | **0** |
  | Avg longest repeated run | 47.6 | 42.3 |
  | Share of tail tokens that are white (0) | 0.47 | 0.577 |
  | Distinct tails (of 80) | 76 | 80 |

  → Completions are far more varied and much closer to the prefixes' color richness, and no tail
  collapses to one token. The white share *rose*, so it's not simply "more detail everywhere"; the
  repeated-run length is still long (~42). Whether the added variety is plausible Pokémon structure
  or noise/memorized fragments is **not something these stats can tell us**.
- Sanity check: the 80 prefixes (first 240 tokens) are identical between exp_1 and exp_2 files, as
  expected for the same test inputs. (Still not compared directly to the dataset's test sequences.)
- Token agreement between the two runs' tails is only 0.431 — the models produce quite different
  completions.
- Visual impression of the grid: _(add yours)_
- FID / PDR: **unknown** (TA-computed).

---

## exp_3 — Anti-overfitting: dropout 0.3 + dev-loss checkpointing

exp_2's training loss reached 0.4366 on 632 images, which strongly suggests memorization, and the
checkpoint rule (lowest *training* loss) always picks the last, most memorized epoch. exp_3 attacks
this two ways: stronger dropout, and a teacher-forced dev loss so we can both *see* overfitting and
save the epoch with the best dev loss instead. Same model, lr and epochs as exp_2.

User's reasoning: try 0.3 and see whether the model ends up *under*fitting; if it does, raise the
epoch count and re-run (that would be exp_4).

| Variable | Value | Changed from exp_2? |
|---|---|---|
| Architecture | GPT-2, `n_layer`=6, `n_embd`=256, `n_head`=8 (4,884,224 params) | no |
| Dropout | attn / embd / resid all **0.3** | **yes** (was 0.1) |
| Epochs | 100 | no |
| Batch size | 16 | no |
| Optimizer | AdamW, lr=3e-4, weight_decay=0.1 | no |
| Loss | CrossEntropyLoss | no |
| Seed | `set_seed(0)` | no |
| Dev loss (new) | teacher-forced CE over all 399 positions of the 80 dev images, eval mode (dropout off), printed every epoch | **yes** (new) |
| Checkpoint rule | lowest **dev loss** | **yes** (was lowest training loss) |
| Decoding | greedy (defaults) | no |
| Platform | Colab T4 via VS Code kernel | no |

Notebook edits: cell 19 (new `dev_loss_dataloader`: dev split through `PixelSequenceDataset` in
`train` mode, since the existing `dev` mode gives the generation split), cell 26 (three dropout
values), cell 35 (dev-loss block each epoch; checkpoint saved on `avg_dev_loss`). Cell 34's
markdown still says "lowest training loss" — stale, not yet updated.

Choices made by the assistant, not explicitly picked by the user (easy to change): dev loss covers
**all 399 positions** (directly comparable to training loss) rather than only the last 160
(closer to the actual completion task, but noisier). The 80 dev images are now used both to report
dev loss and to pick the checkpoint, so the dev loss of the saved epoch is slightly optimistic.

**What we'd watch for:** the epoch where dev loss bottoms out (and whether it rises after — that
is the overfitting point); how train loss vs. dev loss compare under 0.3 dropout (train loss
printed is measured *with* dropout on, so it reads higher than it would in eval mode — don't
compare it to exp_2's directly); underfitting would look like dev loss still falling at epoch 100
with train and dev close together; whether tail diversity/visuals hold up or get worse than exp_2.
Dev reconstruction accuracy is expected to stay uninformative.

**Result**
- **It overfits, not underfits** — the "underfit → raise epochs" branch does not apply. Dev loss
  fell from 2.71 (ep 1) to a minimum of **1.566 at epoch 52**, then *rose* steadily to 1.741 at
  epoch 100, while training loss (dropout on) kept falling: 3.21 → 1.78 (ep 11) → 1.30 (ep 51) →
  0.95 (ep 100). Even at dropout 0.3 the model memorizes after ~50 epochs.

  | Epoch | 1 | 11 | 31 | 52 (best) | 71 | 91 | 100 |
  |---|---|---|---|---|---|---|---|
  | Train loss (dropout on) | 3.21 | 1.78 | 1.49 | 1.30 (ep 51) | 1.15 | 1.01 | 0.95 |
  | Dev loss | 2.71 | 1.81 | 1.66 | **1.566** | 1.61 | 1.70 | 1.74 |

- The **checkpoint used for the test generation is epoch 52** (the dev-loss rule worked as
  intended; the last 48 epochs only added memorization). 4,884,224 params as before. Speed ~5s
  train epoch / ~4s per dev batch, same as exp_2.
- Dev reconstruction accuracy again uninformative: best 0.4134 at epoch 4, 0.33-0.40 afterwards,
  0.3612 at epoch 100.
- Generated `.txt` (`hw_4/exp_3_results.txt`, validated 80 × 400, values 0-166; the 80 prefixes
  are identical to exp_2's). Tail stats:

  | Stat (last 160 tokens, 80 images) | exp_1 | exp_2 | exp_3 |
  |---|---|---|---|
  | Avg distinct colors per tail (prefix avg = 22.8) | 4.59 | 14.19 | **4.36** |
  | Distinct colors used across all tails | 87 | 154 | 88 |
  | Single-token tails | 7 | 0 | **9** |
  | Avg longest repeated run | 47.6 | 42.3 | **63.4** |
  | White share of tail tokens | 0.47 | 0.577 | 0.544 |
  | Distinct tails (of 80) | 76 | 80 | 76 |

  → **The better-generalizing checkpoint produces *blander* greedy completions** — back to
  exp_1-like collapse (≈4 colors per tail, 9 single-token tails, even longer repeated runs than
  exp_1). Token agreement of exp_3's tails with exp_1 is 0.520, with exp_2 only 0.463.
- Interpretation (a hypothesis, **not verified**): exp_2's extra tail diversity may have come from
  memorization (confident recall of training-image detail) rather than better modeling. A model
  that generalizes is less sure about any one continuation, so greedy decoding (always take the
  single most likely token) falls back to the safest, most common token — white or a repeat. If
  that's right, the collapse is mostly a *decoding* problem, and lower dev loss does not by itself
  mean better-looking generations. Needs a look at the grids and/or a sampling test to confirm.
- Visual impression of the grid: _(add yours)_
- FID / PDR: **unknown** (TA-computed).

---

## exp_4 — Llama architecture (larger)

Switch from GPT-2 to Llama, the architecture the slides recommend for the Strong tier. Chosen
by the user from the options presented: **size C (larger)** and **simple regularization** (attention
dropout 0.3 only + weight decay 0.1 + dev-loss checkpointing). Decoding was dropped as exp_4's
focus in favor of this (the sampling idea is still open for later; see exp_3's interpretation).

Llama differs from GPT-2 in: rotary position embeddings (no learned position table, relative
positions), RMSNorm instead of LayerNorm, SwiGLU MLP instead of GELU MLP, and separate input
embedding / output head here (`tie_word_embeddings=False`).

| Variable | Value | Changed from exp_3? |
|---|---|---|
| Architecture | `LlamaConfig`: 8 layers, hidden 384, 6 heads (head dim 64), MLP intermediate 1024, `num_key_value_heads`=6 (plain MHA), `rms_norm_eps`=1e-5, `max_position_embeddings`=400, untied embeddings | **yes** (was GPT-2 6L/256d/8h) |
| Trainable params | ~14M (rough estimate; record the count printed in cell 28) | **yes** (was 4,884,224) |
| Dropout | `attention_dropout`=0.3 only (Llama in HF has no embedding/residual dropout) | **yes** (was attn/embd/resid 0.3) |
| Special tokens | `bos`/`eos`/`pad` = None | no (GPT-2 config also had eos/pad None) |
| Epochs | 100 | no |
| Batch size | 16 | no |
| Optimizer | AdamW, lr=3e-4, weight_decay=0.1 | no |
| Loss | CrossEntropyLoss | no |
| Seed | `set_seed(0)` | no |
| Dev loss / checkpoint | teacher-forced dev loss over 399 positions; save lowest dev loss | no |
| Decoding | greedy (defaults) | no |
| Platform | Colab T4 via VS Code kernel | no |

Notebook edits: cell 10 (import `LlamaConfig`), cell 26 (GPT-2 config dict replaced by
`LlamaConfig(...)`). Training loop, dev-loss block, inference and extraction cells are unchanged —
they only use `model(...).logits` and `.generate`, but this has **not been run on Llama yet**.

Things flagged before running:
- Llama's default `eos_token_id=2` would stop `generate()` as soon as it produced color token 2, a
  real pixel. Set to None, as the GPT-2 config effectively did. Also set `bos` (default 1) to None.
- Not verified: the `transformers` version on the Colab runtime. Some newer versions moved the
  rotary-embedding settings (`rope_theta`) in `LlamaConfig`; if cell 26/28 errors, check this first.
- Cell 28's printed param count and the first epoch of cell 35 are the smoke test (no local
  torch/transformers install to test beforehand).
- Per-epoch time will be higher than exp_2/3 (~14M params; both training and the dev `generate`
  pass), and a bigger model may overfit sooner than exp_3's best epoch (52).
- Attention-only dropout is weaker regularization than exp_3's three dropout sites, so the dev-loss
  curve may turn up earlier; dev-loss checkpointing is the safeguard.

**What we'd watch for:** where dev loss bottoms out and its value vs. exp_3's 1.566 (direct,
same-metric comparison; lower = better generalization, though not guaranteed to mean better
images); tail stats vs. exp_1-3 (especially single-token tails and long repeated runs, which
greedy decoding produced in exp_3); visual grid; epoch time.

**Result**
- The Llama config worked first try: no errors, 14,290,560 params (my ~14M estimate was close),
  training/dev-loss/generation/extraction cells all ran unchanged. Per-epoch time ~11s train +
  ~7s per dev generation batch (vs ~5s / ~4s for exp_2/3).
- **Best dev loss 1.4533 at epoch 12** — better than exp_3's 1.566 (about 7% lower, same metric),
  so Llama generalizes better. But it reaches that point *much* earlier and then overfits hard:
  dev loss rises monotonically to 2.83 by epoch 100 while training loss collapses to 0.078
  (essentially memorized). The saved checkpoint is epoch 12, i.e. ~88 of the 100 epochs were
  wasted time (and exp_3's best was at epoch 52).

  | Epoch | 1 | 6 | 12 (best) | 21 | 41 | 61 | 81 | 100 |
  |---|---|---|---|---|---|---|---|---|
  | Train loss (dropout on) | 3.40 | – | – | 0.97 | 0.35 | 0.15 | 0.10 | 0.078 |
  | Dev loss | 3.04 | 1.57 | **1.453** | 1.55 | 2.12 | 2.52 | 2.72 | 2.83 |

- Dev reconstruction accuracy stayed uninformative: 0.36-0.39 throughout, best 0.3916 at epoch 85
  (a late, overfit epoch — the saved ep-12 checkpoint isn't the one that scored this).
- Generated `.txt` (`hw_4/exp_4_results.txt`, validated 80 × 400, values 0-166; prefixes identical
  to exp_3's). Tail stats:

  | Stat (last 160 tokens, 80 images) | exp_1 | exp_2 | exp_3 | exp_4 |
  |---|---|---|---|---|
  | Avg distinct colors per tail (prefix avg = 22.8) | 4.59 | 14.19 | 4.36 | **6.76** |
  | Distinct colors used across all tails | 87 | 154 | 88 | 119 |
  | Single-token tails | 7 | 0 | 9 | **0** |
  | Avg longest repeated run | 47.6 | 42.3 | 63.4 | **39.8** |
  | White share of tail tokens | 0.47 | 0.577 | 0.544 | 0.507 |
  | Distinct tails (of 80) | 76 | 80 | 76 | 80 |

  → Among the *dev-loss-selected* checkpoints (exp_3, exp_4), exp_4 is clearly less collapsed than
  exp_3: more colors per tail, no single-token tails, shortest repeated runs of all four runs.
  It's still far below the prefixes' 22.8 colors, and exp_2's higher color count (14.2) came from a
  heavily overfit checkpoint, so color count alone isn't a quality score. Tails agree only
  0.46-0.48 with every earlier run.
- Visual impression of the grid: _(add yours)_
- FID / PDR: **unknown** (TA-computed).
- Caveats: the checkpoint is from epoch 12 of a constant-lr run with no warmup or decay, so the
  curve's shape (sharp early min, steep climb) is partly an artifact of that schedule and
  attention-only dropout; with only dev loss as a guide we can't say how much better images look.

---

## exp_5 — Llama + LR warmup/cosine decay, 30 epochs

exp_4's dev loss bottomed at epoch 12 and then climbed for 88 epochs under a constant lr of 3e-4
with no warmup or decay, so its curve shape was partly an artifact of the schedule. exp_5 changes
only the optimization: a standard transformer schedule, and a shorter run since the useful part
ended early. Same Llama model, regularization and checkpoint rule as exp_4. Chosen by the user
from the options presented (settings = the assistant's stated lean, accepted as-is).

| Variable | Value | Changed from exp_4? |
|---|---|---|
| Architecture | Llama 8L / hidden 384 / 6 heads / MLP 1024 (14,290,560 params) | no |
| Dropout | `attention_dropout`=0.3 only | no |
| Epochs | **30** | **yes** (was 100) |
| Batch size | 16 (40 optimizer steps per epoch → 1,200 total) | no |
| Optimizer | AdamW, weight_decay=0.1, **peak** lr 3e-4 | no (peak lr unchanged) |
| LR schedule | linear warmup over first 5% (60 steps) to 3e-4, then cosine decay to 0; manual `LambdaLR`, stepped **per batch** | **yes** (was constant lr) |
| Loss | CrossEntropyLoss | no |
| Seed | `set_seed(0)` | no |
| Dev loss / checkpoint | teacher-forced dev loss over 399 positions; save lowest dev loss | no |
| Decoding | greedy (defaults) | no |
| Platform | Colab T4 via VS Code kernel | no |

Notebook edits: cell 10 (`import math`), cell 31 (epochs 100 → 30; schedule block after the
optimizer: `total_steps`, `warmup_steps`, `lr_lambda`, `scheduler = LambdaLR(...)`), cell 35
(`scheduler.step()` after each `optimizer.step()`; the per-epoch loss print now also shows the
current LR). The schedule math was checked in plain Python (no torch locally): peak 3e-4 reached
at step 60, ~1.6e-4 at the midpoint, ~0 at the final step; the printed end-of-epoch LRs should read
roughly 2.0e-4 (ep 1), 3.0e-4 (ep 2), 2.1e-4 (ep 12), 1.6e-4 (ep 15), 8.2e-05 (ep 20), 0 (ep 30).
**What we'd watch for:** the best dev loss vs. exp_4's 1.4533 and which epoch it lands on (with
decay it may move later than 12); whether dev loss still climbs after its minimum (overfitting
doesn't go away with a schedule; the dev-loss checkpoint is the safeguard); the printed LR matching
the expected values above; tail stats and visuals vs. exp_1-4; run time (~30 × ~11s train + dev
`generate` per epoch, roughly 9-10 min).

**Result**
- **The schedule worked as designed**: the printed LRs matched the predicted values (2.05e-4 at
  ep 1, 3.00e-4 at ep 2, 2.10e-4 at ep 12, 1.62e-4 at ep 15, 8.23e-05 at ep 20, 0 at ep 30). No
  errors. ~10-11s per training epoch, ~7s per dev batch, so the run is roughly a third of exp_4's.
- **Best dev loss 1.4652 at epoch 12** — essentially the same as exp_4's 1.4533 (slightly *worse* by
  0.012; with a single seed and no noise estimate we can't call that a real difference). The
  schedule did **not** lower the best dev loss, and the best epoch did not move later (still 12).
- **It did make overfitting far gentler.** Dev loss stays flat near its minimum for several epochs
  (1.465-1.481 across epochs 12-17) and ends at 1.620 at epoch 30, vs. exp_4's 2.83 at epoch 100.
  Training loss ends at 0.747 (dropout on) vs. exp_4's 0.078, i.e. no heavy memorization.

  | Epoch | 1 | 6 | 12 (best) | 20 | 25 | 30 |
  |---|---|---|---|---|---|---|
  | LR printed | 2.05e-4 | 2.82e-4 | 2.10e-4 | 8.23e-05 | 2.22e-05 | 0 |
  | Train loss (dropout on) | 3.68 | 1.58 | 1.26 | 0.96 | 0.79 | 0.747 |
  | Dev loss | 3.21 | 1.59 | **1.465** | 1.517 | 1.595 | 1.620 |

- Dev reconstruction accuracy: 0.30-0.40 after the first epochs, best 0.4010 (ep 15), 0.3759 at ep
  30; one odd outlier of 0.1793 at epoch 3. Still uninformative about quality.
- Generated `.txt` (`hw_4/exp_5_results.txt`, validated 80 × 400, values 0-166; prefixes identical
  to exp_4's). Tail stats:

  | Stat (last 160 tokens, 80 images) | exp_1 | exp_2 | exp_3 | exp_4 | exp_5 |
  |---|---|---|---|---|---|
  | Avg distinct colors per tail (prefix avg = 22.8) | 4.59 | 14.19 | 4.36 | 6.76 | 6.54 |
  | Distinct colors used across all tails | 87 | 154 | 88 | 119 | 114 |
  | Single-token tails | 7 | 0 | 9 | 0 | 0 |
  | Avg longest repeated run | 47.6 | 42.3 | 63.4 | 39.8 | 42.6 |
  | White share of tail tokens | 0.47 | 0.577 | 0.544 | 0.507 | 0.527 |
  | Distinct tails (of 80) | 76 | 80 | 76 | 80 | 80 |

  → Practically the same as exp_4 on every stat. Tail agreement with exp_4 is 0.596, the highest of
  any pair of runs, i.e. these two models produce similar completions.
- Takeaways: the warmup + cosine schedule is **not a lever for the best dev loss** here, but it
  reaches the same quality in ~30% of the epochs and doesn't overfit as badly after the minimum.
  The ceiling now looks set by something other than the LR recipe (regularization, data size,
  model choice, or greedy decoding) — not verified.
- Visual impression of the grid: _(add yours)_
- FID / PDR: **unknown** (TA-computed). First proxy read (cells 41-45, run after exp_5; first run
  of the proxy section — worked without errors, `pytorch-fid` installed and downloaded its weights
  on Colab):

  | Proxy metric | Value |
  |---|---|
  | Detector sanity: real dev judged Pokémon | 0.875 (wanted ~1) |
  | Detector sanity: corrupted dev rejected — flat color / noise / swapped tail | 0.988 / 0.988 / 0.875 |
  | **Proxy PDR, generated test images** | **0.762** (mean prob 0.716) |
  | Proxy PDR, blank-white-bottom control | 0.000 |
  | Proxy PDR, real dev ceiling | 0.875 |
  | **Proxy FID, generated vs 712 real** | **82.95** |
  | Proxy FID, blank-white-bottom control | 101.77 |
  | Proxy FID, real dev vs real train floor (n=80) | 63.76 |

  → Generated images sit about halfway between a blank bottom and real-vs-real on FID, and just
  under the detector's own real-image ceiling on PDR. Detector training loss ended at 0.068, so it
  is fairly overfit to the 632 training positives (it rejects 12.5% of held-out real images).
  These proxies are for comparing runs only; the TAs' FID/PDR use unknown reference sets and a
  classifier we can't see, so these numbers cannot be read against the 86/80/73 or 0.1/0.5/0.85
  thresholds.

---

<!-- Add new experiments below, following the same template: variable table with "Changed from
     exp_N?" column, then Result. Update the Summary table at the top. -->
