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
| exp_6 | exp_5 checkpoint (reproduced exactly) + **sampling** (6a/6b, 2 seeds each) | 14,290,560 | – | – | – | – | done — proxy FID 82.95 → 66-70, proxy PDR up under both detectors |

## Proxy scores (local `score_proxies.py`, all runs scored by the same detector)

Run: `hw_4/.venv/Scripts/python.exe score_proxies.py exp_1_results.txt ... exp_5_results.txt`
(~1 min on CPU; writes `proxy_scores.csv`). **Local stand-ins, not the TAs' numbers — compare
runs with each other only.** One detector is trained per invocation, so rows are comparable within
a table, but a re-run (or the notebook's own Colab-trained detector) gives somewhat different PDR.

| Run | Proxy PDR | Mean prob | Proxy FID | What it is |
|---|---|---|---|---|
| exp_1 | 0.287 | 0.296 | 81.99 | GPT-2 tiny, defaults |
| exp_2 | 0.450 | 0.438 | **71.04** | GPT-2 4.9M, 100 ep (heavily overfit ckpt) |
| exp_3 | 0.225 | 0.227 | 79.03 | exp_2 + dropout 0.3, dev-loss ckpt (ep 52) |
| exp_4 | 0.575 | 0.573 | 82.94 | Llama 14.3M, ckpt ep 12 |
| exp_5 | **0.637** | 0.607 | 82.95 | Llama + warmup/cosine, ckpt ep 12 |
| *blank-bottom control* | 0.000 | – | 101.77 | prefix + white bottom (bad end) |
| *real dev reference* | 0.850 (ceiling) | – | 63.76 (floor) | real dev vs real train, n=80 |

Detector sanity (this table's detector): real dev judged Pokémon 0.850; corrupted dev rejected
0.988 (flat) / 1.000 (noise) / 0.938 (swapped tail).

Reading it:
- **PDR and FID disagree.** PDR ranks the Llama runs best (exp_5 > exp_4 > exp_2 > exp_1 > exp_3),
  while FID ranks the overfit exp_2 best (71.04) and puts both Llama runs at ~83, near exp_1.
  No run is good on both at once, and the TAs require both to pass.
- exp_2's FID plausibly comes from its higher tail color variety (14.2 colors per tail vs ~4-7),
  which FID's feature statistics reward even if those tails are memorized fragments. Hypothesis,
  not verified.
- Consistency with the notebook's own run on exp_5: FID **82.95 in both** (and the blank control /
  floor, 101.77 / 63.76, are identical) — the FID side is deterministic and reproducible. PDR was
  **0.762 in the notebook vs 0.637 here** for the same file, with real-dev ceilings of 0.875 vs
  0.850: a different detector draw moves PDR by ~0.12, so PDR differences of that size between
  *tables* are noise; only compare rows within one invocation, and treat gaps of a few hundredths
  between rows as unreliable (detector seed variance not measured).
- FID noise at n=80 is not measured either; 82.94 vs 82.95 (exp_4 vs exp_5) is a tie, and
  differences of a few points elsewhere may be noise.

Setup notes: venv at `hw_4/.venv` (gitignored) with torch 2.14.1 (CPU), pytorch-fid 0.3.0,
datasets **5.1.0**. The notebook's pinned `datasets==3.3.2` fails to load data on Python 3.14
(`dill` pickling error: `Pickler._batch_setitems() takes 2 positional arguments but 3 were given`),
so the venv uses a newer `datasets`; the loaded data was checked to match (632/80/80 images, 167
colors, 240-token test prefixes). The notebook is unchanged (still pins 3.3.2 on Colab).

## Proxy scores v2 (stronger detectors; `score_proxies.py` rewritten; step 1 of the Strong push)

Changes vs. the v1 table above: seven corruption families (flat, noise, swap, mirror, shift,
rowshuffle, smear; cut always at row 12), horizontally flipped real images added as positives, a
corruption is rejected if it leaves an image unchanged (e.g. mirroring an all-white tail), and
**two detectors** averaged over **3 seeds each**: A = CNN trained on flat/noise/swap/mirror/shift;
B = MLP on the flattened image trained on flat/noise/swap/rowshuffle/smear. Run: 9 files, ~minutes
on CPU; log `proxy_v2_run.log`, table `proxy_scores_v2.csv`. v1 and v2 numbers are **not**
comparable (different detectors).

Detector check on held-out dev images (mean over 3 seeds; `*` = family in that detector's training set):

| | A (CNN) | B (MLP) |
|---|---|---|
| real dev accepted | 0.854 | 0.792 |
| flipped real dev accepted | 0.854 | 0.800 |
| rejected: flat | 0.963* | 0.996* |
| rejected: noise | 0.987* | 0.954* |
| rejected: swap | 0.942* | **0.229*** |
| rejected: mirror | 0.750* | 0.212 |
| rejected: shift | 0.912* | 0.383 |
| rejected: rowshuffle | 0.842 (unseen) | 0.804* |
| rejected: smear | 0.517 (unseen) | 0.987* |

| File | PDR A (mean (std)) | PDR B | Proxy FID |
|---|---|---|---|
| exp_1 greedy tiny GPT-2 | 0.354 (0.02) | 0.400 (0.07) | 81.99 |
| exp_2 GPT-2 4.9M, overfit | 0.333 (0.04) | **0.858** (0.06) | 71.04 |
| exp_3 GPT-2 + dropout, dev-loss ckpt | 0.242 (0.02) | 0.325 (0.06) | 79.03 |
| exp_4 Llama greedy | 0.667 (0.02) | 0.567 (0.07) | 82.94 |
| exp_5 Llama + cosine, greedy | 0.662 (0.09) | 0.533 (0.11) | 82.95 |
| exp_6a_s0 (T 0.8, top-k 50) | 0.742 (0.06) | 0.725 (0.07) | 70.48 |
| exp_6a_s1 | 0.817 (0.03) | 0.700 (0.05) | 67.11 |
| exp_6b_s0 (T 1.0, top-p 0.9) = exp_7 submission | 0.692 (0.06) | 0.721 (0.04) | 67.15 |
| exp_6b_s1 | 0.758 (0.06) | 0.750 (0.10) | 66.17 |
| *real dev ceiling / FID floor* | 0.854 | 0.792 | 63.76 |

Reading it (honest, mixed outcome):
- **The detectors did not get better at accepting real images**: ceilings 0.854 (A) and 0.792 (B),
  no higher than v1's 0.85-0.875, and flip augmentation did not raise them. With 632 real
  positives and an 80-image dev set the acceptance estimate itself is only good to ~±0.04, and
  per-seed PDR std is 0.02-0.11.
- **Detector A is a decent judge of seams/coherence**: it rejects swap 0.94 and also catches the
  *unseen* rowshuffle (0.84) and, partly, mirror/shift — but only 0.52 of the unseen smear family.
- **Detector B (MLP) is a weak judge of coherence**: it catches statistically obvious corruptions
  (flat, noise, smear, rowshuffle) but rejects only 0.23 of swapped tails even though it trained on
  swap, and 0.21-0.38 of mirror/shift. It is lenient toward real-looking-but-wrong tails.
- **A and B disagree sharply on exp_2** (A 0.33, B 0.86). Plausible reading (unverified): exp_2's
  overfit tails are fragments that look statistically like real tails but don't fit the prefix —
  exactly what B can't see and A partly can.
- **Sampling still wins, but the PDR evidence is weaker than v1 suggested.** FID: 82.95 → 66-70
  (unchanged, deterministic). PDR: greedy exp_5 0.662 (A) / 0.533 (B) vs sampled 0.69-0.82 (A,
  mean ~0.75) / 0.70-0.75 (B, mean ~0.72): about +0.09 under A (within seed noise for some files,
  e.g. 6b_s0 0.692 vs 0.662) and +0.19 under B. The two sampling settings are not separable.
- **Headroom is the problem**: sampled PDR under A (0.69-0.82) is already near A's ceiling (0.854);
  a detector with ceiling ~0.85 can't show whether a 0.85 PDR on the TAs' classifier is reachable,
  and best-of-n can't be demonstrated to help beyond the ceiling.

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

## exp_6 — Sampling vs. greedy decoding on the exp_5 checkpoint

No retraining. The proxy scores (see "Proxy scores" above) showed PDR and FID disagreeing: the
Llama runs (exp_4/5) have the best proxy PDR but a middling FID (~83), while exp_2's high tail
variety gave the best FID (71). Greedy decoding always takes the single most likely token, which
collapses completions toward white/repeated runs; sampling should add variety. Question: can
sampling raise FID quality without losing the coherence PDR rewards? Settings and seed counts
chosen by the user from the options presented (A+B with two seeds each, the assistant's lean).

| Variable | Value | Changed from exp_5? |
|---|---|---|
| Model / checkpoint | exp_5's recipe (Llama 14.3M, epoch 12, dev loss 1.4652). Planned as "load the saved checkpoint, no retraining"; the Colab session had expired, so the full run retrained it — and reproduced exp_5 exactly (see Result) | no |
| Decoding | `do_sample=True`; **6a**: temperature 0.8, top_k 50 · **6b**: temperature 1.0, top_p 0.9, top_k 0 (disabled) | **yes** (was greedy) |
| Seeds | 0 and 1 for each setting → 4 outputs: `6a_s0`, `6a_s1`, `6b_s0`, `6b_s1` | **yes** (greedy is deterministic) |
| Everything else | `max_length=400`, same test prefixes | no |

Notebook edits: cells 46-48 appended (markdown header; a generation cell that loads the checkpoint
and runs the four sampling configurations, checking each result is 80 × 400; a print cell that
emits each result between `===BEGIN_SUBMISSION:<label>===` markers). Cell 37's greedy path is
unchanged, so exp_5's output stays reproducible. `extract_submission.py` gained `--label` (and now
joins a cell's output chunks before searching); tested on synthetic notebooks (labeled block, a
block split across chunks, missing label, truncation) and on the real notebook's unlabeled block.

To extract and score (after running cells 47-48 and saving the notebook):
```
python extract_submission.py --label 6a_s0 --out exp_6a_s0_results.txt   # likewise 6a_s1, 6b_s0, 6b_s1
.venv/Scripts/python.exe score_proxies.py exp_5_results.txt exp_6a_s0_results.txt ... exp_6b_s1_results.txt
```

Flagged / not verified:
- Needs the Colab session from exp_5 still alive with its checkpoint on disk; cells 47-48 only
  need variables already in memory (`model`, `save_dir`, `test_dataloader`, `device`, `set_seed`).
- `top_k=0` is meant to disable top-k in `generate()` (its default is 50); not verified on the
  runtime's `transformers` version — if `generate` errors or warns about it, check this first.
- ~340 KB of printed output across four blocks; if the saved notebook truncates it, the extraction
  script reports a missing END marker (the data stays in memory, so re-printing is cheap).
- Proxy numbers for a sampled file depend on the random draw; the two seeds per setting are there
  to show how much that moves them (alongside the unmeasured detector-seed noise).

**What we'd watch for:** proxy FID below exp_5's 82.95 and proxy PDR not dropping below exp_5's
0.637 (same script run, same detector); how far apart the two seeds of one setting land; tail stats
(colors per tail, longest repeated run, white share) vs. greedy; the visual grids.

**Result**
- Run in browser Colab (the VS Code Colab extension ran out of GPU quota); the downloaded notebook
  is `hw_4/ECE592HW4_colab.ipynb` (all 49 cells, outputs included). The fresh session meant the
  model was **retrained** with the exp_5 recipe rather than loading a saved checkpoint — and it
  reproduced exp_5 **exactly**: same dev-loss trajectory (best 1.4652 at epoch 12, final train loss
  0.747), and the greedy output (`exp_6_greedy_results.txt`) is **byte-identical** to
  `exp_5_results.txt`. So exp_5 is the greedy baseline for this comparison. No errors; `top_k=0`
  caused no warnings. Sampling took ~8-15s per setting/seed.
- Extracted with `extract_submission.py --notebook ECE592HW4_colab.ipynb --label <label>` →
  `exp_6a_s0/6a_s1/6b_s0/6b_s1_results.txt` (all validated 80 × 400, prefixes identical to exp_5's).
  Scored with `score_proxies.py` twice (detector seeds 0 and 1) → `proxy_scores_exp6_seed{0,1}.csv`.

  | File | Proxy FID | Proxy PDR (detector seed 0) | Proxy PDR (detector seed 1) |
  |---|---|---|---|
  | exp_5 (greedy) | 82.95 | 0.637 | 0.775 |
  | 6a_s0 (T 0.8, top-k 50) | 70.48 | 0.738 | 0.875 |
  | 6a_s1 | 67.11 | 0.850 | 0.938 |
  | 6b_s0 (T 1.0, top-p 0.9) | 67.15 | 0.800 | 0.875 |
  | 6b_s1 | 66.17 | 0.787 | 0.863 |
  | *real dev reference* | 63.76 (floor) | 0.850 (ceiling) | 0.925 (ceiling) |
  | *blank-bottom control* | 101.77 | 0.000 | 0.013 |

  Tail stats (last 160 tokens, 80 images; prefix avg = 22.8 colors):

  | File | Colors/tail | Distinct colors | Single-token tails | Avg longest run | White share |
  |---|---|---|---|---|---|
  | exp_5 (greedy) | 6.54 | 114 | 0 | 42.6 | 0.527 |
  | 6a_s0 | 16.06 | 156 | 0 | 24.5 | 0.491 |
  | 6a_s1 | 16.09 | 160 | 0 | 27.8 | 0.504 |
  | 6b_s0 | 18.62 | 161 | 0 | 27.9 | 0.492 |
  | 6b_s1 | 18.94 | 161 | 0 | 28.8 | 0.507 |

  Tail agreement between the two seeds of one setting: 0.505 (6a), 0.469 (6b); sampled vs greedy:
  0.52-0.56 — i.e. the sampled completions genuinely differ run to run.
- **Sampling beat greedy decisively on both proxies, under both detector seeds.** Proxy FID fell
  from 82.95 to 66-70, to within ~3 of the real-vs-real floor (63.76) — a 12-17 point gain, far
  larger than the seed-to-seed spread. Proxy PDR rose in every sampled file under both detectors
  (e.g. 0.637 → 0.74-0.85 with detector seed 0; 0.775 → 0.86-0.94 with seed 1), reaching the
  detector's own real-image ceiling. Tails have 16-19 colors (vs 6.5), long repeated runs shrank
  from 42.6 to ~25-29, white share dropped slightly. This supports the hypothesis from exp_3/4:
  greedy decoding's collapse toward white/repeats was the main bottleneck, not the model.
- **Noise (now partly measured).** FID is deterministic given the file (identical across detector
  seeds); across sampling seeds it varied by 3.4 points for 6a (70.48 vs 67.11) and 1.0 for 6b
  (67.15 vs 66.17). The detector seed alone moves PDR by ~0.1-0.14 (exp_5: 0.637 vs 0.775) and
  changes the detector's real-image ceiling (0.850 vs 0.925), but the *ordering* greedy < sampled
  held under both. **6a vs 6b cannot be separated** with this evidence (FID within noise, PDR
  ordering flips between rows/detectors); 6b looks marginally more stable on FID.
- Caveats: the proxies can't see everything — in particular local speckle/noisy pixels from
  sampling at temperature 1.0 may not be penalized, so the generated grids still need a visual
  look (blank for now: _add your impression_). Proxy thresholds don't transfer to the TAs' FID/PDR;
  the notebook-side proxy cell (45) on this Colab run gave PDR 0.700 for the greedy file (a third
  detector draw: ceiling 0.863) and FID 82.95, matching the local script's FID exactly.
- Open decision (not made): the submitted `.txt` still comes from the greedy path (cell 37/40).
  Adopting sampling means changing which cell produces the final file, picking a setting, and
  fixing a seed *in advance* rather than choosing the best-scoring seed (that would overfit a noisy
  proxy). FID / PDR from the TAs: **unknown**.

---

## exp_7 — Submission run: sampling in the main inference cell

Goal: make the submitted `.txt` come from the notebook's main path (cell 37 → cell 40 printout)
using the sampling that exp_6 showed beats greedy on both proxies, rather than from a side cell.
Decision (user, on a tight schedule): setting and seed fixed **in advance**, not picked by proxy
score — **6b** (temperature 1.0, top-p 0.9, top_k disabled) with `set_seed(0)`. 6a vs 6b was a
statistical tie in exp_6; 6b was marginally more stable on FID.

| Variable | Value | Changed from exp_5/6 greedy path? |
|---|---|---|
| Model / training | Llama 14.3M, exp_5 recipe (30 ep, warmup+cosine, dev-loss ckpt) — retrained by run-all, reproduced exp_5 exactly in exp_6 | no |
| Cell 37 decoding | `set_seed(0)` then `generate(..., do_sample=True, temperature=1.0, top_p=0.9, top_k=0)` | **yes** (was greedy) |
| Everything else | cells 0-36, 38-48 unchanged (dev eval in cell 35 still greedy; proxy and comparison-sampling cells kept) | no |

Expected check: cell 40's printed `.txt` should be byte-identical to `exp_6b_s0_results.txt` (same
checkpoint, same seed, same calls). If it isn't, sampling isn't fully reproducible on that GPU run
— worth knowing before submitting. Submitted `.txt` = cell 40's extraction (`extract_submission.py`
with no `--label`); the zip is `HW4_<unityid>.zip` (assumed `HW4_rrkulka3.zip`) with the executed
notebook + that single `.txt`.

**Result**
- Full run-all in browser Colab; executed notebook saved as `hw_4/ECE592HW4_colab.ipynb` (49 cells,
  no errors). Training reproduced exp_5/6 again: best dev loss 1.4652 at epoch 12, checkpoint saved
  there. Cell 37 ran with the sampling edit.
- **Cell 40's extracted `.txt` is byte-identical to `exp_6b_s0_results.txt`** — sampling was fully
  reproducible, so the file submitted is exactly the file scored in exp_6 (proxy FID 67.15; proxy
  PDR 0.800 / 0.875 under detector seeds 0 / 1; tail stats: 18.6 colors/tail, longest run 27.9).
- Packaged `hw_4/submission/HW4_rrkulka3.zip` (verified by reopening): exactly 2 files —
  `ECE592HW4_rrkulka3.ipynb` (the executed notebook; 14 code cells with outputs; scanned for
  emails/tokens, none) and `reconstructed_results.txt` (80 lines × 400 ints, values 0-166).
  Local `ECE592HW4_rrkulka3.ipynb` (no outputs, pre-run) was left untouched; the zip's notebook is
  a copy of the executed Colab one under the assignment's name.
- Not verified / assumptions: the UnityID in the zip name (`rrkulka3`); that the TAs accept the
  notebook's name and extra cells (proxy + comparison-sampling sections); FID/PDR from the TAs:
  **unknown** — the proxies can't confirm any tier.

---

## exp_8 — Horizontal-flip augmentation for the generator (Strong push, step 2)

Overfitting is the generator's bottleneck (dev loss bottoms at epoch 12 then rises). Mirroring the
training sprites gives more variety without new data. Chosen by the user from the options presented
(the assistant's leans): **random flip at probability 0.5 each time an image is drawn** (not a
duplicated dataset), keeping compute and the schedule identical to exp_5/exp_7 so any change is
attributable to the augmentation; plus a **copy of the best checkpoint to Google Drive** so later
sessions (step 3 development) can reload it instead of retraining.

| Variable | Value | Changed from exp_7? |
|---|---|---|
| Model / schedule / epochs / lr / dropout | Llama 14.3M, 30 epochs, warmup + cosine, peak 3e-4, attention dropout 0.3, weight decay 0.1 | no |
| Training data | 632 real train images; each draw mirrored left-right (each of the 20 rows reversed) with p = 0.5 | **yes** (was never flipped) |
| Dev / dev-loss / test loaders | unflipped (dev loss stays comparable with exp_5's 1.4652) | no |
| Checkpoint rule | lowest dev loss | no |
| Decoding in cell 37 | sampling: T 1.0, top-p 0.9, top_k 0, `set_seed(0)` | no |
| Checkpoint copy | `best_model.pth` → `/content/drive/MyDrive/ece592_hw4/best_model_flip_aug.pth` (needs Drive mounted by cell 5; failures only print a message) | **yes** (new) |

Notebook edits (cell numbers unchanged): cell 17 (`import random`; `hflip_prob` argument on
`PixelSequenceDataset`, used only in `train` mode), cell 19 (`hflip_prob=0.5` on the *training*
dataset only), cell 35 (guarded Drive copy after the training loop). Checked locally with real torch:
p=0 identical to the original, p=1 equals the mirrored image (inputs and labels), p=0.5 flips ~46%
of draws and is reproducible with a fixed seed, dev/test modes unaffected, Drive block falls back to a
message when Drive isn't mounted.

Flagged: the augmentation makes sampling order differ from exp_5 even with the same seed, so exact
reproduction of exp_5 is not expected; sprites may have a favored facing direction, so mirroring could
hurt — dev loss decides. A single seed means small dev-loss differences from 1.4652 are not
meaningful.

**What we'd watch for:** best dev loss clearly below 1.4652 (and which epoch it lands on; flip
augmentation may push it later); a flatter or later-rising dev-loss curve; then the usual extraction
and proxy scoring of cell 40's output (compare with exp_7 / `exp_6b_s0`: proxy FID 67.15, PDR A 0.692
/ B 0.721 under v2). If dev loss does not improve, keep the exp_5 recipe for step 3.

**Result**
- Run in browser Colab, executed notebook = `hw_4/ECE592HW4_colab.ipynb` (49 cells, no errors, step-2
  code confirmed present). Drive copy worked (`best_model_flip_aug.pth` saved).
- **Best dev loss 1.4685 at epoch 18** vs exp_5/7's **1.4652 at epoch 12** — a tie (0.003 is
  meaningless at one seed). What changed is the curve's shape: the best epoch moved later (12 → 18),
  and overfitting is gentler — dev loss at epoch 30 is **1.5095** vs 1.6202 before; final training
  loss (dropout on) 1.02 vs 0.75. So flip augmentation slows memorization but did not improve the
  best model. Dev accuracy: best 0.3885 (ep 17), 0.3805 at ep 30 (uninformative as usual).
- Outputs extracted: `exp_8_results.txt` (cell 40; byte-identical to `exp_8_6b_s0_results.txt`, same
  setting/seed) and `exp_8_6a_s0/6a_s1/6b_s0/6b_s1_results.txt` (cells 47-48). Scored together with the
  four exp_6 files (old model) by the v2 script (3 detector seeds each): `proxy_scores_exp8.csv`,
  `proxy_exp8_run.log`.

  | File | PDR A | PDR B | Proxy FID | Colors/tail | Longest run |
  |---|---|---|---|---|---|
  | old 6a_s0 | 0.742 | 0.725 | 70.48 | 16.06 | 24.5 |
  | old 6a_s1 | 0.817 | 0.700 | 67.11 | 16.09 | 27.8 |
  | old 6b_s0 (exp_7 submission) | 0.692 | 0.721 | 67.15 | 18.62 | 27.9 |
  | old 6b_s1 | 0.758 | 0.750 | 66.17 | 18.94 | 28.8 |
  | **new 6a_s0** | 0.808 | 0.704 | 67.84 | 14.44 | 27.1 |
  | **new 6a_s1** | 0.787 | 0.792 | 69.58 | 15.05 | 29.0 |
  | **new 6b_s0** (= exp_8_results.txt) | 0.808 | 0.812 | **65.40** | 16.07 | 28.7 |
  | **new 6b_s1** | 0.779 | 0.775 | 65.83 | 16.26 | 25.9 |
  | *real dev ceiling / floor* | 0.854 | 0.792 | 63.76 | – | – |

  Means over the four files: **old** FID 67.73, PDR A 0.752, PDR B 0.724; **new** FID 67.16, PDR A
  0.795, PDR B 0.771. For the pre-committed setting (6b, seed 0): old 67.15 / 0.692 / 0.721 vs new
  65.40 / 0.808 / 0.812.
- **Reading:** the new model is slightly better on all three proxies on average (FID −0.6, PDR A
  +0.04, PDR B +0.05) and for 6b/seed 0, but every difference is within the noise we've measured
  (sampling-seed FID spread 1-3 points, per-file PDR std 0.04-0.08, so a mean of four files is
  uncertain by roughly ±0.03-0.04). Tails have slightly *fewer* colors (14-16 vs 16-19). Honest
  summary: flip augmentation is, at best, a small win; a tie on dev loss with a gentler curve.
- **Both models are now near what our proxies can measure**: FID 65-68 vs real-vs-real floor 63.76;
  PDR A 0.78-0.81 vs detector ceiling 0.854 — little headroom left to show further gains.
- **Decision (user): submit this run (exp_8), and skip step 3 (best-of-n)** — little proxy headroom
  and a risk of fitting our own detector. The final zip `hw_4/submission/HW4_rrkulka3.zip` was
  rebuilt from this run: `ECE592HW4_rrkulka3.ipynb` (copy of the executed `ECE592HW4_colab.ipynb`;
  49 cells, 14 code cells with outputs, no errors, no unexecuted cells, no emails/tokens) +
  `reconstructed_results.txt` (= `exp_8_results.txt`, 80 × 400, values 0-166, verified equal to the
  notebook's own cell-40 output). The previous exp_7 zip is kept as
  `submission/HW4_rrkulka3_exp7_backup.zip`. Submission config: Llama 14.3M, flip augmentation,
  warmup+cosine, dev-loss checkpoint (epoch 18, dev loss 1.4685), sampling T 1.0 / top-p 0.9 / seed 0.
  Proxy read for this file: FID 65.40, PDR A 0.808 / B 0.812 (v2 detectors). TA FID/PDR: **unknown**.

---

<!-- Add new experiments below, following the same template: variable table with "Changed from
     exp_N?" column, then Result. Update the Summary table at the top. -->
