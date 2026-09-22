# ECE 592 HW2 — Report

**Notebook:** `ECE592HW2_rrkulka3.ipynb`
**Deadline:** 27 Sept 2026

This report is written alongside implementation, phase by phase, and will be the basis
for the final report submitted via Overleaf.

---

## Phase 0 — Compute Strategy

**Decision date:** 2026-09-17

### Platform: Kaggle

A T4 GPU was available via Google Colab, but Colab's free-tier session limits
(~12h hard cap, disconnects after ~90 min of browser inactivity) would become a
bottleneck once training moves past the Simple/Medium tiers into Strong (10–12h
reference training time) and Boss (20h+). Kaggle offers a 30 GPU-hr/week quota and
supports background kernel execution ("Save & Run All") that continues running without
the browser open, for up to roughly 9–12 continuous hours. The assignment PDF also
explicitly recommends Kaggle for the Strong/Boss tiers.

Note on GPU access mechanics: a GPU attached to a hosted notebook (Colab or Kaggle) is
only reachable by processes running *inside that same VM*. A script run on a local
machine, disconnected from the hosted session, cannot use that GPU. A script run via
`!python train.py` (or a background process) from *within* the hosted VM can, since it's
just another process on the same machine as the notebook kernel.

### Resumability: Required from the start

Rather than retrofitting checkpoint/resume logic once we hit a session limit on the
Strong/Boss tiers, the training script will be designed with checkpoint save/resume
built in from the first version.

### Invocation: Standalone script, deferred until after Phase 1/2

Training will eventually be invoked via a standalone `train.py` (parameterized with
flags like `--exp_name`, `--epochs`, `--lr`) rather than run interactively in notebook
cells, once past the Simple tier. However, the notebook's data pipeline is not yet
verified (Phase 1) and known bugs haven't been fixed (Phase 2), so writing the script
now would risk extracting unverified/buggy logic into its structure. The script will be
extracted only after a working smoke-test pipeline (data → model → loss → backward →
checkpoint save → reload → predict → `submission.csv`) is confirmed inside the notebook.
Until then, Phase 1/2 work continues interactively in the notebook.

**Next actionable step:** set up a Kaggle account/notebook with GPU enabled and the
food-11 dataset attached, ahead of Phase 1 (data acquisition).

### Addendum: local editing + Kaggle sync tooling

To support editing the notebook locally in VS Code while running/training on Kaggle's
GPU, a small sync layer was added:

- `kernel-metadata.json` — static Kaggle kernel config (id `rithikkulkarni1/ece592-hw2-food11`,
  GPU enabled, dataset sources left empty until Phase 1 decides on data hosting).
- `kaggle_sync.py` — wraps the official `kaggle` CLI with `push` / `status` / `pull` / `run`
  subcommands. `run` pushes the notebook, polls until the Kaggle run finishes, then pulls
  the output notebook + files back locally.
- Kaggle API credentials live in `~/.kaggle/kaggle.json` (not committed; `.gitignore` was
  added to keep `.env`, `kaggle.json`, and other local-only files out of git).

Considered reverting to Colab entirely once realizing the submission only requires the
final `.ipynb` + report — but that reasoning doesn't actually change the platform
tradeoff (Kaggle vs. Colab affects *where GPU-hours are spent*, not what gets submitted).
More importantly, Colab has no official CLI/API for a "push from terminal, run remotely,
pull results" loop the way Kaggle does; an equivalent would require unofficial
SSH-tunneling hacks. Decision: **stay with the Kaggle sync workflow.**

---

## Phase 1 — Get the Data Working

**Completed:** 2026-09-17

- The Dropbox link in cell 5 is alive (verified via `curl -IL`, and by an actual full
  download — 1,163,525,370 bytes, matching the expected ~1.1GB food-11 zip). The
  Google Drive fallback link is dead (404), but wasn't needed. No Kaggle-dataset fallback
  was required either.
- Cell 5 originally used `!wget`, which isn't available on this local Windows setup
  (no shell has it on PATH by default). Installing it via `conda-forge` hung indefinitely
  on environment solving and was abandoned. Replaced with `!curl -L -o food11.zip ...`,
  which is natively available on Windows 10+, Git Bash, and Kaggle — a portable
  equivalent with no behavior change.
- Data downloaded and unzipped locally (not on Kaggle) for fast iteration, since Phase 1
  verification doesn't need a GPU. Disk space was tight (11GB free before, ~7.8GB after
  unzip) but sufficient; the zip was deleted immediately after extraction to reclaim ~1.1GB.
- Verified directory structure and counts exactly match the PDF's stated split:
  - `train/`: 10,000 `.jpg` files
  - `valid/`: 3,643 `.jpg` files
  - `test/`: 3,000 `.jpg` files
- Spot-checked 3 random images per split (9 total) for corruption via PIL's `verify()` —
  all opened cleanly, sizes mostly 512×512 (some 512×384), consistent with a raw
  variable-resolution source dataset that gets resized by `test_tfm`/`train_tfm`.

**Done when (met):** `FoodDataset("./train")`, `("./valid")`, `("./test")` all instantiate
without error and `len()` matches expected counts.

---

## Phase 2 — Fix Known Bugs / Sanity-Check the Pipeline

**Completed:** 2026-09-17

### Bug found beyond the two ROADMAP called out

**Label parsing breaks silently on Windows (cell 14, `FoodDataset.__getitem__`).**
The original code derived the class label via `fname.split("/")[-1].split("_")[0]`,
assuming Unix-style forward-slash paths. `os.path.join` produces backslash paths on
Windows (`./train\0_0.jpg`), so `split("/")` doesn't isolate the filename — the `int()`
parse then throws, and the bare `except: label = -1` swallows it silently. Practical
effect if left unfixed: every training/validation label would silently become `-1` when
run locally on Windows, with no error raised — the model would train on corrupted
supervision with no obvious signal that anything was wrong. This wouldn't have surfaced
on Colab/Kaggle (both Linux), which is presumably why the sample code never hit it.

**Fix:** `os.path.basename(fname)` instead of `fname.split("/")[-1]` — correctly handles
both path separators regardless of OS. Verified against the real downloaded data: train
labels now parse to the expected 0–10 range (11 classes) with zero `-1` values; test
labels (which have no `<label>_` prefix) correctly fall through to `-1`.

### ROADMAP's two known bugs

1. **Logging bug (cell 22 — the training loop's log block).** Originally opened the log
   file inside a `with open(...) as f:`-less `with open(...):` block and then called
   `print(...)`, which only writes to stdout — the file handle was never used, so nothing
   was ever written to `{_exp_name}_log.txt`. Fixed by capturing the log line into a
   variable, printing it, and explicitly `f.write()`-ing it inside a proper
   `with open(...) as f:` block.
2. **`_exp_name` discipline (cell 8).** Was hardcoded to `"sample"`. Adopted the naming
   convention `<tier>_<variant>` (e.g. `simple_v1`, `medium_aug1`, `strong_resnet18`) so
   checkpoint files and t-SNE loading never collide across experiments. Set to
   `smoke_test_v1` for the Phase 2 smoke test specifically, so it's obviously distinct
   from the real Simple-tier baseline run that comes next in Phase 4a.

### Smoke test

Ran the notebook's actual (post-fix) `FoodDataset`/`Classifier`/training-loop logic
end-to-end as a standalone script, locally on CPU, against a small random subset (200
train / 100 valid / 50 test images) for 2 epochs — not by editing the real training
cells to use a subset, since that would need reverting before Phase 4's real runs.

Result: training loop ran, loss moved (2.43 → 2.06 on the tiny 2-class-imbalanced
subset — not meaningful accuracy, just confirms gradients flow), checkpoint saved,
log file written correctly (confirming the logging fix), checkpoint reloaded, predictions
generated, and `submission.csv`-equivalent output had the correct shape/format
(`Id` zero-padded to 4 digits, `Category` column). Scratch artifacts from this test were
deleted afterward — they aren't part of the submission.

**Done when (met):** full pipeline pass (train → save checkpoint → load checkpoint →
predict → write `submission.csv`) completes without errors.

---

## Phase 3 — Q1: Implement `train_tfm` (Augmentation)

**Completed:** 2026-09-17

### What Q1 requires

Finish `train_tfm` so that applying it to the same image multiple times produces 5+
*visibly* different outputs, and explain (below) why each transform helps. Per the PDF,
`train_tfm` exists in two places that are allowed to differ: cell 12 (the one actually
wired into the training `DataLoader`) and cell 30 (a copy for the report/Gradescope).
Both were kept identical here for simplicity.

### Decisions made

- **Image size: kept at 128×128.** `Classifier.fc` (cell 17) hardcodes
  `nn.Linear(512*4*4, ...)`, which assumes a 128×128 input survives 5 stride-2 maxpools
  down to exactly 4×4. Changing resolution now would mean also updating that layer (or
  switching to `AdaptiveAvgPool2d`) before we even know if this architecture survives
  into Phase 4c (Strong tier swaps in a torchvision/timm model). Deferred.
- **Augmentation stack: "moderate."** `RandomResizedCrop(128, scale=(0.7,1.0))` →
  `RandomHorizontalFlip()` → `RandomRotation(15)` → `ColorJitter(brightness=0.2,
  contrast=0.2, saturation=0.2)` → `ToTensor()`. `RandomResizedCrop` replaces the plain
  `Resize()` since it resizes *and* randomly crops in one step.

### Why each transform helps (for the report)

- **`RandomResizedCrop`** — randomly crops a sub-region (70–100% of the image area) before
  resizing to 128×128. Teaches the model that a food class shouldn't depend on the object
  being perfectly centered or filling the whole frame — counters overfitting to exact
  framing/composition of the training photos.
- **`RandomHorizontalFlip`** — mirrors the image left-right with 50% probability. Food
  photos have no canonical "handedness," so this teaches left-right invariance for free
  and effectively doubles the visual variety of each image across epochs.
- **`RandomRotation(15)`** — rotates up to ±15°. Counters overfitting to the exact camera
  angle/tilt the photo happened to be taken at; small angles avoid distorting food
  identity (unlike, say, 90° rotations which could make some dishes look unnatural).
- **`ColorJitter`** — randomly perturbs brightness/contrast/saturation. Counters
  overfitting to the specific lighting conditions/white balance of individual photos,
  which matters a lot for a food dataset where lighting varies a lot across sources.
- We're not doing a random combination of these 4 transforms. Instead, it runs all 4 transforms every single call, in the same fixed order. Nothing gets shuffled in this order, but what's actually random is the parameters that each individual transform draws internally every time it's called.

### Verification

Added a new cell (13) that loads one real training image, applies `train_tfm` to it 5
times, and plots the original + 5 augmented versions side by side, saving
`q1_augmentation_grid.png` — this is the image to paste into the Overleaf report for Q1.

Ran this logic locally (CPU) against the real downloaded data. Visual check: the 5
outputs show clearly different rotation angles, crop framing/zoom, and flip state.
Quantitative check: pairwise mean-absolute-difference between all 5 output tensors
ranged 0.07–0.20 (all comfortably distinct, no near-duplicate pairs) — confirms the
"5+ different results" requirement isn't just assumed, it's measured.

**Done when (met):** visual grid showing 5+ distinct augmented versions of the same
image exists, and `train_tfm` is wired into the training `DataLoader` (cell 21 already
passes `tfm=train_tfm` to the train `FoodDataset`, unchanged from the sample code).

*Note: the augmentation parameters were later widened (crop scale 0.7-1.0 → 0.5-1.0,
rotation ±15° → ±35°, color jitter 0.2 → 0.4 + added hue=0.1) after the first grid
looked too subtle visually. The numbers above reflect the final widened version
(pairwise diffs 0.12–0.30); see `hw_2_report_latex.txt` for the final write-up.*

---

## Phase 4 — Baseline Progression

### exp_1 (first real training run)

**Completed:** 2026-09-17. Full configuration logged in `ROADMAP.md`'s Experiment Log.

First full run on real data (10,000 train / 3,643 valid images), deliberately using
sample-code defaults (`n_epochs=8`, `patience=5`, batch size 64, Adam lr=0.0003) plus
Phase 3's `train_tfm` already wired in — no tuning yet, just to see where we start from.

**Infrastructure hurdles hit getting this to actually run on Kaggle** (worth remembering,
since they'll bite again on future experiments if forgotten):

1. **`kernel-metadata.json`'s `enable_gpu`/`enable_internet` must be the strings
   `"true"`/`"false"`, not JSON booleans.** They were booleans initially; the CLI
   silently ignored them (no error, just defaulted to disabled) rather than failing
   loudly. Fixed, and added `machine_shape: "NvidiaTeslaT4"` to be explicit.
2. **Even after that fix, GPU/internet still didn't attach** — root cause turned out to
   be that the Kaggle account needed **phone number verification** (a platform-level
   requirement, unrelated to any file in this repo). Runs silently fall back to
   CPU-only/no-internet without it, with no explicit error pointing at the real cause.
   Two failed kernel versions (v1, v2) both failed identically inside ~60 seconds of
   real execution (`curl: Could not resolve host`, `torch.cuda.is_available() == False`)
   before this was identified. After verifying, v3 ran correctly on GPU.
3. **This CLI version (`kaggle` 2.2.4, built on `kagglesdk`) uses a different auth
   scheme** than the classic `kaggle.json` (username+key) — it wants a raw token string
   in `~/.kaggle/access_token`, matching the newer `KGAT_...`-style token format.
4. **`kaggle kernels output` does not include the executed notebook** (`__notebook__.ipynb`
   / `__results__.html`), only files the kernel explicitly wrote (checkpoint, log, csv).
   `kaggle kernels pull` fetches the notebook *source* instead, but without outputs. As
   of now there's no confirmed CLI path to the executed-with-outputs notebook — it may
   only be downloadable from the kernel's page in-browser. **This needs to be resolved
   before final submission**, since the assignment requires the notebook with visible
   run outputs.
5. **Kaggle's reported kernel status lags reality substantially.** `RUNNING` covers both
   genuine queueing/provisioning time *and* actual execution — the first two failed
   attempts each showed `RUNNING` for 50+ minutes before flipping to `ERROR`, despite
   only ~60 seconds of real execution once the container actually started. There's no
   way to distinguish "queued" from "executing" via the CLI's status command alone.

**Results** (from `exp_1_log.txt`, pulled via `kaggle_sync.py`):

| Epoch | Train loss | Train acc | Valid loss | Valid acc |
|---|---|---|---|---|
| 1 | 2.131 | 0.248 | 1.937 | 0.322 (best) |
| 2 | 1.981 | 0.302 | 1.959 | 0.318 |
| 3 | 1.870 | 0.350 | 1.779 | 0.369 (best) |
| 4 | 1.757 | 0.389 | 1.780 | 0.383 (best) |
| 5 | 1.667 | 0.423 | 1.587 | 0.451 (best) |
| 6 | 1.582 | 0.454 | 1.643 | 0.430 |
| 7 | 1.530 | 0.469 | 1.474 | 0.496 (best) |
| 8 | — | — | 1.429 | 0.494 |

**Best validation accuracy: 0.496** (epoch 7 checkpoint). `submission.csv` generated
correctly (3000 rows, `Id`/`Category`, zero-padded IDs). No overfitting gap yet —
train/valid loss tracked closely throughout, and both were still improving at epoch 8
with no plateau, suggesting more epochs would keep helping.

**Below both the Simple (0.637) and Medium (0.700) tier thresholds** — expected, since
`n_epochs=8` is quite short and the (moderately aggressive) augmentation stack trades
faster early convergence for better eventual generalization. This is the baseline
starting point; next step is deciding how much to increase `n_epochs`/`patience` for a
follow-up run aimed at clearing Simple/Medium.

### exp_2 (bumped n_epochs/patience)

**Completed:** 2026-09-17. Same config as `exp_1` except `n_epochs` 8→30 and
`patience` 5→8 (see `ROADMAP.md` Experiment Log for the full diff).

**Results** (from `exp_2_log.txt`):

| Epoch | Valid loss | Valid acc | Epoch | Valid loss | Valid acc |
|---|---|---|---|---|---|
| 1 | 1.937 | 0.322 (best) | 16 | 1.471 | 0.525 |
| 2 | 1.959 | 0.318 | 17 | 1.227 | 0.581 |
| 3 | 1.779 | 0.369 (best) | 18 | 1.193 | 0.596 (best) |
| 4 | 1.780 | 0.383 (best) | 19 | 1.233 | 0.597 (best) |
| 5 | 1.587 | 0.451 (best) | 20 | 1.112 | 0.618 (best) |
| 6 | 1.643 | 0.430 | 21 | 1.175 | 0.617 |
| 7 | 1.474 | 0.496 (best) | 22 | 1.076 | 0.634 (best) |
| 8 | 1.429 | 0.494 | 23 | 1.034 | 0.650 (best) |
| 9 | 1.481 | 0.496 (best) | 24 | 1.066 | 0.647 |
| 10 | 1.267 | 0.554 (best) | 25 | 1.013 | **0.658 (best)** |
| 11 | 1.407 | 0.524 | 26 | 1.148 | 0.609 |
| 12 | 1.543 | 0.489 | 27 | 1.032 | 0.655 |
| 13 | 1.352 | 0.538 | 28 | 1.141 | 0.624 |
| 14 | 1.386 | 0.541 | 29 | 1.018 | 0.654 |
| 15 | 1.222 | 0.591 (best) | 30 | 1.109 | 0.643 |

**Best validation accuracy: 0.658** (epoch 25, `exp_2_best.ckpt`). Training ran the
full 30 epochs — early stopping never triggered (patience=8; the longest streak
without a new best was 5 epochs, epochs 26-30). `submission.csv` generated correctly
(3000 rows).

**Clears the Simple tier (0.637).** Does not clear Medium (0.700). By epoch 30, train
accuracy (0.715) had pulled noticeably ahead of the epoch-30 valid accuracy (0.643) —
a widening train/valid gap consistent with mild overfitting setting in during the last
few epochs, which is exactly why the epoch-25 checkpoint (not epoch-30) ended up as
"best" and is what `submission.csv`'s predictions actually came from.

The run also hit an expected, harmless error at the very end: the Q2 (t-SNE) cells
(cells 32-33) still have `index = ...` and `target_label = ...` as literal placeholders
(Phase 5 isn't implemented yet), so `model.cnn[:index]` throws
`TypeError: slice indices must be integers` when Kaggle tries to execute that cell.
This happens *after* training, checkpointing, and `submission.csv` generation already
completed successfully — it doesn't invalidate any of exp_2's training results.

**Takeaway for Medium tier:** the trend suggests more epochs alone likely won't cleanly
clear 0.700 given the overfitting signs already appearing by epoch 30 — Medium will
probably need either regularization (the augmentation is already fairly aggressive) or
a longer training budget with a mechanism to combat the emerging overfitting, rather
than just "more of the same." To evaluate later once Strong tier's architecture change
is in play.

### exp_3 (Strong tier: ResNet18 sanity run)

**Completed:** 2026-09-17. Swapped `Classifier` for `build_model()` (ResNet18,
`weights=None`, final FC → 11 classes). Short sanity run (`n_epochs=10`,
`patience=5`) to confirm the new architecture trains correctly on Kaggle before
committing to the PDF's 10-12hr reference budget for Strong tier. Everything else
(augmentation, image size, batch size, optimizer) unchanged from `exp_2`.

**Results** (from `exp_3_log.txt`):

| Epoch | Valid loss | Valid acc |
|---|---|---|
| 1 | 1.990 | 0.287 (best) |
| 2 | 2.163 | 0.308 (best) |
| 3 | 1.687 | 0.408 (best) |
| 4 | 1.620 | 0.445 (best) |
| 5 | 1.545 | 0.483 (best) |
| 6 | 1.580 | 0.455 |
| 7 | 1.467 | 0.498 (best) |
| 8 | 1.485 | 0.497 |
| 9 | 1.525 | 0.494 |
| 10 | 1.296 | **0.551 (best)** |

**Best validation accuracy: 0.551 (epoch 10 — the last epoch run).** For rough context,
this roughly matches `exp_2`'s (the from-scratch CNN's) epoch-10 point (0.554) — but the
important signal isn't the raw number, it's the trajectory: `exp_3` hit a **new best on
the very last epoch**, with no sign of plateauing, whereas by this point checking
`exp_2` alone wouldn't have told us much either way. The sanity check's actual goal —
confirming ResNet18 trains correctly end-to-end on Kaggle at 128×128 with no pretrained
weights — succeeded cleanly: loss decreased, accuracy climbed, checkpointing/logging
worked, `submission.csv` generated correctly (3000 rows).

Hit the expected, harmless error at the end: `AttributeError: 'ResNet' object has no
attribute 'cnn'` — the t-SNE cell's `model.cnn[:index]` slicing only ever worked for the
old `Classifier`'s flat `Sequential`; ResNet18 exposes `conv1`, `bn1`, `relu`, `maxpool`,
`layer1`-`layer4`, `avgpool`, `fc` instead. This is expected (Phase 5 isn't implemented
yet) and doesn't affect training/prediction, which both completed before this cell runs.
The full printed layer structure is now available for planning Phase 5's mid/top layer
split once we get there.

**Next step:** since the sanity check succeeded and showed no plateau, follow up with a
longer real run (more epochs) aimed at the Strong tier (0.814).

### exp_4 (Strong tier: ResNet18 real run)

**Completed:** 2026-09-18. Same ResNet18 as `exp_3`, with `n_epochs` bumped 10→40 and
`patience` 5→10 now that the sanity check confirmed clean training with no plateau.
Everything else unchanged.

**Results** (from `exp_4_log.txt`, full 40-epoch run, condensed to key points — see
`exp_4_log.txt` in `kaggle_output/` for every epoch):

| Epoch | Valid acc | Epoch | Valid acc | Epoch | Valid acc | Epoch | Valid acc |
|---|---|---|---|---|---|---|---|
| 1 | 0.287 (best) | 11 | 0.477 | 21 | 0.606 | 31 | 0.685 (best) |
| 5 | 0.483 (best) | 13 | 0.580 (best) | 24 | 0.638 (best) | 33 | 0.675 |
| 7 | 0.498 (best) | 15 | 0.610 (best) | 26 | 0.640 (best) | 36 | 0.683 |
| 10 | 0.551 (best) | 18 | 0.612 (best) | 27 | 0.642 (best) | 37 | **0.691 (best)** |
| | | 19 | 0.624 (best) | 30 | 0.646 (best) | 40 | 0.685 |

**Best validation accuracy: 0.691 (epoch 37).** Ran the full 40 epochs, no early stop
(longest non-improving streak was well under the patience of 10). Clears neither Medium
(0.700 — missed by only ~0.009) nor Strong (0.814), but is a clear improvement over
`exp_2`'s 0.658, confirming the architecture swap is moving in the right direction —
just not far enough yet.

**Notably noisier than `exp_2`'s validation curve.** `exp_2` climbed fairly smoothly;
`exp_4` swings hard epoch-to-epoch even late in training (e.g. epoch 30→31: 0.646→0.685,
then 31→32: 0.685→0.652; epoch 35 dips to 0.633 then 36 recovers to 0.683). One plausible
explanation: the learning rate (0.0003) was inherited unchanged from the from-scratch CNN
experiments and never re-tuned for ResNet18 — BatchNorm + residual connections can behave
differently under the same LR/batch-size combination than a plain CNN, and this kind of
oscillation is a common symptom of a learning rate that's a bit too high for the
architecture. Not confirmed, just the most likely candidate — worth trying a lower LR or
a learning-rate scheduler on a follow-up run rather than just adding more epochs, since
the raw accuracy trend (best epoch near the very end, epoch 37/40) doesn't clearly say
"just needs more time" the way exp_1→exp_2 did.

Hit the same expected, harmless `AttributeError: 'ResNet' object has no attribute 'cnn'`
at the end (Phase 5 t-SNE placeholder, unaffected by training/prediction which both
completed first). `submission.csv` verified correct (3000 rows).

### exp_5 (Strong tier: ResNet18, lr lowered + more epochs)

**Completed:** 2026-09-18. Same ResNet18 as `exp_4`, but `lr` lowered 0.0003→0.0001
and `n_epochs`/`patience` bumped 40/10 → 60/15, changed together (not isolated) for
time-cost reasons.

**Best validation accuracy: 0.680 (epoch 48/60)** — surprisingly *lower* than `exp_4`'s
0.691, despite 50% more epochs. Full 60 epochs ran, no early stop.

**Did the LR fix work?** Partially. Computed average absolute epoch-to-epoch change in
validation accuracy over the last 30 epochs (31–60) vs. `exp_4`'s last 10 (31–40):
**0.0146 vs. 0.0262 — about 44% calmer.** So the lower LR did measurably reduce
volatility. But it came at the cost of slower convergence: `exp_5`'s train accuracy at
epoch 60 (0.779) is roughly where `exp_4` already was by epoch 40 (0.763) — the lower LR
just takes longer to reach the same point, not a better one, within a comparable epoch
budget.

**Since LR and epoch count changed together, causality is ambiguous** — can't say for
sure which change drove the reduced volatility, or whether the slightly lower peak
accuracy is real or noise. What is clear: the instability isn't fully explained by "LR
too high" — calming it down didn't translate into higher accuracy, and validation
accuracy was still swinging several points epoch-to-epoch even at epoch 60. Leading
candidate for a next hypothesis: the aggressive augmentation stack from Q1 interacting
with BatchNorm's running statistics — not confirmed, would need a dedicated isolated
test (lighter augmentation at the same LR) to check.

Same expected `AttributeError: 'ResNet' object has no attribute 'cnn'` at the end.
`submission.csv` verified correct (3000 rows).

### exp_6 (Strong tier: ResNet18, dropout + weight decay)

**Completed:** 2026-09-18. Same ResNet18/lr=0.0001/60 epochs as `exp_5`, but added
`nn.Dropout(p=0.3)` before the final FC layer and bumped `weight_decay` 1e-5→1e-4,
directly targeting the train/valid gap found in `exp_5` rather than touching LR again.

**Best validation accuracy: 0.679 (epoch 59/60)** — essentially unchanged from `exp_5`'s
0.680 (very slightly lower). The train/valid gap also didn't close — still ~0.05–0.14 in
later epochs, about the same magnitude as unregularized `exp_5`. This moderate dosage
didn't meaningfully counteract overfitting, but since the model's capacity to fit
training data clearly wasn't the bottleneck (76% train accuracy reached easily), the
conclusion was "dosage too weak," not "wrong approach" — motivating a stronger dose next
rather than a bigger model or more epochs. (Verified directly against the pasted Kaggle
log; artifacts weren't pulled locally for this one, superseded by `exp_7` below.)

### exp_7 (Strong tier: ResNet18, stronger dropout + weight decay)

**Completed:** 2026-09-19. Same setup as `exp_6`, dosage pushed further: dropout
0.3→0.5, weight_decay 1e-4→1e-3. `lr`/`n_epochs`/`patience` held at `exp_6`'s
0.0001/60/15, keeping dosage the only variable.

**Best validation accuracy: 0.7073 (epoch 58/60) — clears the Medium tier (0.700) for
the first time.** Full 60 epochs ran, no early stop.

**The train/valid gap stayed meaningfully tighter throughout:**

| Epoch | Train acc | Valid acc | Gap |
|---|---|---|---|
| 10 | 0.470 | 0.468 | 0.002 |
| 20 | 0.560 | 0.532 | 0.028 |
| 30 | 0.622 | 0.551 | 0.071 |
| 40 | 0.670 | 0.632 | 0.038 |
| 50 | 0.709 | 0.671 | 0.038 |
| 58 | 0.736 | **0.707 (best)** | 0.029 |
| 60 | 0.732 | 0.677 | 0.055 |

Compare to `exp_6`'s later-epoch gaps of ~0.05–0.14 — this run's gap stayed roughly
half that for most of the second half of training. Train accuracy at epoch 60 (0.732)
is also notably lower than `exp_5`'s (0.779) and `exp_6`'s (0.760) at the same point —
the stronger regularization is visibly holding the model back from memorizing as fast,
which is exactly the intended effect, and it bought a real, if modest, accuracy gain to
show for it.

`submission.csv` verified correct (3000 rows). Confirms the earlier diagnosis: dosage,
not approach, was the issue in `exp_6` — pushing regularization further (rather than
going to a bigger model) was the right call.

### exp_8 (Boss tier: 3-fold cross-validation + ensembling)

**Completed:** 2026-09-20. Pooled `train/`+`valid/` (13,643 images) into 3 true k-fold
splits (each fold rotates which ~1/3 is held out). Same recipe as `exp_7` (ResNet18,
dropout=0.5, weight_decay=1e-3, lr=0.0001, 60 epochs/patience 15) reused unchanged
across all 3 folds — diversity comes from the data split, not the recipe. Final
prediction averages softmax probabilities across all 3 fold models, then argmax.

Smoke-tested locally first on a tiny subset before spending real GPU time — caught a
real edge case: `best_acc` starting at `0` meant a fold scoring exactly `0.0` on epoch
1 would never save a checkpoint, crashing the ensembling step after that fold's full
training time was already spent. Fixed by starting `best_acc` at `-1.0`.

**Per-fold results** (from `exp_8_fold{0,1,2}_log.txt`):

| Fold | Best valid acc | Epoch |
|---|---|---|
| 0 | 0.7055 | 60 |
| 1 | 0.6951 | 60 |
| 2 | 0.6886 | 54 |

All three individually land in the same range as `exp_7`'s single-model 0.7073 — none
clearly beats it alone. Average across folds ≈ 0.6964, slightly *below* `exp_7`,
plausibly because each fold trains on less data (9096-9096-9094 vs. `exp_7`'s full
10,000) due to the 3-way split.

**Important limitation, not yet resolved:** there is no way to measure the *ensemble's*
own accuracy locally. Each fold validates only on its own held-out slice, and the
ensemble's actual target — the unlabeled test set — has no ground truth we can check
against. The per-fold numbers above describe individual model quality, not what
averaging all three together actually achieves; ensembling could measurably help even
if no individual fold beats `exp_7` (that's the whole premise of ensembling — combining
decorrelated errors), but we can't currently *prove* it did without submitting for
scoring, since every fold saw most of the pooled data during training (each pooled
image was excluded from only 1 of 3 folds' training sets), so no clean held-out
evaluation set remains to test the combined ensemble on. Worth deciding whether to
reserve a true holdout for this before treating this as final, or accept
submission-based scoring as the only real signal here.

`submission.csv` verified correct (3000 rows), generated from the 3-fold ensemble.

---
