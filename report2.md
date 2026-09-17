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

---
