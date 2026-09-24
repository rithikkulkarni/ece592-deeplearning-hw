# ECE 592 HW2 — Image Classification Roadmap

**Deadline:** 27 Sept 2026
**Deliverables:** Jupyter notebook (with run outputs), report (Q1 + Q2 + final accuracy), `submission.csv`

This roadmap breaks the assignment into ordered phases. Each phase has a goal, concrete
tasks, and a "done when" checkpoint so we know when to move on. Phases are meant to be
worked roughly in order, but Phase 5 (baseline tiers) is iterative — you'll revisit it
multiple times as your model improves.

---

## Phase 0 — Compute Strategy

**Goal:** Decide *where* training actually runs before writing more code, since this
shapes how the training script should be structured.

- [ ] Check local GPU availability (`nvidia-smi` or equivalent). If none, decide between
      Google Colab and Kaggle (the assignment PDF explicitly recommends Kaggle for
      the 20+ hour "boss" tier).
- [ ] Note the runtime limits of whichever platform you pick (Colab free/pro session
      timeouts, Kaggle's weekly GPU quota) — this determines whether the training script
      needs to be resumable from a checkpoint mid-run.
- [ ] Decide how you'll invoke training: interactively in the notebook, or as a
      standalone `.py` script launched from a terminal (recommended once you move past
      the "simple" tier, since long runs shouldn't live inside an interactive session).

**Done when:** You know which platform you're training on, and whether resumability is
a hard requirement.

### Note on using Claude Code / VS Code for this project

- Claude Code is best used for **writing, reviewing, and debugging** code — not for
  supervising a 10–20 hour training run turn-by-turn. Long runs should be kicked off
  in the background (e.g. `nohup python train.py --exp_name strong_v1 &` in a terminal,
  or as a Colab/Kaggle background execution) and revisited later for analysis.
- Once you're past the "simple" tier, it's worth extracting the training loop from the
  notebook into a parameterized script (`--exp_name`, `--epochs`, `--lr`, etc.) so you
  can launch multiple experiments from the VS Code terminal without hand-editing the
  notebook each time. The notebook can then just call/import that script, or you can
  keep a "final" notebook version for submission that mirrors your best run.
- Make sure checkpoint files (`{_exp_name}_best.ckpt`) and logs are named per-experiment
  so parallel or sequential runs don't clobber each other.

---

## Phase 1 — Get the Data Working

**Goal:** `./train`, `./valid`, `./test` populated and loadable by `FoodDataset`.

- [ ] Try the Dropbox link in cell 4. If it's dead (likely, given it's an old link):
  - [ ] Try the Google Drive link (commented out in the same cell).
  - [ ] Fall back to the Kaggle "food-11" dataset — requires a Kaggle API token
        (`kaggle.json`) if downloading programmatically.
- [ ] Unzip and verify the directory structure matches what `FoodDataset.__init__`
      expects: filenames like `<label>_<index>.jpg` directly inside `train/`, `valid/`,
      `test/`.
- [ ] Sanity check counts match the PDF: 10,000 train / 3,643 valid / 3,000 test
      (unlabeled).
- [ ] Quick visual check: load and display a few images from each split to confirm
      they're not corrupted and labels parse correctly (`FoodDataset.__getitem__`
      derives the label from the filename prefix).

**Done when:** `FoodDataset("./train")`, `("./valid")`, `("./test")` all instantiate
without error and `len()` matches expected counts.

---

## Phase 2 — Fix Known Bugs / Sanity-Check the Pipeline

**Goal:** Confirm the existing scaffolding actually works before building on top of it.

- [ ] **Logging bug (cell 21):** the log block currently does
      `with open(f"./{_exp_name}_log.txt", "a"):` followed by a `print(...)` — this
      opens the file but never writes to it. Fix so the log lines are actually written
      to disk (you'll want this for comparing experiments later).
- [ ] **`_exp_name` discipline (cell 7):** currently hardcoded to `"sample"`. Since both
      the checkpoint filename and the t-SNE loading step (cell 30) key off this value,
      decide on a naming convention per experiment (e.g. `simple_v1`, `medium_aug1`,
      `strong_resnet18`) so you don't overwrite checkpoints you want to keep.
- [ ] Run a **very short smoke test**: 1–2 epochs, possibly on a small subset of the
      training data, entirely to confirm the full pipeline (data → model → loss →
      backward → checkpoint save → reload → predict → `submission.csv`) runs
      end-to-end without errors. This should be fast and cheap, done before spending
      any real GPU-hours.

**Done when:** A full pipeline pass (train → save checkpoint → load checkpoint →
predict → write `submission.csv`) completes without errors, even if accuracy is poor.

---

## Phase 3 — Q1: Implement `train_tfm` (Augmentation)

**Goal:** Satisfy the assignment's explicit requirement — `train_tfm` must produce 5+
visibly different outputs from the same input image — and use it to move past the
"simple" baseline.

- [ ] Choose an image size (the sample uses 128×128; you're free to change it, but note
      this affects the flattened dimension into `self.fc` in the `Classifier` — if you
      change resize dimensions you may need to adjust the linear layer input size, or
      switch to `AdaptiveAvgPool2d` before flattening).
- [ ] Pick and stack `torchvision.transforms` — e.g. random crop, horizontal flip,
      rotation, color jitter, etc. The PDF explicitly encourages diversity and stacking
      multiple transforms.
- [ ] Implement this in **two places**, and note they're allowed to differ:
  - [ ] Cell 11 — the `train_tfm` actually used during training.
  - [ ] Cell 28 — the copy you'll paste into the report / Gradescope, which per the PDF
        can be a different (e.g. more illustrative) version.
- [ ] **Verify the "5+ different results" requirement concretely**: apply `train_tfm`
      to the same image 5+ times and plot the outputs side by side (mirrors the
      astronaut example in the PDF) — don't just assume the transforms are random
      enough.
- [ ] Note (for the report) *why* each transform helps — what overfitting behavior it's
      meant to counteract.

**Done when:** You have a visual grid showing 5+ distinct augmented versions of the same
image, and `train_tfm` is wired into the training `DataLoader`.

---

## Phase 4 — Baseline Progression (Simple → Medium → Strong → Boss)

**Goal:** Incrementally climb the four accuracy tiers, getting a valid submission at
each stage before escalating complexity. Don't skip straight to "boss" — each tier is a
useful checkpoint that confirms the pipeline still works.

| Tier | Target Acc. | What changes | Rough training time |
|---|---|---|---|
| Simple | 0.637 | Run sample code as-is | 0.5–1 hr |
| Medium | 0.700 | Real augmentation (Phase 3) + longer training | 1.5–2 hr |
| Strong | 0.814 | Swap in a torchvision/timm architecture | 10–12 hr |
| Boss | 0.874 | Cross-validation + ensembling | 20+ hr |

### 4a. Simple
- [ ] Run the baseline `Classifier` with minimal/no augmentation, default epoch count.
- [ ] Confirm `submission.csv` is generated and formatted correctly (this is your first
      real checkpoint that the eval path works).

### 4b. Medium
- [ ] Plug in the Phase 3 `train_tfm`.
- [ ] Increase `n_epochs` / adjust `patience` as needed; watch for overfitting via the
      train/valid accuracy gap.
- [ ] Re-check the (now-fixed) log file to compare against the simple run.

### 4c. Strong
- [ ] Choose an architecture from `torchvision.models` or `timm`.
- [ ] **Critical constraint from the PDF: pretrained weights are not allowed.**
      Explicitly pass `weights=False` (torchvision ≥0.13) or `pretrained=False`
      (older torchvision).
- [ ] Note: switching architectures changes what `model.cnn` looks like (or whether that
      attribute even exists) — keep this in mind for Phase 6 (Q2), since the
      bottom/mid/top layer indexing will need to be redone for the new architecture.
- [ ] Consider: model capacity, whether subsampling (max pooling) is present (PDF notes
      this tends to help), training time budget.

### 4d. Boss
- [ ] Implement cross-validation: merge current train + valid paths, resample into new
      train/valid splits per fold (PDF's diagram shows a standard k-fold rotation).
- [ ] Train multiple models — varying random seeds, splits, and/or architectures — in
      parallel or sequentially, saving each checkpoint separately.
- [ ] Implement ensembling: average logits/probabilities across models (less ambiguous,
      but requires saving verbose outputs) or use majority voting (simpler).
- [ ] Optionally combine with **Test-Time Augmentation (TTA)**: run `train_tfm`
      (not `test_tfm`) on test images multiple times, average those predictions, and
      optionally weight them against the deterministic `test_tfm` prediction (PDF
      suggests something like `0.2 * avg_train_tfm_pred + 0.8 * test_tfm_pred` as a
      starting point — this needs tuning, not a fixed rule).

**Done when (each tier):** A `submission.csv` exists for that tier's model, and you've
recorded its validation accuracy for the report.

---

## Phase 5 — Q2: t-SNE Visualization

**Goal:** Visualize learned representations from both "mid" and "top" layers on the
**validation set**, producing 2 images, with a brief written explanation.

- [ ] Map the PDF's colored-box diagram (bottom=red, mid=green, top=blue) onto actual
      list indices of whatever `self.cnn` ends up being **for your final chosen
      architecture** — this must be redone if Phase 4c/4d changed the model structure
      away from the original 5-block `Sequential`.
- [ ] Fill in `index` (cell 31) for the mid-layer slice, run `model.cnn[:index]` on the
      validation set, collect features, run t-SNE, plot with all 11 classes colored.
- [ ] Repeat for the top-layer slice (different `index` value) — this needs its own
      pass since `index` is currently a single variable reused across both.
- [ ] Fill in `target_label` if you also want the single-class close-up plot from the
      second half of cell 31 (optional per the PDF's core requirement, but useful for
      the report discussion).
- [ ] Make sure this uses `test_tfm` (deterministic) on the **validation set**, not
      training data with augmentation — augmented inputs would muddy the
      representation plot.
- [ ] Save both plots as image files you can embed in the report.

**Done when:** You have 2 saved t-SNE plot images (mid layer, top layer) from the
validation set, generated from your best/final trained model.

---

## Phase 6 — Report Writing

**Goal:** Write this alongside the work above, not after — capture reasoning while it's
fresh rather than reconstructing it at the end.

- [ ] **Q1 section:** show the augmented-image grid (5+ variants), list the specific
      transforms used, and explain the effect of each (what kind of invariance/
      robustness it's meant to teach the model).
- [ ] **Q2 section:** embed both t-SNE images (mid + top layer), and briefly discuss
      what you observe — e.g., do top-layer features cluster more tightly by class than
      mid-layer features? Does this match the intuition that later layers encode more
      class-discriminative information?
- [ ] **Final accuracy:** explicitly state your best test accuracy and which tier(s) it
      clears, per the PDF's explicit submission requirement.
- [ ] Optionally document your baseline progression (simple → medium → strong → boss)
      as a short table — useful evidence of process, even if not strictly required.

**Done when:** Report contains the Q1 explanation + image, the Q2 images + discussion,
and a clearly stated final accuracy number.

---

## Phase 7 — Final Packaging & Submission Checklist

- [ ] Notebook has all cells **run with visible outputs** (not just code) — a notebook
      with empty output cells likely won't satisfy "contains the running results."
- [ ] `_exp_name` in the final notebook version matches the checkpoint actually used to
      generate your final `submission.csv` and t-SNE plots (easy to mismatch after
      multiple experiments).
- [ ] `submission.csv` present, correctly formatted (`Id`, `Category` columns, 3000
      rows, `Id` zero-padded to 4 digits per `pad4`).
- [ ] Report includes: Q1 explanation + augmentation grid image, Q2 t-SNE images +
      discussion, final accuracy.
- [ ] Double-check pretrained weights were **not** used anywhere in the final model
      (re-verify `weights=False`/`pretrained=False` if you used a torchvision/timm
      architecture).
- [ ] Submit before 27 Sept 2026.

---

## Experiment Log

Every real training run (Phase 4+) gets an entry here with its exact configuration,
keyed by `_exp_name`. Results/interpretation for each go in `report2.md` instead, to
keep this file a quick-reference config log rather than a narrative.

### exp_1

First real full training run — sample-code defaults, plus Phase 3's `train_tfm`
already wired in (no deliberate tuning yet, to establish a baseline before changing
anything).

| Variable | Value |
|---|---|
| Image size | 128×128 |
| `train_tfm` | `RandomResizedCrop(128, scale=(0.5,1.0))`, `RandomHorizontalFlip()`, `RandomRotation(35)`, `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)`, `ToTensor()` |
| `test_tfm` | `Resize((128,128))`, `ToTensor()` |
| Model | `Classifier` — sample 5-block CNN (Conv+BN+ReLU+MaxPool ×5, channels 64→128→256→512→512), trained from scratch, no pretrained weights |
| Batch size | 64 |
| `n_epochs` | 8 |
| `patience` | 5 |
| Optimizer | Adam, lr=0.0003, weight_decay=1e-5 |
| Loss | CrossEntropyLoss |
| Seed | 6666 |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` |

**Result:** best validation accuracy 0.496 (epoch 7/8) — below Simple tier (0.637).
Full per-epoch breakdown and infrastructure notes in `report2.md`.

### exp_2

Same as exp_1 except `n_epochs` and `patience` bumped — exp_1 hadn't plateaued by
epoch 8, so giving it room to actually converge toward the Simple tier.

| Variable | Value | Changed from exp_1? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_1 | no |
| `test_tfm` | same as exp_1 | no |
| Model | same as exp_1 | no |
| Batch size | 64 | no |
| `n_epochs` | 30 | yes (was 8) |
| `patience` | 8 | yes (was 5) |
| Optimizer | Adam, lr=0.0003, weight_decay=1e-5 | no |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

**Result:** best validation accuracy 0.658 (epoch 25/30) — **clears Simple tier
(0.637)**, does not clear Medium (0.700). Ran full 30 epochs, no early stop. Mild
overfitting visible by epoch 30 (train acc 0.715 vs valid acc 0.643). Full per-epoch
breakdown in `report2.md`.

### exp_3

Strong tier attempt — swapped `Classifier` (from-scratch 5-block CNN) for
`build_model()` (ResNet18, `weights=None`, no pretrained weights). exp_2 plateaued
with a widening train/valid gap, suggesting the bottleneck had shifted from epoch
count to architecture capacity. Short sanity run (10 epochs) to confirm the new
architecture trains correctly on Kaggle before committing to a long run (PDF
reference: 10-12hr for Strong vs. 1.5-2hr for Medium).

| Variable | Value | Changed from exp_2? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_2 | no |
| `test_tfm` | same as exp_2 | no |
| Model | ResNet18 (`weights=None`, final FC → 11 classes) | **yes** (was `Classifier`) |
| Batch size | 64 | no |
| `n_epochs` | 10 | yes (was 30) |
| `patience` | 5 | yes (was 8) |
| Optimizer | Adam, lr=0.0003, weight_decay=1e-5 | no |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

**Result:** best validation accuracy 0.551 (epoch 10/10 — new best on the last epoch,
no plateau yet). Sanity check succeeded: ResNet18 trains cleanly end-to-end on Kaggle.
Next: longer follow-up run targeting Strong tier (0.814). Full breakdown in `report2.md`.

### exp_4

Real Strong-tier run — same ResNet18 as exp_3, `n_epochs`/`patience` bumped now that
the sanity check confirmed it trains cleanly with no plateau at epoch 10. exp_3's pace
(~88.5s/epoch, nearly identical to exp_2's CNN) puts this at ~59min of real training.

| Variable | Value | Changed from exp_3? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_3 | no |
| `test_tfm` | same as exp_3 | no |
| Model | ResNet18 (`weights=None`, final FC → 11 classes) | no |
| Batch size | 64 | no |
| `n_epochs` | 40 | yes (was 10) |
| `patience` | 10 | yes (was 5) |
| Optimizer | Adam, lr=0.0003, weight_decay=1e-5 | no |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

**Result:** best validation accuracy 0.691 (epoch 37/40, full run, no early stop). Misses
Medium (0.700) by ~0.009, well short of Strong (0.814), but clears exp_2's 0.658.
Validation curve notably noisier than exp_2's — possible untuned-LR symptom (lr=0.0003
inherited unchanged from the from-scratch CNN, never re-tuned for ResNet18). Full
breakdown in `report2.md`.

### exp_5

Combined fix — same ResNet18 as exp_4, but lr lowered 0.0003→0.0001 (targeting the
epoch-to-epoch validation noise seen in exp_4) and n_epochs bumped 40→60 at the same
time. Not an isolated ablation (both changed together, for time cost reasons), so a
follow-up may be needed to attribute which change did what if this helps.

| Variable | Value | Changed from exp_4? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_4 | no |
| `test_tfm` | same as exp_4 | no |
| Model | ResNet18 (`weights=None`, final FC → 11 classes) | no |
| Batch size | 64 | no |
| `n_epochs` | 60 | yes (was 40) |
| `patience` | 15 | yes (was 10) |
| Optimizer | Adam, **lr=0.0001**, weight_decay=1e-5 | **yes** (lr was 0.0003) |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

**Result:** best validation accuracy 0.680 (epoch 48/60) — *lower* than exp_4's 0.691
despite 50% more epochs. Lower LR did reduce late-epoch volatility ~44% (avg abs
epoch-to-epoch change 0.0146 vs exp_4's 0.0262) but didn't improve accuracy — instability
not fully explained by LR alone. Full breakdown in `report2.md`.

### exp_6

Regularization run — exp_4/exp_5's real finding was a widening train/valid accuracy
gap (overfitting), not just epoch-to-epoch noise. Added `nn.Dropout(p=0.3)` before the
final FC layer (torchvision's resnet18 has no dropout by default) and bumped
`weight_decay` 1e-5→1e-4. `lr` kept at exp_5's 0.0001 (not re-touched) to isolate
regularization as this run's variable. `n_epochs`/`patience` kept at exp_5's 60/15.

| Variable | Value | Changed from exp_5? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_5 | no |
| `test_tfm` | same as exp_5 | no |
| Model | ResNet18 (`weights=None`) + `Dropout(p=0.3)` before final FC | **yes** (dropout added) |
| Batch size | 64 | no |
| `n_epochs` | 60 | no |
| `patience` | 15 | no |
| Optimizer | Adam, lr=0.0001, **weight_decay=1e-4** | **yes** (weight_decay was 1e-5) |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

**Result:** best validation accuracy 0.679 (epoch 59/60) — essentially unchanged from
exp_5's 0.680 (very slightly lower). Train/valid gap still widened to similar magnitude
(~0.05-0.14 in later epochs). Moderate dropout+weight_decay dosage didn't meaningfully
counteract overfitting. Full breakdown in `report2.md`.

### exp_7

Stronger dosage — exp_6's moderate regularization (dropout=0.3, weight_decay=1e-4)
barely moved accuracy or the train/valid gap, so pushing both considerably higher:
dropout=0.5, weight_decay=1e-3. `lr`/`n_epochs`/`patience` kept at exp_6's
0.0001/60/15 unchanged, to isolate dosage as this run's only variable.

| Variable | Value | Changed from exp_6? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_6 | no |
| `test_tfm` | same as exp_6 | no |
| Model | ResNet18 (`weights=None`) + `Dropout(p=0.5)` before final FC | **yes** (was p=0.3) |
| Batch size | 64 | no |
| `n_epochs` | 60 | no |
| `patience` | 15 | no |
| Optimizer | Adam, lr=0.0001, **weight_decay=1e-3** | **yes** (weight_decay was 1e-4) |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

**Result:** best validation accuracy 0.7073 (epoch 58/60) — **clears Medium tier
(0.700) for the first time.** Train/valid gap stayed ~0.03-0.06 in later epochs (vs.
exp_6's ~0.05-0.14) — stronger dosage measurably reduced overfitting and delivered a
real accuracy gain. Confirms dosage (not approach) was the issue in exp_6. Full
breakdown in `report2.md`.

### exp_8

Boss-tier attempt — cross-validation + ensembling. Pooled `train/` + `valid/`
(13,643 images) and split into 3 folds (true k-fold: each fold rotates which
slice is held out, not just a different seed on the same fixed split). Same
recipe as exp_7 (ResNet18, dropout=0.5, weight_decay=1e-3, lr=0.0001, 60/15)
reused unchanged across all 3 folds — diversity comes from the data split, not
the recipe. Each fold trains independently (fresh model/optimizer) and saves
its own checkpoint (`exp_8_fold{0,1,2}_best.ckpt`). Final prediction averages
softmax probabilities across all 3 fold models, then argmax.

Smoke-tested locally first (tiny subset, 1 epoch/fold) — caught and fixed a
real edge case: `best_acc` starting at `0` meant a fold scoring exactly `0.0`
accuracy on its first epoch would never save a checkpoint, crashing the
ensembling step after that fold's full training time was already spent.
Fixed by starting `best_acc` at `-1.0` instead.

| Variable | Value | Changed from exp_7? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_7 | no |
| `test_tfm` | same as exp_7 | no |
| Model | ResNet18 (`weights=None`) + `Dropout(p=0.5)`, ×3 independent folds | **yes** (single model → 3-fold ensemble) |
| Data split | 3-fold CV over pooled train+valid (13,643 images) | **yes** (was fixed 10000/3643 split) |
| Batch size | 64 | no |
| `n_epochs` | 60 (per fold) | no |
| `patience` | 15 (per fold) | no |
| Optimizer | Adam, lr=0.0001, weight_decay=1e-3 | no |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Prediction | averaged softmax probabilities across 3 folds | **yes** (was single-model argmax) |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

Expected real execution: ~3× a single run at exp_7's pace (~285 min / 4.75hr),
before queue time.

**Result:** per-fold best valid accuracy: fold0=0.7055 (epoch 60), fold1=0.6951
(epoch 60), fold2=0.6886 (epoch 54) — average ≈0.696, slightly below exp_7's single
0.7073. **Cannot measure the ensemble's own accuracy locally** — no clean held-out
set remains after pooling all labeled data into folds, since each image was excluded
from only 1 of 3 folds' training. `submission.csv` generated (3000 rows, verified) but
whether ensembling actually helped is only knowable via submission-based scoring.
Full breakdown in `report2.md`.

### exp_9

Fixes exp_8's measurement gap. Confirmed with the user: there is no external
grading/leaderboard for this assignment — the instructor runs the submitted notebook
themselves to generate `submission.csv` and scores it against hidden labels we never
see. Validation accuracy is therefore the *only* feedback signal available, which
makes exp_8's inability to measure the ensemble's own accuracy a real problem, not
just a nice-to-have.

Only fold-splits the original `./train` (10,000 images) into 3 folds — `./valid`
(3,643 images) is deliberately excluded from all fold splits and never trained on by
any fold, staying a clean holdout. Same recipe as exp_7/exp_8 (ResNet18, dropout=0.5,
weight_decay=1e-3, lr=0.0001, 60 epochs/patience 15), unchanged. Added a new
evaluation cell after training that measures each fold *and* the ensemble (averaged
softmax) on that untouched `./valid` set — directly comparable to exp_7's 0.70730 on
the same data. Locally verified before pushing: fold splits confirmed to have zero
overlap with `./valid`, and holdout label parsing confirmed correct (3,643 labels,
range 0-10, zero parse failures) against the real data.

| Variable | Value | Changed from exp_8? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_8 | no |
| `test_tfm` | same as exp_8 | no |
| Model | ResNet18 (`weights=None`) + `Dropout(p=0.5)`, ×3 independent folds | no |
| Data split | 3-fold CV over `./train` only (10,000 images); `./valid` held out | **yes** (was pooled train+valid, 13,643) |
| Batch size | 64 | no |
| `n_epochs` | 60 (per fold) | no |
| `patience` | 15 (per fold) | no |
| Optimizer | Adam, lr=0.0001, weight_decay=1e-3 | no |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Evaluation | each fold + ensemble measured on untouched `./valid` | **yes** (new — exp_8 had no way to do this) |
| Prediction | averaged softmax probabilities across 3 folds | no |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

Expected real execution: ~4.75hr (same as exp_8), before queue time.

**Result:** per-fold accuracy on the untouched `./valid` holdout: fold0=0.64782,
fold1=0.62394, fold2=0.59155 (early-stopped at epoch 47/60). **Ensemble (avg of 3
folds) = 0.67582.** Compare to exp_7's single-model 0.70730 on the same holdout —
**the ensemble underperformed the single model by ~0.031, and every individual fold
was also worse than exp_7.** With the measurement gap fixed, this is a fair,
directly comparable answer: 3-fold CV + ensembling did not help here.

Most likely cause: each fold trained on only ~6,667 images (2/3 of `./train`), vs.
exp_7's full 10,000 — less training data per fold plausibly cost more accuracy than
3-way averaging recovered. Fold 2 also hit early stopping before using its full
epoch budget. (Kaggle kernel status showed `ERROR` after this cell — traced to the
already-known, pre-flagged Q2/t-SNE placeholder cell failing on `model.cnn`, which
doesn't exist on `ResNet`; unrelated to this result. All fold checkpoints and
`submission.csv` were produced successfully before that.)

Decision (discussed with user): given exp_9's result, and that even exp_7's 0.707
is still well short of Strong (0.814)/Boss (0.874), dropping the CV/ensembling
approach for now in favor of scaling up the single model — see exp_10.

---

### exp_10

Ensembling (exp_8/exp_9) didn't beat the single model, and the bigger gap is
Strong/Boss tier being far out of reach regardless (exp_7's single-model best is
0.707, vs. 0.814/0.874 targets). Going back to a single model, but scaling it up:
ResNet34 (deeper than exp_7's ResNet18) + a longer training budget. dropout=0.5,
weight_decay=1e-3, lr=0.0001 kept identical to exp_7 so architecture+epoch count
are the only *intentional* changes — though changing both at once means any result
can't be cleanly attributed to one or the other (accepted for time cost; flagged to
the user before running).

| Variable | Value | Changed from exp_7? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_7 | no |
| `test_tfm` | same as exp_7 | no |
| Model | ResNet34 (`weights=None`) + `Dropout(p=0.5)`, single model | **yes** (was ResNet18) |
| Data split | fixed `./train` (10,000) / `./valid` (3,643), no CV | **yes** (back from exp_8/exp_9's fold-splitting) |
| Batch size | 64 | no |
| `n_epochs` | 150 | **yes** (was 60) |
| `patience` | 25 | **yes** (was 15) |
| Optimizer | Adam, lr=0.0001, weight_decay=1e-3 | no |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Prediction | single-model argmax (no ensembling) | **yes** (back from exp_8/exp_9's averaged softmax) |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

Pushed as kernel version 12.

**Result:** best validation accuracy **0.73784 at epoch 90/150**, early-stopped at
epoch 116 (26 epochs without improvement, patience=25). Clears exp_7's 0.70730 by
~0.031 (+4.3% relative), and clears exp_9's ensemble (0.67582) too — the best single-
or multi-model result so far. Still short of Strong (0.814). Training accuracy
climbed to ~0.85 by epoch 116 while validation plateaued/declined from its epoch-90
peak — overfitting eventually caught up, just later and from a higher peak than
exp_7. Kaggle kernel status showed `ERROR` again — confirmed to be the same known
Q2/t-SNE placeholder issue as exp_9 (`model.cnn` doesn't exist on `ResNet`),
unrelated to training; checkpoint/log/submission.csv all produced successfully.

---

### exp_11

exp_10's curve showed validation accuracy plateauing/drifting down after its epoch-90
peak while training accuracy kept climbing — a flat lr may not have been giving late
training room to settle into a better minimum. Discussed next-step options with the
user (LR schedule / re-tried ensembling on top of the stronger single model / bigger
architecture or stronger augmentation); user chose to try the LR schedule first.

Added `CosineAnnealingLR(optimizer, T_max=n_epochs)`, stepped once per epoch (not per
batch). Starting lr raised from exp_10's flat 0.0001 to 0.0003, since the schedule
brings it back down over the run anyway — lets early training move faster while
still ending low for late-epoch fine-tuning. Current lr now also printed/logged each
epoch. Everything else kept identical to exp_10 so the schedule (+ its higher
starting lr) is the only intentional change.

| Variable | Value | Changed from exp_10? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_10 | no |
| `test_tfm` | same as exp_10 | no |
| Model | ResNet34 (`weights=None`) + `Dropout(p=0.5)`, single model | no |
| Data split | fixed `./train` (10,000) / `./valid` (3,643), no CV | no |
| Batch size | 64 | no |
| `n_epochs` | 150 | no |
| `patience` | 25 | no |
| Optimizer | Adam, lr=0.0003 (was 0.0001), weight_decay=1e-3 | **yes** (starting lr) |
| LR schedule | `CosineAnnealingLR`, T_max=150, stepped per epoch | **yes** (new — exp_10 had no schedule) |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Prediction | single-model argmax (no ensembling) | no |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

Pushed as kernel version 13.

**Result:** best validation accuracy **0.79948 at epoch 147/150** — ran the full
150-epoch budget with no early stopping (vs. exp_10 stopping at 116). Clears exp_10's
0.73784 by ~0.062 (+8.4% relative), and is now only ~0.0145 short of Strong (0.814).
Unlike exp_10, validation accuracy tracked upward with training accuracy for
essentially the whole run instead of peaking early and drifting down — the schedule's
intended effect (fast early progress, fine-grained late updates) showed up clearly in
the curve. Train accuracy reached ~0.95 by the end (real train/valid gap remains,
~0.15), but it's no longer costing validation accuracy the way exp_10's overfitting
did. Kaggle kernel status showed `ERROR` again — confirmed to be the same known
Q2/t-SNE placeholder issue, unrelated to training; checkpoint/log/submission.csv all
produced successfully.

Found the actual assignment instructions PDF partway through this experiment (had
been working from indirect context before). Cross-checked against it: confirmed no
Kaggle leaderboard exists for this course despite the PDF's public/private grading
slide (asked user directly) — the exp_9 redesign (keeping `./valid` untouched rather
than merging per the PDF's literal CV instructions) remains correct for us, since a
real local holdout is still the only way to measure anything without a leaderboard.
PDF also gave a concrete definition of TTA (test-time augmentation): averaging
predictions from multiple `train_tfm` passes with the deterministic `test_tfm`
prediction at inference time, weighted (PDF's example: 0.2 avg(train_tfm) + 0.8
test_tfm) — an inference-only technique, no retraining required. Planned as exp_12
alongside a further architecture bump.

---

### exp_12

Two changes bundled: (1) architecture — ResNet50 in place of exp_11's ResNet34
(training recipe otherwise identical: dropout=0.5, weight_decay=1e-3, lr=0.0003
cosine-annealed, n_epochs=150, patience=25); (2) TTA — additive, inference-only,
doesn't touch training. Bundling is safe here specifically because TTA's effect is
measured independently: a new evaluation cell compares plain (`test_tfm`-only)
accuracy against TTA (5× `train_tfm` passes averaged with `test_tfm`, weighted
0.2/0.8 per the PDF's example) on the labeled `./valid` set, before TTA is trusted
for the actual `submission.csv` predictions on `./test`.

| Variable | Value | Changed from exp_11? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_11 | no |
| `test_tfm` | same as exp_11 | no |
| Model | ResNet50 (`weights=None`) + `Dropout(p=0.5)`, single model | **yes** (was ResNet34) |
| Data split | fixed `./train` (10,000) / `./valid` (3,643), no CV | no |
| Batch size | 64 | no |
| `n_epochs` | 150 | no |
| `patience` | 25 | no |
| Optimizer | Adam, lr=0.0003, weight_decay=1e-3 | no |
| LR schedule | `CosineAnnealingLR`, T_max=150, stepped per epoch | no |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Evaluation | plain vs. TTA accuracy compared on `./valid` before submission | **yes** (new) |
| Prediction | TTA blend (5× train_tfm + test_tfm, 0.2/0.8) | **yes** (was test_tfm-only argmax) |
| Platform | Kaggle (GPU), via `kaggle_sync.py run` | no |

Pushed as kernel version 14.

**Result: best validation accuracy 0.81302 at epoch 132/150** (ResNet50 alone,
`test_tfm`-only) — full 150-epoch budget, no early stopping. Re-measured as 0.81307
in the dedicated evaluation cell (same checkpoint, non-shuffled loader — the ~0.00005
difference is noise). TTA (5x `train_tfm` passes averaged with `test_tfm`, 0.2/0.8)
brought this to **0.81938 on `./valid` — clears Strong (0.814) for the first time**,
a +0.00631 gain from TTA on top of ResNet50's already-strong 0.81307. Both changes
contributed: ResNet50 alone already beat exp_11's 0.79948 by ~0.014, and TTA supplied
the final push over the line. Kaggle kernel status showed `ERROR` again — confirmed
to be the same known Q2/t-SNE placeholder issue, unrelated to this result.

---

### exp_13

Now clearing Strong, targeting Boss (0.874) — a gap unlikely to close with more single-
model tuning alone. Discussed with the user: exp_9's ensemble failed because 3-fold CV
starved each fold of training data (~2/3 of `./train`); the fix isn't to abandon
ensembling, it's to ensemble models each trained on the FULL `./train` set
independently, exactly per the assignment PDF's own "sample procedure for beating the
boss baseline" (multiple random seeds / multiple model structures, each fully
trained, then ensembled).

3-model ensemble: exp_11 (ResNet34+cosine, 0.79948), exp_12 (ResNet50+cosine+TTA,
0.81938), and this experiment (DenseNet121 — a structurally different family, dense
connections instead of residual — same recipe otherwise: dropout=0.5, weight_decay=
1e-3, lr=0.0003 cosine-annealed, 150 epochs/patience 25). Chose DenseNet121 over
EfficientNet/ConvNeXt/ViT specifically because those are tuned for pretrained
initialization and tend to train poorly from scratch on small datasets — a real risk
given pretrained weights are banned for this assignment.

Reused exp_11/exp_12's checkpoints rather than retraining them (would cost ~3x the GPU
time for identical results) — required a real platform fix: Kaggle kernel runs start
from a clean filesystem, so a previous run's output isn't automatically available in a
new run. Uploaded both checkpoints as a private Kaggle Dataset
(`rithikkulkarni1/hw2-checkpoints`, via `kaggle datasets create`) and added it as a
`dataset_source` in `kernel-metadata.json`, so this run reads them from
`/kaggle/input/hw2-checkpoints/`.

`build_model` refactored to take an `arch` argument (`resnet34`/`resnet50`/
`densenet121`) so the ensemble-evaluation cell can construct and load all 3
architectures in one notebook. New `tta_predict_probs()` helper factors out exp_12's
TTA logic (now called once per ensemble member instead of duplicated). Per discussion
with the user: ensemble input is each model's **TTA-blended** prediction (not plain),
since TTA and ensembling address different error sources and stack cleanly; weighting
is **equal** across the 3 members (simplest, matches the PDF's own Ensemble slide,
and the accuracy spread between members isn't large enough to obviously justify
tuning weights against the same `./valid` set being used to judge the ensemble).

New evaluation cell measures each model's individual TTA accuracy on `./valid`, then
the equal-weighted ensemble accuracy, before trusting it for `submission.csv`.

| Variable | Value | Changed from exp_12? |
|---|---|---|
| Image size | 128×128 | no |
| `train_tfm` | same as exp_12 | no |
| `test_tfm` | same as exp_12 | no |
| Model | DenseNet121 (`weights=None`) + `Dropout(p=0.5)`, single model | **yes** (was ResNet50) |
| Data split | fixed `./train` (10,000) / `./valid` (3,643), no CV | no |
| Batch size | 64 | no |
| `n_epochs` | 150 | no |
| `patience` | 25 | no |
| Optimizer | Adam, lr=0.0003, weight_decay=1e-3 | no |
| LR schedule | `CosineAnnealingLR`, T_max=150, stepped per epoch | no |
| Loss | CrossEntropyLoss | no |
| Seed | 6666 | no |
| Ensemble | exp_11 + exp_12 + exp_13, equal-weighted avg of each model's TTA-blended probabilities | **yes** (new — first genuine multi-model ensemble, each on full data) |
| Prediction | 3-model TTA-ensemble | **yes** (was single-model TTA) |
| Platform | Kaggle (GPU), via `kaggle_sync.py run`; reads `rithikkulkarni1/hw2-checkpoints` dataset as input | **yes** (first use of a dataset input) |

Pushed as kernel version 15. Result pending.

---

## Quick Reference: Phase Dependencies

```
Phase 0 (compute strategy)
   │
Phase 1 (data) ──► Phase 2 (bug fixes / smoke test)
                        │
                   Phase 3 (Q1: train_tfm) ──► Phase 4 (baseline tiers, iterative)
                                                     │
                                              Phase 5 (Q2: t-SNE) ◄── needs final model from Phase 4
                                                     │
                                              Phase 6 (report, written throughout)
                                                     │
                                              Phase 7 (packaging & submission)
```