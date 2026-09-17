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