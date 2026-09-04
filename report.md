# HW1 Working Notes — COVID-19 Cases Prediction

Running log of code changes made to `ECE592HW1_rrkulka3.ipynb` and the reasoning behind them, kept alongside the notebook while working through the assignment questions. This is scratch/process notes, not the final polished report submitted to Moodle.

## Environment

- Notebook is configured for a Colab **T4 GPU** runtime (`accelerator: GPU, gpuType: T4` in notebook metadata). It is not set up for TPU — `get_device()` only checks `torch.cuda.is_available()`, so TPU execution would require rewriting the device/training code with `torch_xla`, which is out of scope.
- Confirmed connection: `get_device()` → `cuda`, `torch.cuda.get_device_name(0)` → `Tesla T4`.
- Reminder: reconnecting to a new Colab runtime clears all kernel state — cells must be re-run from the top (or Runtime → Run all) before any diagnostic/one-off cell will work.

## Q1 — Baseline training timing

**Change:** Wrapped the training call in cell-28 with `time.time()` before/after and printed the elapsed seconds.

```python
import time

start_time = time.time()
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
elapsed_time = time.time() - start_time
print(f'Training took {elapsed_time:.2f} seconds')
```

**Why:** Question 1 asks specifically to time "the training process," not data loading, model construction, or the plotting/testing cells afterward — those are near-instant and would just dilute the number.

**Result (T4 GPU, baseline config — 1 hidden layer, SGD, batch_size=270):** 54.2s.

**Caveat noted for the report:** An earlier timing attempt on a CPU-only kernel gave a much slower (untrustworthy for comparison) number — worth mentioning in the report as a sanity-check aside, but the GPU number above is the one to report for Q1.

## Q2a — SGD vs Adam

**Change:** Added an experiment-switch cell right before the hyperparameter config cell:

```python
# Experiment switch: set this to 'sgd' or 'adam', then Run All to get that optimizer's results.
OPTIMIZER_CHOICE = 'sgd'  # 'sgd' or 'adam'

assert OPTIMIZER_CHOICE in ('sgd', 'adam'), "OPTIMIZER_CHOICE must be 'sgd' or 'adam'"
```

Rewrote the config cell to define both optimizers' full hyperparameter sets in an `OPTIMIZER_CONFIGS` dict, then merge in whichever one `OPTIMIZER_CHOICE` points to:

```python
OPTIMIZER_CONFIGS = {
    'sgd':  {'optimizer': 'SGD',  'optim_hparas': {'lr': 0.001, 'momentum': 0.9}},
    'adam': {'optimizer': 'Adam', 'optim_hparas': {'lr': 0.001}},
}
config = {
    'n_epochs': 3000,
    'batch_size': 270,
    'early_stop': 200,
    'save_path': f'models/model_{OPTIMIZER_CHOICE}.pth',
}
config.update(OPTIMIZER_CONFIGS[OPTIMIZER_CHOICE])
```

Also tagged the training-time print and the learning-curve plot title with `OPTIMIZER_CHOICE`, so saved outputs from each run are clearly labeled.

**Why:**
- Adam doesn't accept a `momentum` kwarg (it uses `betas` internally, default `(0.9, 0.999)`, left untouched) — so the SGD and Adam hyperparameter dicts can't just share one shape, they need to be defined separately.
- Keeping both configs fully written out (not commented in/out) means the notebook always has both ready — flipping `OPTIMIZER_CHOICE` and doing Runtime → Run All is enough to reproduce either run without hand-editing config values, which also makes results easy to reproduce for the report.
- Only the optimizer changes between the two runs — batch_size (270), model architecture (baseline 1-hidden-layer net), n_epochs, and early_stop are all held fixed, so the comparison isolates the optimizer's effect.
- `save_path` is now per-optimizer (`model_sgd.pth` / `model_adam.pth`) so the two runs don't clobber each other's saved checkpoint — useful if we want to reload and compare predictions from both later.

**How to reproduce both results:** set `OPTIMIZER_CHOICE = 'sgd'`, Run All, record the printed time/dev-MSE and learning curve; then set `OPTIMIZER_CHOICE = 'adam'`, Run All again, record the same. Both go into the Q2a comparison in the report.

**SGD baseline result:** best dev MSE ≈ 0.7454 (epoch 1707), training took 54.2s, stopped after 1908 epochs (early stopping).

**Adam result:** best dev MSE ≈ 0.7880, training took 24.2s, stopped after 896 epochs (early stopping).

**Takeaway:** Adam converges faster per epoch, reaching its best result and triggering early stopping (patience=200) in under half the epochs SGD needed — so its shorter wall-clock time reflects fewer total epochs, not a cheaper per-epoch cost. Under the same early-stop patience, though, Adam plateaus and halts before refining as precisely as SGD+momentum's slower, steadier descent, leaving it with a higher final dev MSE. Both used the same lr (0.001), which was chosen for SGD and not independently tuned for Adam — worth flagging as a caveat rather than a fully-tuned comparison.

## Q2b — Batch size 50 vs 500

**Concept:** Batch size controls how many training samples contribute to each gradient estimate before a weight update. With 2430 training samples: batch_size=270 → 9 updates/epoch (original baseline), batch_size=50 → ~49 updates/epoch, batch_size=500 → ~5 updates/epoch. This is the classic gradient noise vs. stability tradeoff — smaller batches give noisier but more frequent updates (can help escape sharp minima, acts as implicit regularization, but is more sensitive to learning rate and slower per epoch on GPU due to less parallelism); larger batches give smoother, more accurate gradient estimates and better GPU utilization per step, but fewer updates per epoch and can converge to sharper minima that generalize worse.

**Change:** Extended the experiment-switch cell (added alongside `OPTIMIZER_CHOICE` for Q2a) with a `BATCH_SIZE` variable:

```python
OPTIMIZER_CHOICE = 'sgd'  # 'sgd' or 'adam'
BATCH_SIZE = 270          # try 50, 270 (original baseline), 500, etc.
```

Config cell now uses `'batch_size': BATCH_SIZE` instead of the hardcoded `270`, and `save_path` includes both the optimizer and batch size (`model_{OPTIMIZER_CHOICE}_bs{BATCH_SIZE}.pth`) so no combination of runs overwrites another. Training-time print and learning-curve title also tag `bs={BATCH_SIZE}` now.

**Why:** Same rationale as the optimizer switch — one toggle at the top, Run All, reproducible results without hand-editing config internals. Keeping batch size and optimizer as independent toggles in the same cell means we can isolate batch size's effect on either optimizer if we want to later, though the assignment's baseline comparison (SGD, bs=50 vs bs=500) only needs `OPTIMIZER_CHOICE='sgd'` held fixed while `BATCH_SIZE` varies.

**How to reproduce:** `OPTIMIZER_CHOICE='sgd'`, `BATCH_SIZE=50`, Run All, record time/best dev MSE/epoch count; repeat with `BATCH_SIZE=500`. Compare against the bs=270 baseline already recorded above.

**Results (SGD, batch_size 50 vs 500 vs 270 baseline):**

| Batch size | Epochs | Best dev MSE | Training time |
|---|---|---|---|
| 50  | 500  | 0.7299 | 39.71s |
| 270 (baseline) | 1908 | 0.7454 | 54.2s |
| 500 | 1067 | 0.7966 | 20.85s |

## Next up

- Q2c: fix feature-selection TODO in `COVID19Dataset` (`feats = list(range(40)) + [57, 75]`) and rerun
- Q2d: layer count sweep (1–5)
- Q2e: best layer count + BatchNorm
- Q2f: + Dropout after BatchNorm
