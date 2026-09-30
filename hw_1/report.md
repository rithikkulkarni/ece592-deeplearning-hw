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

## Q2c — Feature selection

**Concept:** The dataset has 93 features per sample: 40 one-hot state indicators + 3 days of survey data (18 features/day: symptoms, mental health, behavior indicators, plus each day's `tested_positive` rate as the last feature in that day's block). Day 3's `tested_positive` is the regression target. `target_only=True` restricts the model to a hand-picked subset: the 40 state indicators plus indices 57 and 75 — day 1's and day 2's `tested_positive` values (day blocks are 18 wide starting at index 40, so index 57 is the last column of day 1's block, index 75 the last column of day 2's block). Rationale: test-positivity is strongly autocorrelated day-to-day, so recent positivity rate is likely one of the most predictive signals, and dropping the other ~32 survey columns/day reduces dimensionality and potential noise — at the risk of discarding features that do carry real signal.

**Change:** Fixed the TODO in `COVID19Dataset` (cell-11), which was a bare `pass` (would `NameError` on `target_only=True`):

```python
if not target_only:
    feats = list(range(93))
else:
    # Using 40 states & 2 tested_positive features (indices = 57 & 75)
    feats = list(range(40)) + [57, 75]
```

Added `TARGET_ONLY` as a third toggle in the experiment-switch cell, alongside `OPTIMIZER_CHOICE` and `BATCH_SIZE`:

```python
OPTIMIZER_CHOICE = 'sgd'
BATCH_SIZE = 500
TARGET_ONLY = False       # False = all 93 features, True = 40 states + 2 tested_positive features only
```

The config cell now sets `target_only = TARGET_ONLY` (previously hardcoded to `False`) and folds it into `save_path` (`model_{OPTIMIZER_CHOICE}_bs{BATCH_SIZE}_to{int(TARGET_ONLY)}.pth`). Training-time print and learning-curve title also tag `target_only=...` now. No changes needed to `prep_dataloader` calls — they already passed `target_only=target_only` through, that variable was just previously hardcoded.

**Why:** Same toggle-and-Run-All pattern as the other two switches, for reproducibility. Fixing the TODO was necessary just to make `target_only=True` runnable at all.

**Caveat:** The assignment says "observe how the training and test loss change," but the test set (`covid.test.csv`) has no ground-truth labels available locally (`test()` only returns predictions, no loss) — so we report train and **dev** loss instead, consistent with how the rest of the notebook evaluates ("dev" is the held-out validation split used throughout).

**How to reproduce:** `TARGET_ONLY=False` is the 93-feature baseline (already have results at bs=270/SGD from Q1: 0.7454 dev MSE). Set `TARGET_ONLY=True` (keep other toggles matching a prior run for a clean comparison, e.g. `OPTIMIZER_CHOICE='sgd'`, `BATCH_SIZE=270`), Run All, record time/best dev MSE/epoch count and the printed input dim (should read `dim = 42` instead of `dim = 93`).

**Results (SGD, batch_size 270) — final, in Overleaf report:**

| Features | Epochs | Best dev MSE | Final train loss | Training time |
|---|---|---|---|---|
| All 93 (baseline) | 1908 | 0.7454 | 0.5007 | 54.2s |
| 42 (feature-selected, `target_only=True`) | 519 | 0.9614 | 1.0605 | 13.14s |

**Takeaway:** Feature selection sped up training a lot (54.2s → 13.14s, 1908 → 519 epochs) but hurt accuracy (dev MSE 0.7454 → 0.9614, train loss 0.5007 → 1.0605) — the dropped ~32 survey features/day apparently carried real signal beyond state identity + recent positivity rate.

**Note for the report:** the test set has no ground-truth labels, so "test loss" from the assignment prompt isn't literally computable — dev MSE is used as the standard proxy throughout. Worth stating this once in the Overleaf doc since Q2c's prompt explicitly says "training and test loss."

## Training loss reporting

**Change:** The training cell only ever printed dev-set MSE (mislabeled colloquially as "test loss" in conversation — the actual test set has no labels, see Q2c caveat). Added two prints after training using `model_loss_record['train']` (per-batch training loss, already recorded by `train()` but never surfaced as a number):

```python
print(f"Final training loss (last batch): {model_loss_record['train'][-1]:.4f}")
print(f"Mean training loss (last 50 batches): {sum(model_loss_record['train'][-50:]) / len(model_loss_record['train'][-50:]):.4f}")
```

**Why:** Gives a reportable scalar for training loss alongside best dev MSE, rather than only being visible qualitatively in the learning-curve plot's red curve. Useful for filling in train-loss columns in report tables/LaTeX going forward — worth re-running prior experiments (Q2a/b/c) if their exact final training-loss numbers are wanted in the report; the results already recorded above only have dev MSE.

## Q2d — Layer count sweep

**Concept:** More hidden layers = more representational capacity (can fit more complex, hierarchical functions of the input), but also more parameters to overfit with on a small dataset (2430 training rows), and generally harder/slower optimization. Classic capacity-vs-generalization tradeoff — there's no guarantee more layers helps; this baseline has no BatchNorm/Dropout yet to stabilize deeper training (that's Q2e/Q2f).

**Change:** Made `NeuralNet` (cell-15) take a `num_layers` param (and `use_bn`/`use_dropout`, unused for now — added ahead of time for Q2e/Q2f) and build its hidden stack in a loop instead of being hardcoded to 1 hidden layer:

```python
def __init__(self, input_dim, num_layers=1, use_bn=False, use_dropout=False):
    ...
    layers = []
    in_features = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_features, 64))
        if use_bn: layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU())
        if use_dropout: layers.append(nn.Dropout(p=0.5))
        in_features = 64
    layers.append(nn.Linear(in_features, 1))
    self.net = nn.Sequential(*layers)
```

Added `NUM_LAYERS = 1` as a fourth toggle in the switch cell. Both places that construct `NeuralNet` (initial construction, cell-28; and the reload-for-plot_pred cell, cell-32) now pass `num_layers=NUM_LAYERS` — the reload one *had* to be updated too, since loading a saved state_dict into a differently-shaped model would error. `save_path` now includes `_L{NUM_LAYERS}`, and training print/plot title tag it too.

**Note:** The leftover `NeuralNet_layer3` scaffold cell (cell-37, under "Q2.2 Layer change") is now superseded by this parameterized `NeuralNet` — it's unused dead code at this point. Flagged for you to decide whether to delete it or leave it as an unused artifact.

**How to reproduce:** Set `NUM_LAYERS` to 1, 2, 3, 4, 5 in turn (keep other toggles fixed, e.g. `sgd`, `bs=270`, `target_only=False`), Run All each time, record epochs/best dev MSE/final train loss/time.

_Results:_

2 layers:
Finished training after 925 epochs
[sgd, bs=270, target_only=False, num_layers=2] Training took 25.12 seconds, best dev MSE = 0.7089
Final training loss (last batch): 0.3407
Mean training loss (last 50 batches): 0.3566

3 layers:
Finished training after 569 epochs
[sgd, bs=270, target_only=False, num_layers=3] Training took 16.54 seconds, best dev MSE = 0.7446
Final training loss (last batch): 0.3505
Mean training loss (last 50 batches): 0.3676

4 layers:
Finished training after 461 epochs
[sgd, bs=270, target_only=False, num_layers=4] Training took 14.54 seconds, best dev MSE = 0.7429
Final training loss (last batch): 0.5544
Mean training loss (last 50 batches): 0.4340

5 layers:
Finished training after 437 epochs
[sgd, bs=270, target_only=False, num_layers=5] Training took 14.22 seconds, best dev MSE = 0.7582
Final training loss (last batch): 0.5023
Mean training loss (last 50 batches): 0.4341

## pred.csv auto-extraction

**Problem:** The notebook runs on a remote Colab GPU kernel via VSCode, so `pred.csv` (written by `save_pred()`, cell-34) lands on the Colab VM's filesystem, not locally. Colab's `files.download()` only works from the actual Colab browser tab, not a remote VSCode Jupyter connection.

**Change:** cell-34 now base64-encodes `pred.csv` and prints it between `PRED_CSV_BASE64_START`/`END` markers. Since VSCode's Jupyter extension syncs remote-kernel cell outputs back into the local `.ipynb` file on disk, that encoded copy ends up in the local notebook file after every run.

Added `watch_pred.py` — a standalone, dependency-free polling script that watches the notebook file's mtime, and whenever it changes, re-parses cell-34's output for the base64 markers and writes/overwrites `pred.csv` locally if the content changed. Run once with `python watch_pred.py` and leave it running; it auto-updates `pred.csv` after every training run without needing to ask me to extract it each time.

**Why:** No true remote-to-local filesystem bridge exists here, so full automation requires either (a) a Colab-browser-only API (`files.download()`, ruled out since we're in VSCode) or (b) leaning on the one thing that *does* sync automatically — cell outputs into the local `.ipynb`. Polling that file locally is the least invasive way to get "automatic" without extra dependencies or Colab-side changes.

## Q2e — BatchNorm (on best layer count = 2)

**Concept:** BatchNorm normalizes each layer's activations (per-batch mean 0, variance 1, then a learned scale/shift) before the nonlinearity. Stabilizes the distribution of activations flowing through the network as weights update ("internal covariate shift"), which tends to smooth optimization and reduce sensitivity to init/learning rate. Directly relevant given what Q2d showed: 3/4/5-layer nets plateaued at *worse* training loss than the 2-layer net, plausibly due to unstable optimization without normalization — this is the standard fix.

**Change:** `NeuralNet` already supported `use_bn` (added ahead of time in Q2d). Added `USE_BATCHNORM` as a fifth toggle in the switch cell, and set `NUM_LAYERS = 2` as the new default (winner from Q2d), per the assignment's "based on the best layer number" instruction:

```python
NUM_LAYERS = 2            # number of hidden layers in NeuralNet (best from Q2d sweep: 2)
USE_BATCHNORM = False     # insert BatchNorm1d after each hidden layer's Linear (before ReLU)
```

Both `NeuralNet(...)` construction sites (cell-28, cell-32) now pass `use_bn=USE_BATCHNORM` alongside `num_layers=NUM_LAYERS`. `save_path`/print/plot title all tag `use_bn` too.

**How to reproduce:** `NUM_LAYERS=2` fixed. `USE_BATCHNORM=False` reproduces the Q2d 2-layer result (0.7089 dev MSE) as this comparison's baseline. Set `USE_BATCHNORM=True`, Run All, record time/best dev MSE/train loss/epoch count.

_Results:_

### first test (without batchnorm)
config:
OPTIMIZER_CHOICE = 'sgd'  # 'sgd' or 'adam'
BATCH_SIZE = 270          # try 50, 270 (original baseline), 500, etc.
TARGET_ONLY = False       # False = all 93 features, True = 40 states + 2 tested_positive features only
NUM_LAYERS = 2            # number of hidden layers in NeuralNet (best from Q2d sweep: 2)
USE_BATCHNORM = False     # insert BatchNorm1d after each hidden layer's Linear (before ReLU)

results:
Finished training after 925 epochs
[sgd, bs=270, target_only=False, num_layers=2, use_bn=False] Training took 29.89 seconds, best dev MSE = 0.7089
Final training loss (last batch): 0.3407
Mean training loss (last 50 batches): 0.3566

### second test (with batchnorm)
config:
OPTIMIZER_CHOICE = 'sgd'  # 'sgd' or 'adam'
BATCH_SIZE = 270          # try 50, 270 (original baseline), 500, etc.
TARGET_ONLY = False       # False = all 93 features, True = 40 states + 2 tested_positive features only
NUM_LAYERS = 2            # number of hidden layers in NeuralNet (best from Q2d sweep: 2)
USE_BATCHNORM = True     # insert BatchNorm1d after each hidden layer's Linear (before ReLU)

results:
Finished training after 391 epochs
[sgd, bs=270, target_only=False, num_layers=2, use_bn=True] Training took 11.72 seconds, best dev MSE = 0.7492
Final training loss (last batch): 0.5287
Mean training loss (last 50 batches): 0.6438

### third test (with batchnorm, batch size switched from 270 to 50)
config:
OPTIMIZER_CHOICE = 'sgd'  # 'sgd' or 'adam'
BATCH_SIZE = 50          # try 50, 270 (original baseline), 500, etc.
TARGET_ONLY = False       # False = all 93 features, True = 40 states + 2 tested_positive features only
NUM_LAYERS = 2            # number of hidden layers in NeuralNet (best from Q2d sweep: 2)
USE_BATCHNORM = True     # insert BatchNorm1d after each hidden layer's Linear (before ReLU)

results:
Finished training after 493 epochs
[sgd, bs=50, target_only=False, num_layers=2, use_bn=True] Training took 51.89 seconds, best dev MSE = 0.7714
Final training loss (last batch): 0.5229
Mean training loss (last 50 batches): 0.7119

**Takeaway:** BatchNorm made things *worse* here, not better — dev MSE rose 0.7089 → 0.7492, training loss rose too (0.3407 → 0.5287 final), and it converged much faster to that worse point (925 → 391 epochs). Counter to the usual expectation, but explainable: this network is shallow (2 layers) with an lr already well-suited to it (no real instability for BN to fix), batch_size=270 means only 9 large batches/epoch (noisy per-batch statistics without the small-batch regime where BN often helps most), and — as in the Q2d layer sweep — faster convergence here reflects early stopping triggering sooner at a worse plateau, not a better minimum found efficiently. Worth reporting as a genuine negative result rather than reframing it as expected.

**Q2e status: complete.** The assignment only asks for one before/after comparison at the best layer count — done, no further runs needed for this question.

## Q2f — Dropout after BatchNorm

**Concept:** Dropout randomly zeroes a fraction (`p=0.5`) of a layer's neurons on each training step, forcing the network to avoid relying on specific neurons/co-adapted groups — a regularizer targeting overfitting (as opposed to BatchNorm, which targets optimization stability). At eval time all neurons are used. Given the small dataset (2430 training rows), dropout is the natural next lever after BatchNorm.

**Change:** `NeuralNet` already supported `use_dropout` (added ahead of time in Q2d). Added `USE_DROPOUT` as a sixth toggle in the switch cell. Reset `BATCH_SIZE` back to `270` (the Q2e primary-comparison setting — bs=50 was an extra exploratory test, not the baseline to build on):

```python
BATCH_SIZE = 270          # reset from the bs=50 exploratory test
USE_BATCHNORM = True      # insert BatchNorm1d after each hidden layer's Linear (before ReLU)
USE_DROPOUT = False       # insert Dropout(p=0.5) after BatchNorm/ReLU in each hidden layer
```

All `NeuralNet(...)` construction sites now pass `use_dropout=USE_DROPOUT` alongside the other params. `save_path`/print/plot title tag it too (`_do{0|1}`).

**How to reproduce:** Per the assignment ("include dropout after batch norm"), build on the Q2e config: `NUM_LAYERS=2`, `USE_BATCHNORM=True`, `BATCH_SIZE=270` fixed. `USE_DROPOUT=False` reproduces the Q2e BatchNorm-only result (0.7492 dev MSE) as this comparison's baseline. Set `USE_DROPOUT=True`, Run All, record time/best dev MSE/train loss/epoch count.

_Results:_

### first run (dropout=false)
config:
OPTIMIZER_CHOICE = 'sgd'  # 'sgd' or 'adam'
BATCH_SIZE = 270          # try 50, 270 (original baseline), 500, etc.
TARGET_ONLY = False       # False = all 93 features, True = 40 states + 2 tested_positive features only
NUM_LAYERS = 2            # number of hidden layers in NeuralNet (best from Q2d sweep: 2)
USE_BATCHNORM = True      # insert BatchNorm1d after each hidden layer's Linear (before ReLU)
USE_DROPOUT = False       # insert Dropout(p=0.5) after BatchNorm/ReLU in each hidden layer

results:
Finished training after 391 epochs
[sgd, bs=270, target_only=False, num_layers=2, use_bn=True, use_dropout=False] Training took 11.70 seconds, best dev MSE = 0.7492
Final training loss (last batch): 0.5287
Mean training loss (last 50 batches): 0.6438

### second run (dropout=true)
config:
OPTIMIZER_CHOICE = 'sgd'  # 'sgd' or 'adam'
BATCH_SIZE = 270          # try 50, 270 (original baseline), 500, etc.
TARGET_ONLY = False       # False = all 93 features, True = 40 states + 2 tested_positive features only
NUM_LAYERS = 2            # number of hidden layers in NeuralNet (best from Q2d sweep: 2)
USE_BATCHNORM = True      # insert BatchNorm1d after each hidden layer's Linear (before ReLU)
USE_DROPOUT = True       # insert Dropout(p=0.5) after BatchNorm/ReLU in each hidden layer

results:
Finished training after 1816 epochs
[sgd, bs=270, target_only=False, num_layers=2, use_bn=True, use_dropout=True] Training took 56.34 seconds, best dev MSE = 1.4663
Final training loss (last batch): 3.9225
Mean training loss (last 50 batches): 4.7508