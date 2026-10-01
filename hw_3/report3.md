# HW3 Working Report — Design of GNNs

Running log of changes, design decisions, and notebook runs for HW3. This is the
raw material for the final Overleaf report — not the report itself. Newest entries
are appended at the bottom of each section.

- **Notebook:** `hw_3/ECE592HW3-rrkulka3.ipynb`
- **Warmup (ungraded, not submitted):** `hw_3/hw3_warmup.ipynb`
- **Config-level run log:** `ROADMAP.md` → HW3 Experiment Log

---

## 0. Initial Audit (2026-09-30)

What the scaffold already provides vs. what we have to write:

| Notebook cell | Content | Our work |
|---|---|---|
| 12 | Loads ENZYMES via `TUDataset` | none |
| 14 | `get_num_classes`, `get_num_features` | **Q1** (~1 line each) |
| 17 | `get_graph_class` | **Q2** (~1 line) |
| 19 | `get_graph_num_edges` (undirected, don't use `data.num_edges` directly) | **Q3** (~4 lines) |
| 22 | Loads `ogbn-arxiv` with `T.ToSparseTensor()` | none |
| 24 | `graph_num_features` | **Q4** (~1 line) |
| 29 | Preprocess arxiv: `adj_t.to_symmetric()`, move to device, get splits | none |
| 31 | `GCN.__init__` + `GCN.forward` | **Q5** (~10 lines + forward) |
| 32 | `train()` for node classification | **Q5** (~4 lines) |
| 33 | `test()` for node classification (uses OGB `Evaluator`) | **Q5** (~1 line) |
| 34 | Hyperparameters (marked "you can change") | optional tuning |
| 36 | Training loop (marked "please do not change these args") | none |
| 38 | Prints best_model train/valid/test, writes `ogbn-arxiv_node.csv` | none |
| 41–42 | Loads `ogbg-molhiv`, builds DataLoaders (batch 32) | none |
| 43 | Hyperparameters (marked "you can change") | optional tuning |
| 47 | `GCN_Graph.__init__` (pool) + `GCN_Graph.forward` | **Q6** (~3 lines + pool) |
| 48 | `train()` for graph classification (masking NaN labels) | **Q6** (~3 lines) |
| 49 | `eval()` (uses OGB `Evaluator`) | none |
| 51 | Training loop (marked "please do not change these args") | none |
| 53 | Prints best_model train/valid/test ROC-AUC, writes CSVs | none |
| 54 | Q7 (optional): other global pooling layers | optional |

Things noticed in the scaffold that may bite us (not yet acted on):

- Notebook metadata says Python 3.7.3 / Colab, but cell 22 calls
  `torch.serialization.add_safe_globals`, which only exists in newer PyTorch (≥2.4).
  So the scaffold was updated for a recent torch; the metadata is just stale.
- `T.ToSparseTensor()` + `adj_t.to_symmetric()` are `torch_sparse.SparseTensor`
  features, so `torch_sparse` must be installed — historically the hardest PyG
  dependency to install, especially on Windows.
- Cell 41 re-imports `Evaluator` from `ogb.graphproppred`, shadowing the
  `ogb.nodeproppred` one from cell 27; `train` is redefined in cell 48; `eval`
  shadows Python's builtin. All fine if cells run top-to-bottom, but running them
  out of order can use the wrong one.
- Cell 41 uses `from torch_geometric.data import DataLoader`, which has been
  deprecated in favor of `torch_geometric.loader.DataLoader`. Unverified whether
  the current PyG version still accepts it — check when we get there.

---

## 1. Environment / Compute

The notebook stays in this repo and is edited locally. It runs on a **Google Colab GPU
runtime attached as the kernel inside VS Code**, so outputs are written straight into
the local `.ipynb` (and are visible to Claude for logging). Datasets are downloaded
on the Colab VM, not locally. No Kaggle pipeline needed: each training run is
estimated at <10 min on GPU.

Runtime observed: **Tesla T4**, PyTorch **2.11.0+cu128**, Python **3.13**. (Cell 4 is a
GPU-check cell added by us; all scaffold cell numbers below are +1 vs. the original.)

**Install-cell fix (cell 7).** First run failed: `Looking in links:
https://data.pyg.org/whl/torch-.11.0+cu128.html` → `No matching distribution found for
pyg_lib`. Two causes:
1. `${TORCH_VERSION}` bug in the scaffold. IPython's `!` expands `{TORCH_VERSION}` first
   (→ `$2.11.0+cu128`), then the shell interprets `$2` as an empty positional argument,
   leaving `torch-.11.0+cu128` — a URL that doesn't exist. Fix: use `{TORCH_VERSION}`.
2. Checked the PyG wheel index for `torch-2.11.0+cu128`: prebuilt Python-3.13 wheels exist
   for `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`, but **not**
   `torch_spline_conv` (pip would compile it from source with CUDA — slow/fragile).
   The notebook never uses `SplineConv`, so it was dropped (decision D2).

---

## 2. Task 1 — PyG & OGB basics (Q1–Q4)

**Q1 (cell 14):** `pyg_dataset.num_classes` and `pyg_dataset.num_features`.
- `num_classes` comes from the graph-level labels `y` (ENZYMES = 6 EC enzyme classes).
- `num_features` is the width of the node-feature matrix `x` (i.e. `x.shape[1]`),
  **not** the number of nodes. Note: `TUDataset` defaults to `use_node_attr=False`,
  which gives only the 3 one-hot node-*label* columns; with `use_node_attr=True`
  ENZYMES would expose 18 extra continuous attributes (21 total). The scaffold uses
  the default, so the expected answer is the default-config count.

**Q2 (cell 17):** `pyg_dataset[idx].y.item()`.
- Indexing the dataset returns one `Data` object (one graph). For graph
  classification its `y` is a 1-element tensor; `.item()` converts it to a
  Python int as the docstring requires.

**Q3 (cell 20):** `graph.is_undirected()` → `graph.num_edges // 2` (decision D3).
- PyG has no "undirected edge" type: `edge_index` is a `[2, E]` tensor of *directed*
  pairs, so an undirected edge u–v is stored twice (u→v and v→u). `num_edges` counts
  columns of `edge_index`, i.e. double-counts.
- `is_undirected()` verifies that for every (u→v) the reverse (v→u) also exists; only
  then is halving valid. If it's not undirected, we fall back to the raw directed count
  rather than halving something that isn't symmetric.
- Known limitation: a self-loop (u→u) is stored once, so halving would undercount it.
  **Unverified** whether graph 200 has self-loops — can confirm with
  `pyg_dataset[200].has_self_loops()` if we want to be certain.

**Q4 (cell 24):** `data.num_features`.
- Same attribute as Q1, but on a single `Data` object instead of a dataset. For
  ogbn-arxiv this is the 128-dim word-embedding vector per paper (slides give 128 as
  a sanity check).

---

## 3. Task 2 — Node property prediction on ogbn-arxiv (Q5)

Before implementing: cell 30 (`adj_t.to_symmetric()`) ran cleanly on Colab, so the
`torch_sparse` install is fully confirmed.

**Baseline implementation (option A architecture, per D4):**

- **`GCN.__init__` (cell 32):** `convs` = `ModuleList` of `num_layers` `GCNConv`s,
  `input_dim → hidden_dim`, `(num_layers − 2) × hidden_dim → hidden_dim`,
  `hidden_dim → output_dim`. `bns` = `num_layers − 1` × `BatchNorm1d(hidden_dim)` (none
  after the last conv — its output is class scores, not a hidden representation).
  `softmax` = `LogSoftmax(dim=1)` (normalize across classes, not across nodes).
  Limitation: assumes `num_layers ≥ 2` (with 1 it would still build 2 convs); fine for
  the scaffold's 3 and 5.
- **`GCN.forward` (cell 32):** for each hidden layer, `conv → bn → relu →
  F.dropout(training=self.training)`; then the final conv; `LogSoftmax` only if
  `return_embeds` is False (Task 3 reuses this class as a node encoder and needs raw
  embeddings).
- **`train()` (cell 33):** `zero_grad` → `out = model(data.x, data.adj_t)` →
  `loss_fn(out[train_idx], data.y[train_idx].squeeze(1))`.
- **`test()` (cell 34):** `out = model(data.x, data.adj_t)` — no slicing; the scaffold's
  evaluator code slices each split afterwards.

Concepts to be able to explain:
- **`.squeeze(1)`**: OGB labels are `[N, 1]`; `nll_loss` wants `[N]` class indices.
- **`LogSoftmax` + `nll_loss` ≡ `CrossEntropyLoss` on logits.** Kept separate so
  `return_embeds` can bypass the softmax.
- **`F.dropout(..., training=self.training)`**: the functional form is stateless and
  doesn't see `model.eval()`; without the flag, dropout would stay on at test time.
  (`nn.Dropout` as a module would handle this automatically.)
- **Full-batch, transductive training**: every step runs the entire 169k-node graph
  through the model; the loss only uses train nodes, but valid/test nodes' features and
  edges still participate in message passing. BatchNorm statistics are therefore
  computed over all nodes.
- **No random seed is set** in the scaffold, so repeated runs will give slightly
  different numbers — relevant when comparing baseline vs. residual later.

**Residual connections (Q5 experiment 2, per D5):**
- `GCN.__init__` gained a `residual=False` keyword argument; `forward` now computes
  each hidden block into `h`, then `x = x + h` if `residual` is on **and** `x.shape ==
  h.shape`, else `x = h`.
- With `num_layers=3` on arxiv, only the **middle** layer (256→256) gets a skip: the first
  (128→256) can't add its input back, and the last conv (→40 classes) is outside the loop.
  In Task 3 the GCN is built as 256→256→…→256, so there *every* hidden layer would match.
- The skip is added **after** dropout (`x + dropout(relu(bn(conv(x))))`). An alternative
  is to add before the ReLU (pre-activation style); not tried.
- Enabled only in cell 36 (`residual=True`). Default `False` means the baseline is still
  reproducible by removing that kwarg, and Task 3's `GCN_Graph` (which constructs `GCN`
  itself) is unaffected unless we decide to turn it on there too.
- Seed added at the same time (D6).

---

## 4. Task 3 — Graph property prediction on ogbg-molhiv (Q6)

**Notebook housekeeping (2026-09-30):** added "START HERE TO RE-RUN" comment blocks at the
top of cell 28 (Task 2) and cell 42 (Task 3), explaining that the two sections share
variable names (`dataset`, `data`, `split_idx`, `args`, `train`, `eval`, `Evaluator`) and
must each be re-run from their start cell. Added via a JSON edit that only prepended
source lines (verified by diff: 22 added lines, outputs untouched).

**CSV question (user):** the Q5/Q6 cells write `ogbn-arxiv_node.csv` /
`ogbg-molhiv_graph_{valid,test}.csv` and mention Gradescope — leftover from Stanford
CS224W. Slides' submission list = notebook + report PDF only, so CSVs aren't required.
They're written to the Colab VM's disk automatically; no action needed.

**Baseline implementation (D11, D12):**

- **Pipeline:** `x [N,9]` → `AtomEncoder` → `[N,256]` → `GCN(256,256,256, 5 layers,
  return_embeds=True)` → `[N,256]` → `global_mean_pool(·, batch)` → `[G,256]` →
  `Linear(256→1)` → one logit per graph.
- **Cell 48 `__init__`:** `self.pool = global_mean_pool` (a function, not a module — it
  has no parameters, which is also why `reset_parameters` doesn't need to touch it).
- **Cell 48 `forward`:** `node_emb = self.gnn_node(embed, edge_index)` →
  `graph_emb = self.pool(node_emb, batch)` → `out = self.linear(graph_emb)`.
  Note `GCN.forward(x, adj_t)` here receives `edge_index` instead of a SparseTensor —
  `GCNConv` accepts either connectivity format.
- **Cell 49 `train()`:** `zero_grad` → `out = model(batch)` →
  `loss_fn(out[is_labeled], batch.y[is_labeled].float())`.
- **Cell 44 args:** added `'seed': 6666`; **cell 51:** `torch.manual_seed(args['seed'])`
  before building the model (fixes init, `reset_parameters()`, dropout, and the
  `train_loader` shuffle order, since shuffling draws from the torch RNG when iterated).
- Hyperparameters otherwise at scaffold defaults: 5 layers, hidden 256, dropout 0.5,
  lr 0.001, 30 epochs, batch 32. Training loop (cell 52) untouched.

Concepts to be able to explain:
- **Mini-batching**: 32 molecules are merged into one big disconnected graph; `batch[i]`
  says which graph node `i` belongs to. Pooling must use it, or it would average across
  molecules.
- **`BCEWithLogitsLoss`** applies the sigmoid inside the loss (log-sum-exp trick → more
  numerically stable than sigmoid then BCE). ROC-AUC is rank-based, and sigmoid is
  monotonic, so evaluating on raw logits gives the same AUC.
- **`is_labeled = batch.y == batch.y`**: `NaN != NaN`, so this masks out missing labels.
  molhiv is single-task and (as far as we know) fully labeled, so it likely masks nothing
  here — the code stays general for multi-task OGB datasets with missing labels.
- **`.float()`**: OGB stores labels as ints; BCE needs float targets.
- **Scaffold's skip condition** `batch.batch[-1] == 0`: skips a batch containing a single
  graph (BatchNorm can't compute batch statistics reliably from it).
- **Why the 5-layer GCN here has no softmax**: `return_embeds=True` → last GCNConv output
  (256-dim, no BN/ReLU after it) is returned raw for pooling.

**Save gotcha (2026-09-30):** the first Task 3 run after the code edits crashed with the
old `'int' object has no attribute 'backward'` — VS Code was still holding the pre-edit
`train()` in its buffer. User re-ran with the reloaded code and got results, but they
only reached disk after an explicit Ctrl+S. Lesson: after Claude edits the .ipynb on disk,
reload in VS Code before running; after running, save before asking Claude to read.

---

## 5. Optional — Q7 pooling experiments / architecture changes

_(not started)_

---

## Decision Log

One entry per meaningful choice: options considered, what was picked, why.

**D1 — Compute platform (2026-09-30).** Options: local RTX 3060 (needed a CUDA torch +
Windows `torch_sparse` build — risky), Kaggle via HW2's pipeline (push/pull overhead
larger than the <10-min runs), Colab. **Chose Colab GPU as a remote kernel inside VS
Code**: Linux environment where PyG installs normally, GPU, and the executed notebook
lives locally so outputs are captured for the submission and the log.

**D2 — PyG extension packages (2026-09-30).** Options: (A) install only what's needed
(`torch_scatter`, `torch_sparse`); (B) scaffold's list minus `torch_spline_conv`;
(C) all five, fix only the `$` bug (risks a from-source CUDA build). **Chose B** — stays
closest to the original scaffold while using only prebuilt wheels.

**D3 — Q3 undirected edge counting (2026-09-30).** Options: (A) `is_undirected()` then
halve `num_edges`; (B) count `edge_index` columns with `src < dst`; (C) convert via
`to_networkx(to_undirected=True)` and count there. **Chose A** — it's what the scaffold's
hint ("PyG built-in functions") points to, and the explicit `is_undirected()` check
guards the halving assumption.

**R8 — Q6 baseline, mean pooling, seed 6666 (2026-09-30, Colab T4).** Defaults: 5 layers,
hidden 256, dropout 0.5, lr 0.001, 30 epochs, batch 32. 1029 train batches/epoch,
129 valid / 129 test batches (≈4.1k graphs each).
**Best model (epoch 26): Train 83.65%, Valid 79.90%, Test 75.86% ROC-AUC** → 15-pt band
(75–76). Need ≥ 77 for full credit (76–77 = 18).

| Epoch | Loss* | Train | Valid | Test |
|---|---|---|---|---|
| 1 | 0.5480 | 72.07 | 71.75 | 70.67 |
| 5 | 0.0351 | 77.28 | 77.78 | 73.48 |
| 10 | 0.0236 | 80.14 | 75.69 | 71.45 |
| 15 | 0.0478 | 81.31 | 79.40 | 73.81 |
| 21 | 0.0235 | 81.56 | **72.17** | 73.28 |
| 22 | 0.0178 | 82.20 | 74.72 | 75.57 |
| 26 | 0.0318 | 83.65 | **79.90** | 75.86 |
| 29 | 0.0219 | 84.12 | 79.06 | 75.84 |
| 30 | 0.0323 | 84.64 | 78.10 | 74.29 |

Observations:
- *The printed "Loss" is **not** an epoch average: `train()` returns `loss.item()` of the
  **last mini-batch only**. Values are bimodal (~0.01–0.06 vs ~0.53–0.68).
  *Hypothesis (unverified)*: the last batch is small (train size isn't a multiple of 32),
  and with only a few % positives, it either contains no positive (tiny loss) or one
  (large loss). Either way, the loss column says nothing about convergence here.
- The printout says "Train/Valid/Test: x%" but these are **ROC-AUC**, not accuracy
  (the loop reuses `*_acc` variable names; `dataset.eval_metric` = rocauc).
- **Valid is very noisy**: 72.17 ↔ 79.90 within 5 epochs. With ~4.1k graphs and few
  positives in each eval split, ROC-AUC depends on how a small number of positives are
  ranked → high variance. Best-model selection inherits that noise.
- **Train still rising at epoch 30** (84.64) → not converged.
- **Valid−test gap ~4 pts** at the best epoch (79.90 vs 75.86), much larger than arxiv's.
  Scaffold split: test molecules have different structural backbones from training
  molecules, so it measures out-of-distribution generalization.
- Train−valid gap ~3.8 pts.

**R9 — Q6 sum pooling (`global_add_pool`), seed 6666 (2026-10-01).**
**Best model (epoch 27): Train 82.15%, Valid 80.29%, Test 75.59% ROC-AUC** → 15-pt band.

| Pooling | Best ep | Train | Valid | Test | V−T |
|---|---|---|---|---|---|
| Mean (R8) | 26 | 83.65 | 79.90 | 75.86 | 4.04 |
| Sum (R9) | 27 | 82.15 | 80.29 | 75.59 | 4.70 |

- Essentially a tie with mean: valid +0.39, test −0.27 — both inside the epoch-to-epoch noise.
- **Much rockier start**: epoch-1 train 62.88 (mean: 72.07); valid swung 74.01 (ep 8) →
  64.51 (ep 11) → 66.25 (ep 21). Plausible cause: summed embeddings scale with atom count,
  so inputs to the final Linear vary widely in magnitude across molecule sizes.
- Slightly larger valid−test gap; size/count info didn't help on unfamiliar scaffolds.

**D13 — After R8 (2026-10-01).** Options: (A) sum / max pooling (doubles as Q7),
(B) pool all GCN layers (scaffold's own hint), (C) more epochs, (D) lr/dropout/hidden.
Claude recommended A; **user chose A**. R9 = sum pooling (`global_add_pool`), everything
else identical to R8 (same seed). R10 = max pooling next. Approach: swap the one
`self.pool` line between runs, so the notebook's printed Q6 output only shows the latest
run; earlier pooling results live in this log + the LaTeX report. (If we want Q7 visible
in the notebook itself, a separate comparison cell under the Q7 heading is an option —
not decided.)

**D11 — Q6 pooling (2026-09-30).** Options: (A) mean — size-invariant, "what the
molecule is made of", stable, matches OGB's reference GCN setup (from memory); (B) sum —
keeps counts / size information, more expressive in theory, magnitudes scale with
molecule size; (C) max — detects presence of a substructure anywhere, ignores counts,
sparse gradients. **Chose A (mean)** as the baseline; Q7 (optional) compares the others.

**D12 — Q6 process (2026-09-30).** Same as Q5: baseline with default args first, improve
only if test ROC-AUC < 77% (full-credit band). Seed 6666 added from the start this time.

**D4 — Q5 architecture strategy (2026-09-30).** Options: (A) exactly the scaffold figure;
(B) add residual/skip connections immediately; (C) implement A, run it as a baseline,
then add B as a separately logged experiment only if test < 72% (full-credit band).
**Chose C** — produces a baseline-vs-improvement comparison for the report, which the
grading slide explicitly invites, and keeps the `GCN` class reused by Task 3 unchanged
unless there's a measured reason to change it.

**D5 — Q5 next experiment after baseline (2026-09-30).** Baseline R2 test = 71.61%.
Options: (1) epochs 100 → 500 only; (2) residual connections only; (3) both. Claude
suggested (1) first because the curve was still rising and it leaves the Task 3 class
untouched; **user chose (2) residual only**, keeping hyperparameters identical to R2 so
the comparison isolates the architecture change. Implemented as an opt-in
`residual=False` constructor flag so the baseline and Task 3 stay unchanged by default.
**D6 — Random seed (2026-09-30, user request).** Added `'seed': 6666` to the Q5 args cell
(cell 35, "you can change") and `torch.manual_seed(args['seed'])` at the top of cell 36,
right before the model is built. Placement matters: the training cell (37, "do not
change") calls `model.reset_parameters()`, which re-draws weights from the RNG — seeding
in cell 36 fixes that draw as long as 36→37 run in order. `torch.manual_seed` seeds CPU
and all CUDA devices. 6666 reused from HW2 for consistency; chosen **before** seeing any
seeded result (not tuned).
Caveat: GPU sparse aggregation (`torch_sparse` scatter/spmm) uses atomic adds, which are
**not bit-for-bit deterministic** on CUDA, so reruns should be very close but may not match
exactly. Also: R2 (baseline) was unseeded, so R3 (residual, seeded) vs. R2 is still not a
same-seed comparison.

---

## Run Log

One entry per notebook run that produced numbers we might report.

**R1 — Task 1, Q1–Q4 (2026-09-30, Colab T4).** Cells 4–25 run top-to-bottom after the
cell-7 install fix (pyg_lib 0.9.0, torch_scatter 2.1.2, torch_sparse 0.6.18,
torch_cluster 1.6.3, all `+pt211cu128`). Cell 9 was also run; harmless (everything
already satisfied).

| Q | Answer | Raw output |
|---|---|---|
| Q1 | 6 classes, 3 features | `ENZYMES dataset has 6 classes` / `... 3 features` |
| Q2 | label 4 | `Data(edge_index=[2, 168], x=[37, 3], y=[1])` / `Graph with index 100 has label 4` |
| Q3 | 53 edges | `Graph with index 200 has 53 edges` |
| Q4 | 128 features | `Data(num_nodes=169343, x=[169343, 128], node_year=[169343, 1], y=[169343, 1], adj_t=[169343, 169343, nnz=1166243])` / `The graph has 128 features` |

Observations worth keeping for the report:
- Q2's labels are 0-indexed (ENZYMES classes are 0–5), so label 4 is the 5th EC class.
- Q4's `adj_t` has `nnz=1,166,243` — exactly the slides' edge count — because at this
  point the citation graph is still **directed** (paper cites paper). Cell 30 later calls
  `adj_t.to_symmetric()`, which adds reverse edges before GCN training.

**R2 — Q5 baseline GCN (2026-09-30, Colab T4).** Plain scaffold architecture (D4 option
A). Config: `num_layers=3, hidden_dim=256, dropout=0.5, lr=0.01, epochs=100`, Adam,
`nll_loss`, no seed.

**Best model (epoch 96): Train 73.53%, Valid 71.93%, Test 71.61%** → 18-pt band (71–72%).

Selected curve points:

| Epoch | Loss | Train | Valid | Test |
|---|---|---|---|---|
| 1 | 4.2458 | 12.29 | 23.51 | 21.84 |
| 10 | 1.3570 | 30.19 | 17.28 | 15.74 |
| 20 | 1.1744 | 58.06 | 56.84 | 57.89 |
| 40 | 1.0422 | 70.11 | 69.82 | 69.37 |
| 60 | 0.9846 | 72.03 | 70.72 | 69.52 |
| 80 | 0.9472 | 73.03 | 71.69 | 70.79 |
| 81 | 0.9451 | 72.89 | 70.82 | 68.98 |
| 96 | 0.9233 | 73.53 | **71.93** | 71.61 |
| 100 | 0.9155 | 73.56 | 71.69 | 70.63 |

Observations:
- **Not converged**: loss fell monotonically (almost) to epoch 100; valid still creeping up.
- **No overfitting**: train–valid gap ≈1.6 pts, both rising together → limited by
  training length/capacity, not memorization.
- **Test < valid consistently**: temporal split (train ≤2017, valid 2018, test 2019) —
  test papers are furthest from the training distribution.
- **Noisy**: test swings 1–2 pts epoch to epoch (e.g. 80→81: 70.79→68.98). lr=0.01
  full-batch is jumpy; best-model test has meaningful luck in it.
- **Epochs 4–11 anomaly**: valid fell 28%→17% while train loss kept decreasing.
  *Hypothesis (unverified)*: `test()` runs in eval mode using BatchNorm **running**
  statistics, which lag behind rapidly changing weights early in training.

Note: the user also ran Task 3 cells 42–52 once (before Task 3 is implemented).
molhiv downloaded fine; `from torch_geometric.data import DataLoader` still works but
emits a deprecation warning (answers the open question from the initial audit); cell 52
fails as expected (`'int' object has no attribute 'backward'` — TODO not written yet).

**R3 — Q5 residual GCN, seed 6666 (2026-09-30, Colab T4).** Same config as R2 plus
`residual=True` (skip on the middle 256→256 layer only) and `torch.manual_seed(6666)`.
Cells 28→39 re-run to undo Task 3's variable overwrites.

**Best model (epoch 100): Train 73.11%, Valid 71.59%, Test 70.57%** → 15-pt band (70–71%).

| | Best epoch | Final loss | Train | Valid | Test | Valid−Test |
|---|---|---|---|---|---|---|
| R2 baseline (unseeded) | 96 | 0.9155 | 73.53 | 71.93 | 71.61 | 0.32 |
| R3 residual (seed 6666) | 100 | 0.9357 | 73.11 | 71.59 | 70.57 | 1.02 |
| Δ (R3 − R2) | | | −0.42 | −0.34 | **−1.04** | |

Interpretation:
- **Worse on paper, but the difference is mostly within noise.** Valid differs by only
  0.34. Test differs by 1.04, but test swings 1–3 pts between adjacent epochs in both
  runs (R3 epochs 96→98: 70.55 → 67.48; epochs 90–100 span 67.48–70.57), and
  `best_model` is whichever epoch valid happened to peak on. One run each, different
  seeds → we **cannot conclude residuals hurt** — only that they didn't clearly help.
- **Plausible reasons for no gain**: at 3 layers only one layer gets a skip, and
  over-smoothing (what residuals mainly counter) is weak at depth 3. Residuals typically
  pay off when going deeper.
- **Both runs peaked at the very end (epoch 96 / 100)** with loss still falling → the
  common bottleneck is still training length, not architecture.
- R3's middle stretch was noisier (e.g. test 65.44% at epoch 47).
- R3's early dip (epochs 4–7, valid ~15–17%) mirrors R2's — consistent with the
  BatchNorm-running-stats hypothesis, still unverified.

**D7 — After R3 (2026-09-30).** Options offered: (A) seeded baseline for a same-seed
comparison, (B) epochs 100→500, (C) deeper + residual. **User chose: revert residual
connections entirely and increase epochs to 500** (≈ option B on the plain architecture).
- Cell 32: `residual` kwarg and skip logic **removed** — class is back to the exact R2
  baseline code (so Task 3's `GCN_Graph` uses the plain scaffold GCN).
- Cell 35: `epochs: 100 → 500`; `seed: 6666` kept (D6).
- Cell 36: `residual=True` removed; `torch.manual_seed` kept.
- Residual experiment (R3) stays in the report as a tried-and-reverted result.

**R4 — Q5 plain GCN, 500 epochs, seed 6666 (2026-09-30, Colab T4).**
**Best model (epoch 490): Train 79.81%, Valid 73.09%, Test 71.55%** → 18-pt band.

| Run | Best ep | Train | Valid | Test | Train−Valid | Valid−Test |
|---|---|---|---|---|---|---|
| R2 baseline, 100 ep | 96 | 73.53 | 71.93 | 71.61 | 1.60 | 0.32 |
| R3 residual, 100 ep | 100 | 73.11 | 71.59 | 70.57 | 1.52 | 1.02 |
| R4 plain, 500 ep | 490 | 79.81 | 73.09 | 71.55 | 6.72 | 1.54 |

21-epoch window averages (R4):

| Epochs | Loss | Train | Valid | Test | Test range |
|---|---|---|---|---|---|
| 90–110 | 0.911 | 73.74 | 71.51 | 70.14 | 69.20–71.22 |
| 190–210 | 0.814 | 76.11 | 71.80 | 70.34 | 66.96–71.76 |
| 290–310 | 0.759 | 77.68 | 72.01 | 70.38 | 66.23–72.14 |
| 390–410 | 0.724 | 78.82 | 72.14 | 70.33 | 68.50–71.49 |
| 480–500 | 0.702 | 79.61 | 72.47 | 70.69 | 69.09–71.68 |

Interpretation:
- **Train kept climbing (+6 pts), valid crept up (+1 pt), test stayed flat (~70.1–70.7
  window average).** Extra epochs were spent mostly fitting training nodes.
- **Overfitting? Partially.** Valid never turned downward, so it's not classic
  overfitting (where held-out accuracy falls). But the train−valid gap grew 1.6 → 6.7:
  a widening generalization gap, with diminishing returns on held-out data.
- **Valid gains didn't transfer to test.** Valid−test gap grew 0.32 → 1.54. Consistent
  with the temporal split: valid (2018) is closer in time to train (≤2017) than test
  (2019), so improvements that fit the older distribution help valid more than test.
- **Test is extremely noisy at lr=0.01**: swings of up to ~6 pts within a 21-epoch
  window (66.23–72.14). Test exceeded 72% at some epochs (72.14 @ 303, 72.13 @ 175), but
  selecting those would be choosing by test = leakage; valid-selected best is 71.55.
- Top-8 valid epochs (all ~72.9–73.1 valid) have test 71.05–71.94 — the selection
  lands in the ~71–72 band regardless of which near-peak epoch wins.
- Consistent with OGB's reported plain-GCN reference (~71.7 ± 0.3, from memory,
  unverified): we appear to be at the plain-GCN ceiling for this config.

**D8 — After R4 (2026-09-30).** Options: (A) lr 0.01→0.005, (B) dropout ↑, (C) LR schedule
(requires editing the "do not change" training cell), (D) stop at 18/20, (E) deeper ±
residual. Claude recommended A: the obstacle is selection noise (valid-chosen epoch lands
at a random point in a ~66–72% test range at lr=0.01); smaller steps should make
neighbouring epochs agree, args-only, single-variable vs. R4. **User chose A**: `lr: 0.005`,
500 epochs, seed 6666, plain architecture. Plan: if R5 doesn't clear 72%, take the best
honest result and move to Task 3.

**R5 — Q5 plain GCN, lr 0.005, 500 epochs, seed 6666 (2026-09-30, Colab T4).**
**Best model (epoch 391): Train 78.93%, Valid 73.18%, Test 71.83%** → 18-pt band.
Highest valid and highest valid-selected test of all runs so far.

| Run | Best ep | Train | Valid | Test |
|---|---|---|---|---|
| R2 baseline (lr .01, 100 ep, unseeded) | 96 | 73.53 | 71.93 | 71.61 |
| R3 residual (lr .01, 100 ep) | 100 | 73.11 | 71.59 | 70.57 |
| R4 (lr .01, 500 ep) | 490 | 79.81 | 73.09 | 71.55 |
| R5 (lr .005, 500 ep) | 391 | 78.93 | **73.18** | **71.83** |

Window averages (R5): 90–110 test 70.13 (range 68.71–71.01); 190–210 70.70; 290–310
70.78 (66.20–72.12); 390–410 **71.13** (69.23–72.14); 480–500 70.88.
- Top-10 valid epochs now have test 71.53–72.21 (R4: 71.05–71.94) — the
  near-peak region shifted up ~0.3–0.5 and is slightly tighter. Window ranges are
  still wide (one 66.20 dip), so noise is reduced only modestly.
- **19 epochs had test ≥ 72%** (max 72.24 @ 447, valid 72.88). None was selected
  because every one had valid < 73.18 (the epoch-391 peak). User asked why the Q5 cell
  shows 71.83 despite these — answer: `best_model` is chosen by valid
  (`if valid_acc > best_valid_acc` in cell 37), per the slides ("chosen by validation
  accuracy, not test"); selecting by test would leak the test set into model selection.
- Choosing R5 as the final Q5 config is itself validation-consistent: it has the highest
  best-valid of all runs (73.18), so we aren't picking the run by its test score.

**D9 — After R5 (2026-09-30).** Options offered: (1) dropout 0.5→0.6, (2) lr 0.005→0.003,
(3) hidden 256→512, (4) deeper + residual, (5) LR schedule / weight decay (needs the
"do not change" training cell). Claude recommended (1): every longer run shows the
train−valid gap widening while test stays flat, i.e. extra training fits training nodes
without transferring — dropout pushes directly against that; args-only, single variable
vs. R5. **User chose (1)**: `dropout: 0.6`, lr 0.005, 500 epochs, seed 6666.

**R6 — Q5 dropout 0.6, lr 0.005, 500 ep, seed 6666 (2026-09-30).**
**Best model (epoch 365): Train 77.12%, Valid 73.06%, Test 71.66%** → 18-pt band.
- vs R5: train −1.81, valid −0.12, test −0.17. Train−valid gap shrank 5.75 → **4.06** —
  dropout did what it's for (less fitting of training nodes) at almost no cost to valid,
  but test didn't move. Top-10 valid epochs: test 71.62–72.19 (R5: 71.53–72.21).
- Conclusion: every args-only config lands at ~71.6–71.8 valid-selected test. Plain
  3-layer GCN appears capped here regardless of epochs/lr/dropout.
- **User decided not to include the dropout change in the LaTeX report** (too small a
  change). Kept here for the dev record. Possible use: freed-up gap → room for capacity.

**D10 — After R6 (2026-09-30).** Options: (A) hidden 256→512 on top of dropout 0.6,
(B) Jumping Knowledge (combine all hidden layers' outputs before the classifier),
(C) deeper + residual, (D) LR schedule (training cell). **User chose: A first, then B
only if A doesn't clear 72% test.** R7 = `hidden_dim: 512`, dropout 0.6, lr 0.005,
500 epochs, seed 6666. Rationale: dropout 0.6 shrank the train−valid gap (5.75 → 4.06),
leaving room to add capacity without the overfitting width would cause at dropout 0.5.

**R7 — Q5 hidden 512, dropout 0.6, lr 0.005, 500 ep, seed 6666 (2026-09-30). ✅ FINAL Q5.**
Model: 128 → 512 → 512 → 40 (~350K params vs ~110K at 256).
**Best model (epoch 404): Train 80.36%, Valid 73.41%, Test 72.19%** → **20-pt band**.

| Run | Best ep | Train | Valid | Test | T−V | V−T |
|---|---|---|---|---|---|---|
| R2 baseline | 96 | 73.53 | 71.93 | 71.61 | 1.60 | 0.32 |
| R3 +residual (reverted) | 100 | 73.11 | 71.59 | 70.57 | 1.52 | 1.02 |
| R4 500 ep | 490 | 79.81 | 73.09 | 71.55 | 6.72 | 1.54 |
| R5 +lr .005 | 391 | 78.93 | 73.18 | 71.83 | 5.75 | 1.35 |
| R6 +dropout .6 | 365 | 77.12 | 73.06 | 71.66 | 4.06 | 1.40 |
| **R7 +hidden 512** | 404 | 80.36 | **73.41** | **72.19** | 6.95 | 1.22 |

- Top-10 valid epochs: test 71.80–72.32, **7 of 10 ≥ 72%** (R5: 71.53–72.21; R6:
  71.62–72.19). The whole near-peak region shifted up, not just the selected epoch.
- 33 epochs total had test ≥ 72 (max 72.40 @ 296) — irrelevant to selection, noted only.
- Gap rose back to 6.95 (more capacity → more fitting), but this time valid and test rose too.
- R7 is chosen as the final config because it has the **highest best-valid (73.41)** of
  all runs — validation-consistent, not test-picked.
- **Caveat:** margin over 72% is 0.19, below the epoch-to-epoch noise; GPU sparse ops are
  non-deterministic and a grader's rerun on other hardware may differ slightly. The
  submitted notebook must show this run's output (currently does).
- Jumping Knowledge (planned fallback in D10) **not needed**.
- User reversed the earlier call: dropout experiment (R6) **is** included in the LaTeX
  report, framed as the regularization step that enabled the wider model.
