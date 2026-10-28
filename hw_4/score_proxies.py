"""Score generated-result files with local proxy metrics (PDR and FID).

Standalone, stronger version of the notebook's "Proxy Metrics" section (cells 41-45), so any number
of exp_N_results.txt files can be scored by the same detectors and reference set.

    .venv/Scripts/python.exe score_proxies.py exp_5_results.txt exp_6b_s0_results.txt ...

These are local stand-ins, NOT the TAs' FID/PDR:
  - Proxy PDR: two detectors trained to tell real images (label 1; horizontally flipped copies are
    added) from real images whose bottom 40% was corrupted (label 0). The cut is always at row 12,
    where the model's generated part begins. Seven corruption families exist; each detector trains
    on a different subset, so each is also tested on families it never saw.
      Detector A: small CNN, trained on flat / noise / swap / mirror / shift.
      Detector B: MLP on the flattened image, trained on flat / noise / swap / rowshuffle / smear.
    Each is trained with several seeds and its PDR reported as mean (std) over seeds.
  - Proxy FID: Frechet distance between Inception features of a file's 80 images and the 712 real
    train + dev images. With n=80 it is noisy and biased (see the real-vs-real floor row).
Use them to compare runs with each other only.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from pytorch_fid.inception import InceptionV3
from torch import nn, optim

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAIL_START = 240  # tokens 240-399 (rows 12-19, the bottom 40% of the 20x20 image) are what the model generates
HERE = Path(__file__).parent

ALL_FAMILIES = ["flat", "noise", "swap", "mirror", "shift", "rowshuffle", "smear"]
TRAIN_FAMILIES = {
    "A": ["flat", "noise", "swap", "mirror", "shift"],
    "B": ["flat", "noise", "swap", "rowshuffle", "smear"],
}


# ----------------------------------------------------------------------------- image helpers
def to_rgb(seqs: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """Token sequences (N, 400) -> RGB images (N, 20, 20, 3), uint8."""
    return cmap[seqs].reshape(len(seqs), 20, 20, 3)


def to_tensor(seqs: np.ndarray, cmap: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(to_rgb(seqs, cmap)).float().div(255).permute(0, 3, 1, 2)


def flip_h(seqs: np.ndarray) -> np.ndarray:
    """Mirror each 20x20 image left-to-right (token sequences in, token sequences out)."""
    return seqs.reshape(-1, 20, 20)[:, :, ::-1].reshape(len(seqs), 400)


def frechet_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    """Frechet distance between Gaussians fit to two sets of features (rows = samples).

    tr(sqrt(S1 S2)) is computed from the eigenvalues of S1 S2, which stays well-behaved when there
    are fewer samples than feature dimensions (80 vs 2048 here).
    """
    f1, f2 = f1.astype(np.float64), f2.astype(np.float64)
    mu1, mu2 = f1.mean(axis=0), f2.mean(axis=0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    eigvals = np.linalg.eigvals(s1 @ s2)
    tr_covmean = np.sqrt(np.clip(eigvals.real, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


# ----------------------------------------------------------------------------- corruptions
def corrupt(seq: np.ndarray, family: str, rng: np.random.Generator, train_seqs: np.ndarray) -> np.ndarray:
    """Copy of a real image whose bottom 40% (rows 12-19) is replaced, giving a 'not a Pokemon' example.

    flat:       the whole tail is one color taken from the image itself
    noise:      tail pixels drawn at random from the image's own top 60%
    swap:       the tail of a different real training image
    mirror:     the real tail, flipped left to right
    shift:      the real tail, rolled sideways by 4-16 columns (wrapping around)
    rowshuffle: the real tail rows in a random order
    smear:      the last real row (row 11) repeated down all 8 tail rows
    """
    out = seq.copy()
    prefix, tail = seq[:TAIL_START], seq[TAIL_START:].reshape(8, 20)
    if family == "flat":
        out[TAIL_START:] = rng.choice(prefix)
    elif family == "noise":
        out[TAIL_START:] = rng.choice(prefix, size=len(seq) - TAIL_START)
    elif family == "swap":
        out[TAIL_START:] = train_seqs[rng.integers(len(train_seqs))][TAIL_START:]
    elif family == "mirror":
        out[TAIL_START:] = tail[:, ::-1].reshape(-1)
    elif family == "shift":
        out[TAIL_START:] = np.roll(tail, int(rng.integers(4, 17)), axis=1).reshape(-1)
    elif family == "rowshuffle":
        out[TAIL_START:] = tail[rng.permutation(8)].reshape(-1)
    elif family == "smear":
        out[TAIL_START:] = np.tile(seq[TAIL_START - 20:TAIL_START], 8)
    else:
        raise ValueError(family)
    return out


def make_negative(seq: np.ndarray, families: list, rng: np.random.Generator, train_seqs: np.ndarray):
    """A corrupted copy of seq from a random allowed family, or None if every try left it unchanged.

    Corruptions can leave an image identical (e.g. mirroring an all-white tail); such a 'negative'
    would be a real image labeled fake, so it is rejected and another family is tried.
    """
    for _ in range(10):
        out = corrupt(seq, families[int(rng.integers(len(families)))], rng, train_seqs)
        if not np.array_equal(out, seq):
            return out
    return None


# ----------------------------------------------------------------------------- detectors
class CNNDetector(nn.Module):
    """Detector A: small CNN on 20x20 RGB images -> one logit (positive = looks like a real Pokemon)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),    # -> 10x10
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # -> 5x5
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 5 * 5, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class MLPDetector(nn.Module):
    """Detector B: MLP on the flattened 20x20x3 image (a different inductive bias from the CNN)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 20 * 3, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_detector(model: nn.Module, families: list, train_seqs, cmap, device, epochs, rng) -> nn.Module:
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    real = np.concatenate([train_seqs, flip_h(train_seqs)])  # positives: real images + mirrored copies
    for epoch in range(epochs):
        model.train()
        # Fresh random corruption of every real image each epoch (label 0), reshuffled
        negatives = [make_negative(s, families, rng, train_seqs) for s in real]
        negatives = np.stack([n for n in negatives if n is not None])
        x_all = np.concatenate([real, negatives])
        y_all = np.concatenate([np.ones(len(real)), np.zeros(len(negatives))])
        order = rng.permutation(len(x_all))
        for i in range(0, len(x_all), 64):
            idx = order[i:i + 64]
            xb = to_tensor(x_all[idx], cmap).to(device)
            yb = torch.from_numpy(y_all[idx]).float().to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def probs(model: nn.Module, seqs: np.ndarray, cmap, device) -> np.ndarray:
    """Probability that each image (rows of token ids) looks like a real Pokemon."""
    model.eval()
    return torch.sigmoid(model(to_tensor(seqs, cmap).to(device))).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="result .txt files (80 lines x 400 ints each)")
    ap.add_argument("--csv", type=Path, default=HERE / "proxy_scores.csv", help="where to write the summary table")
    ap.add_argument("--det-epochs", type=int, default=30, help="detector training epochs")
    ap.add_argument("--det-seeds", type=int, default=3, help="number of detector seeds to average over")
    ap.add_argument("--seed", type=int, default=0, help="first detector seed")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Real data (same HF dataset the notebook uses) ---
    pokemon = load_dataset("lca0503/ml2025-hw4-pokemon")
    cmap = np.array(load_dataset("lca0503/ml2025-hw4-colormap")["train"]["color"], dtype=np.uint8)
    train_seqs = np.array(pokemon["train"]["pixel_color"])  # (632, 400)
    dev_seqs = np.array(pokemon["dev"]["pixel_color"])      # (80, 400)
    dev_flipped = flip_h(dev_seqs)

    # --- Result files ---
    scored = {}
    for path in args.files:
        seqs = np.loadtxt(path, dtype=np.int64, ndmin=2)
        if seqs.shape != (80, 400) or seqs.min() < 0 or seqs.max() >= len(cmap):
            sys.exit(f"{path}: expected 80 x 400 ints in 0-{len(cmap) - 1}, got shape {seqs.shape}")
        scored[path.name] = seqs
    blanks = {}
    for name, seqs in scored.items():
        blank = seqs.copy()
        blank[:, TAIL_START:] = 0  # control: the given prefix + a blank white bottom
        blanks[name] = blank

    # Fixed corrupted dev images per family (same for every detector), for the test matrix
    dev_rng = np.random.default_rng(1234)
    dev_negatives = {}
    for fam in ALL_FAMILIES:
        negs = [make_negative(s, [fam], dev_rng, train_seqs) for s in dev_seqs]
        dev_negatives[fam] = np.stack([n for n in negs if n is not None])

    # --- Detectors: A (CNN) and B (MLP), several seeds each ---
    makers = {"A": CNNDetector, "B": MLPDetector}
    pdr = {k: {n: [] for n in scored} for k in makers}        # detector -> file -> [PDR per seed]
    prob = {k: {n: [] for n in scored} for k in makers}       # mean probability per seed
    pdr_blank = {k: {n: [] for n in scored} for k in makers}
    ceiling = {k: [] for k in makers}
    flip_accept = {k: [] for k in makers}
    matrix = {k: {f: [] for f in ALL_FAMILIES} for k in makers}  # rejection rate per family
    for s in range(args.seed, args.seed + args.det_seeds):
        for key, maker in makers.items():
            torch.manual_seed(s)
            rng = np.random.default_rng(s)
            model = train_detector(maker(), TRAIN_FAMILIES[key], train_seqs, cmap, device, args.det_epochs, rng)
            ceiling[key].append(float((probs(model, dev_seqs, cmap, device) > 0.5).mean()))
            flip_accept[key].append(float((probs(model, dev_flipped, cmap, device) > 0.5).mean()))
            for fam in ALL_FAMILIES:
                matrix[key][fam].append(float((probs(model, dev_negatives[fam], cmap, device) <= 0.5).mean()))
            for name in scored:
                p = probs(model, scored[name], cmap, device)
                pdr[key][name].append(float((p > 0.5).mean()))
                prob[key][name].append(float(p.mean()))
                pdr_blank[key][name].append(float((probs(model, blanks[name], cmap, device) > 0.5).mean()))
            print(f"seed {s} detector {key}: real dev accept {ceiling[key][-1]:.3f}")

    ms = lambda xs: (float(np.mean(xs)), float(np.std(xs)))

    print("\nDETECTOR CHECK on held-out dev images (mean over seeds; want ~1 everywhere)")
    print(f"{'':32}{'A (CNN)':>10}{'B (MLP)':>10}")
    print(f"{'real dev accepted':32}{ms(ceiling['A'])[0]:>10.3f}{ms(ceiling['B'])[0]:>10.3f}")
    print(f"{'flipped real dev accepted':32}{ms(flip_accept['A'])[0]:>10.3f}{ms(flip_accept['B'])[0]:>10.3f}")
    for fam in ALL_FAMILIES:
        cells = []
        for key in ("A", "B"):
            star = "*" if fam in TRAIN_FAMILIES[key] else " "
            cells.append(f"{ms(matrix[key][fam])[0]:>9.3f}{star}")
        print(f"{'rejected: ' + fam:32}{cells[0]}{cells[1]}")
    print("(* = family was in that detector's training set; unmarked = unseen family)")

    # --- Inception features / FID ---
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    @torch.no_grad()
    def inception_features(seqs: np.ndarray, batch_size: int = 64) -> np.ndarray:
        feats = []
        for i in range(0, len(seqs), batch_size):
            x = to_tensor(seqs[i:i + batch_size], cmap).to(device)
            feats.append(inception(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(feats)

    f_train, f_dev = inception_features(train_seqs), inception_features(dev_seqs)
    f_ref = np.concatenate([f_train, f_dev])  # real reference set: 712 images
    floor = frechet_distance(f_dev, f_train)

    rows = []
    for name in scored:
        a_m, a_s = ms(pdr["A"][name])
        b_m, b_s = ms(pdr["B"][name])
        rows.append({
            "file": name,
            "pdr_A": a_m, "pdr_A_std": a_s, "pdr_B": b_m, "pdr_B_std": b_s,
            "proxy_fid": frechet_distance(inception_features(scored[name]), f_ref),
            "pdr_A_blank": ms(pdr_blank["A"][name])[0], "pdr_B_blank": ms(pdr_blank["B"][name])[0],
            "fid_blank": frechet_distance(inception_features(blanks[name]), f_ref),
        })

    # --- Report ---
    header = (f"{'file':26}{'PDR A':>14}{'PDR B':>14}{'proxy FID':>11}{'A blank':>9}{'B blank':>9}{'FID blank':>11}")
    print("\n" + "=" * len(header))
    print(f"PROXY METRICS (local stand-ins; compare runs with each other only)  PDR = mean (std) over {args.det_seeds} detector seeds")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['file']:26}{r['pdr_A']:>8.3f} ({r['pdr_A_std']:.2f}){r['pdr_B']:>8.3f} ({r['pdr_B_std']:.2f})"
              f"{r['proxy_fid']:>11.2f}{r['pdr_A_blank']:>9.3f}{r['pdr_B_blank']:>9.3f}{r['fid_blank']:>11.2f}")
    print("-" * len(header))
    print(f"{'real dev (ceiling / floor)':26}{ms(ceiling['A'])[0]:>14.3f}{ms(ceiling['B'])[0]:>14.3f}{floor:>11.2f}"
          "   <- PDR ceiling = real dev accepted; FID floor = real dev vs real train (n=80)")
    print("=" * len(header))

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"file": "REAL_DEV_REFERENCE", "pdr_A": ms(ceiling["A"])[0], "pdr_B": ms(ceiling["B"])[0],
                         "proxy_fid": floor})
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
