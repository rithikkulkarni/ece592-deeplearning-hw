"""Score generated-result files with local proxy metrics (PDR and FID).

Standalone version of the notebook's "Proxy Metrics" section (cells 41-45), so any number of
exp_N_results.txt files can be scored against the same detector and reference set.

    .venv/Scripts/python.exe score_proxies.py exp_1_results.txt exp_2_results.txt ...

These are local stand-ins, NOT the TAs' FID/PDR:
  - Proxy PDR: a small CNN trained on real images (label 1) vs. real images whose bottom 40% is
    corrupted (label 0). One detector is trained per invocation and used for every file.
  - Proxy FID: Frechet distance between Inception features of a file's 80 images and the 712 real
    train + dev images. With n=80 it is noisy and biased (see the real-vs-real floor row).
Use them to compare runs with each other only. The detector here trains on CPU (or GPU if present),
so its numbers will differ slightly from the one trained inside the notebook on Colab.
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

TAIL_START = 240  # tokens 240-399 (bottom 40% of the 20x20 image) are what the model generates
HERE = Path(__file__).parent


def to_rgb(seqs: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """Token sequences (N, 400) -> RGB images (N, 20, 20, 3), uint8."""
    return cmap[seqs].reshape(len(seqs), 20, 20, 3)


def to_tensor(seqs: np.ndarray, cmap: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(to_rgb(seqs, cmap)).float().div(255).permute(0, 3, 1, 2)


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


def corrupt(seq: np.ndarray, kind: int, rng: np.random.Generator, train_seqs: np.ndarray) -> np.ndarray:
    """Copy of a real image whose bottom 40% is replaced, giving a 'not a Pokemon' example.

    kind 0: one flat color (taken from the image itself)
    kind 1: noise (pixels drawn at random from the image's own top 60%)
    kind 2: the bottom of a different real training image
    """
    out = seq.copy()
    prefix = seq[:TAIL_START]
    if kind == 0:
        out[TAIL_START:] = rng.choice(prefix)
    elif kind == 1:
        out[TAIL_START:] = rng.choice(prefix, size=len(seq) - TAIL_START)
    else:
        out[TAIL_START:] = train_seqs[rng.integers(len(train_seqs))][TAIL_START:]
    return out


class PokemonDetector(nn.Module):
    """Small CNN on 20x20 RGB images -> one logit (positive = looks like a real Pokemon)."""

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


def train_detector(train_seqs, cmap, device, epochs, rng) -> PokemonDetector:
    detector = PokemonDetector().to(device)
    optimizer = optim.AdamW(detector.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        detector.train()
        # Real images (label 1) + a fresh random corruption of each one (label 0), reshuffled every epoch
        negatives = np.stack([corrupt(s, int(rng.integers(3)), rng, train_seqs) for s in train_seqs])
        x_all = np.concatenate([train_seqs, negatives])
        y_all = np.concatenate([np.ones(len(train_seqs)), np.zeros(len(negatives))])
        order = rng.permutation(len(x_all))
        epoch_loss = 0.0
        for i in range(0, len(x_all), 64):
            idx = order[i:i + 64]
            xb = to_tensor(x_all[idx], cmap).to(device)
            yb = torch.from_numpy(y_all[idx]).float().to(device)
            loss = criterion(detector(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0:
            print(f"Detector epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(x_all):.4f}")
    return detector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="result .txt files (80 lines x 400 ints each)")
    ap.add_argument("--csv", type=Path, default=HERE / "proxy_scores.csv", help="where to write the summary table")
    ap.add_argument("--det-epochs", type=int, default=30, help="detector training epochs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- Real data (same HF dataset the notebook uses) ---
    pokemon = load_dataset("lca0503/ml2025-hw4-pokemon")
    cmap = np.array(load_dataset("lca0503/ml2025-hw4-colormap")["train"]["color"], dtype=np.uint8)
    train_seqs = np.array(pokemon["train"]["pixel_color"])  # (632, 400)
    dev_seqs = np.array(pokemon["dev"]["pixel_color"])      # (80, 400)

    # --- Result files ---
    scored = {}
    for path in args.files:
        seqs = np.loadtxt(path, dtype=np.int64, ndmin=2)
        if seqs.shape != (80, 400) or seqs.min() < 0 or seqs.max() >= len(cmap):
            sys.exit(f"{path}: expected 80 x 400 ints in 0-{len(cmap) - 1}, got shape {seqs.shape}")
        scored[path.name] = seqs

    # --- Detector (one for all files) ---
    detector = train_detector(train_seqs, cmap, device, args.det_epochs, rng)

    @torch.no_grad()
    def detector_probs(seqs: np.ndarray) -> np.ndarray:
        detector.eval()
        return torch.sigmoid(detector(to_tensor(seqs, cmap).to(device))).cpu().numpy()

    print(f"Real dev images judged Pokemon:                {(detector_probs(dev_seqs) > 0.5).mean():.3f}  (want ~1)")
    for kind, name in enumerate(["flat color", "noise", "swapped tail"]):
        corrupted = np.stack([corrupt(s, kind, rng, train_seqs) for s in dev_seqs])
        print(f"Corrupted dev ({name:12s}) judged NOT Pokemon: {(detector_probs(corrupted) <= 0.5).mean():.3f}  (want ~1)")

    # --- Inception features ---
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

    rows = []
    for name, seqs in scored.items():
        blank = seqs.copy()
        blank[:, TAIL_START:] = 0  # control: the given prefix + a blank white bottom
        probs = detector_probs(seqs)
        rows.append({
            "file": name,
            "proxy_pdr": float((probs > 0.5).mean()),
            "mean_prob": float(probs.mean()),
            "proxy_fid": frechet_distance(inception_features(seqs), f_ref),
            "pdr_blank_control": float((detector_probs(blank) > 0.5).mean()),
            "fid_blank_control": frechet_distance(inception_features(blank), f_ref),
        })
    ceiling = float((detector_probs(dev_seqs) > 0.5).mean())
    floor = frechet_distance(f_dev, f_train)

    # --- Report ---
    header = f"{'file':26}{'proxy PDR':>10}{'mean prob':>11}{'proxy FID':>11}{'PDR blank':>11}{'FID blank':>11}"
    print("\n" + "=" * len(header))
    print("PROXY METRICS (local stand-ins; compare runs with each other only)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['file']:26}{r['proxy_pdr']:>10.3f}{r['mean_prob']:>11.3f}{r['proxy_fid']:>11.2f}"
              f"{r['pdr_blank_control']:>11.3f}{r['fid_blank_control']:>11.2f}")
    print("-" * len(header))
    print(f"{'real dev (ceiling/floor)':26}{ceiling:>10.3f}{'':>11}{floor:>11.2f}   <- PDR ceiling; FID floor = real dev vs real train (n=80)")
    print("=" * len(header))

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"file": "REAL_DEV_REFERENCE", "proxy_pdr": ceiling, "proxy_fid": floor})
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
