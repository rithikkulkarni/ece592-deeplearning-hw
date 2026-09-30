"""
Sync ECE592HW2_rrkulka3.ipynb between this local VS Code checkout and a Kaggle
notebook (kernel), using the `kaggle` CLI under the hood.

Reads the kernel id (username/kernel-slug) from kernel-metadata.json, which
must live alongside this script and the notebook per the Kaggle CLI's
push convention.

Usage:
    python kaggle_sync.py push                 # upload notebook, start a run
    python kaggle_sync.py status                # one-shot status check
    python kaggle_sync.py pull [-o OUTPUT_DIR]  # download output notebook/files
    python kaggle_sync.py run [-o OUTPUT_DIR]   # push, poll until done, pull
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
METADATA_PATH = REPO_DIR / "kernel-metadata.json"
DEFAULT_OUTPUT_DIR = REPO_DIR / "kaggle_output"

TERMINAL_STATUSES = {"complete", "error", "cancelAcknowledged"}


def kernel_id() -> str:
    metadata = json.loads(METADATA_PATH.read_text())
    return metadata["id"]


def run_cli(args: list[str]) -> str:
    result = subprocess.run(
        ["kaggle", *args], cwd=REPO_DIR, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def push() -> None:
    run_cli(["kernels", "push", "-p", str(REPO_DIR)])


def status() -> str:
    return run_cli(["kernels", "status", kernel_id()])


def pull(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclude food11.zip: it's re-created by the notebook's own download cell on
    # every Kaggle run, and re-pulling the full 1.1GB copy each time is slow and
    # unnecessary since we already have the dataset locally.
    run_cli([
        "kernels", "output", kernel_id(), "-p", str(output_dir),
        "--file-pattern", r"^(?!food11\.zip$).*",
    ])


def wait_until_done(poll_interval: int, timeout: int) -> None:
    elapsed = 0
    while elapsed <= timeout:
        output = status()
        if any(term in output for term in TERMINAL_STATUSES):
            return
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise SystemExit(f"Timed out after {timeout}s waiting for kernel to finish.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("push")
    subparsers.add_parser("status")

    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--poll-interval", type=int, default=60, help="seconds between status checks")
    run_parser.add_argument("--timeout", type=int, default=3600 * 13, help="max seconds to wait")

    args = parser.parse_args()

    if args.command == "push":
        push()
    elif args.command == "status":
        status()
    elif args.command == "pull":
        pull(args.output_dir)
    elif args.command == "run":
        push()
        wait_until_done(args.poll_interval, args.timeout)
        pull(args.output_dir)


if __name__ == "__main__":
    main()
