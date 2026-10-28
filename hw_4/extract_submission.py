"""Extract a submission .txt from the saved notebook's cell output.

Looks for the text printed between ===BEGIN_SUBMISSION=== and ===END_SUBMISSION=== (cell 40), or,
with --label NAME, between ===BEGIN_SUBMISSION:NAME=== and ===END_SUBMISSION:NAME=== (the sampling
cells at the end of the notebook). Writes the text to a file and validates it: 80 lines, 400 ints
per line, values 0-166.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def find_output_text(nb: dict, begin: str) -> str:
    """Return the stream text of the last code cell whose output contains the BEGIN marker.

    All stream outputs of a cell are joined first, in case a long print was split into chunks.
    """
    text = None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join("".join(out["text"]) for out in cell.get("outputs", []) if out.get("output_type") == "stream")
        if begin in s:
            text = s
    if text is None:
        sys.exit(f"No {begin} marker found in any cell output. Did the cell run and the notebook get saved?")
    return text


def validate(lines: list[str]) -> list[str]:
    problems = []
    if len(lines) != 80:
        problems.append(f"expected 80 lines, got {len(lines)}")
    for i, line in enumerate(lines, 1):
        toks = line.split()
        if len(toks) != 400:
            problems.append(f"line {i}: {len(toks)} numbers (expected 400)")
            continue
        for t in toks:
            if not t.isdigit() or int(t) > 166:
                problems.append(f"line {i}: bad token {t!r}")
                break
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", type=Path, default=HERE / "ECE592HW4_rrkulka3.ipynb")
    ap.add_argument("--out", type=Path, default=HERE / "reconstructed_results.txt")
    ap.add_argument("--label", default=None, help="extract the block printed with this label (e.g. 6a_s0)")
    args = ap.parse_args()

    suffix = f":{args.label}" if args.label else ""
    begin, end_marker = f"===BEGIN_SUBMISSION{suffix}===", f"===END_SUBMISSION{suffix}==="

    nb = json.loads(args.notebook.read_text(encoding="utf-8-sig"))
    text = find_output_text(nb, begin)

    start = text.index(begin) + len(begin)
    end = text.find(end_marker, start)
    if end < 0:
        sys.exit(f"Found {begin} but no {end_marker} - the saved output is TRUNCATED.")

    lines = text[start:end].strip().splitlines()
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {args.out} ({len(lines)} lines)")

    problems = validate(lines)
    if problems:
        print("INVALID:")
        for p in problems[:15]:
            print(f"  {p}")
        sys.exit(1)
    print("VALID: 80 lines x 400 ints, all values in 0-166")


if __name__ == "__main__":
    main()
