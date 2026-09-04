"""
Watches ECE592HW1_rrkulka3.ipynb for changes and auto-extracts pred.csv
whenever the notebook's cell-34 output contains a fresh base64-encoded
copy of it (see the PRED_CSV_BASE64_START/END markers printed by that cell).

Run once in a terminal and leave it running:
    python watch_pred.py

No external dependencies -- just polls the notebook's mtime.
"""
import base64
import json
import time
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "ECE592HW1_rrkulka3.ipynb"
OUTPUT = Path(__file__).parent / "pred.csv"
POLL_SECONDS = 2

START_MARKER = "PRED_CSV_BASE64_START"
END_MARKER = "PRED_CSV_BASE64_END"


def extract_pred_csv(nb_path: Path) -> bytes | None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if START_MARKER not in source:
            continue
        for out in cell.get("outputs", []):
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if START_MARKER in text and END_MARKER in text:
                lines = text.splitlines()
                start = lines.index(START_MARKER)
                end = lines.index(END_MARKER)
                b64 = "".join(lines[start + 1:end])
                return base64.b64decode(b64)
    return None


def main():
    print(f"Watching {NOTEBOOK.name} for pred.csv updates (Ctrl+C to stop)...")
    last_mtime = None
    last_written_bytes = None
    while True:
        try:
            mtime = NOTEBOOK.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                data = extract_pred_csv(NOTEBOOK)
                if data is not None and data != last_written_bytes:
                    OUTPUT.write_bytes(data)
                    last_written_bytes = data
                    print(f"[{time.strftime('%H:%M:%S')}] Wrote {OUTPUT.name} ({len(data)} bytes)")
        except (json.JSONDecodeError, FileNotFoundError):
            pass  # notebook mid-write or momentarily missing; just retry next poll
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
