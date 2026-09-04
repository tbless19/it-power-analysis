"""Download the professor's ExaData bundle from Google Drive into ./data.

Folder: https://drive.google.com/drive/folders/1M0t-VCc8n8SLASO934fr3h5tW7_aw46J

The folder must be shared as "Anyone with the link" (Viewer). Google sign-in
pages cannot be completed from this environment.
"""
from __future__ import annotations

import os
import sys

FOLDER_ID = "1M0t-VCc8n8SLASO934fr3h5tW7_aw46J"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(_ROOT, "data")


def main() -> int:
    try:
        import gdown
    except ImportError:
        print("Install gdown: pip install gdown", file=sys.stderr)
        return 2

    out = os.environ.get("EXADATA_ROOT") or DEFAULT_OUT
    os.makedirs(out, exist_ok=True)
    print(f"Downloading {FOLDER_URL}", flush=True)
    print(f" → {out}", flush=True)
    try:
        result = gdown.download_folder(
            id=FOLDER_ID,
            output=out,
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "The Drive folder is not publicly downloadable. In Google Drive: "
            "Share → General access → Anyone with the link → Viewer, then rerun "
            "`python -m rl_tune.fetch_drive_data`.",
            file=sys.stderr,
        )
        return 1
    if not result:
        print(
            "FAILED: empty download (folder is private or requires Google sign-in).",
            file=sys.stderr,
        )
        print(
            "Share the folder as Anyone with the link (Viewer) and rerun.",
            file=sys.stderr,
        )
        return 1
    print("Downloaded:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
