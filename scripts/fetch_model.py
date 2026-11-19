"""Download the optional semantic model (MiniLM-L6-v2, ONNX, int8) into models/.

    python scripts/fetch_model.py            # ~23.5 MB, verified by SHA-256

Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0), pinned to a revision.
The files are not committed to Git (models/ is ignored). On Render, append this script to the build
command if you enable semantic matching: `pip install -r requirements.txt && python scripts/fetch_model.py`.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILES = {
    # published path -> (local name, sha256)
    "onnx/model_quint8_avx2.onnx": ("model.onnx", "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21"),
    "tokenizer.json": ("tokenizer.json", None),  # pinned after first verified download, see TOKENIZER_SHA256
}
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
DEST = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for remote, (local, expected) in FILES.items():
        expected = expected or TOKENIZER_SHA256
        target = DEST / local
        if target.exists() and sha256(target) == expected:
            print(f"ok      {local} (already present)")
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
        tmp = target.with_suffix(target.suffix + ".part")
        print(f"fetch   {remote}")
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:  # noqa: S310 - fixed https URL
            while chunk := response.read(1 << 20):
                out.write(chunk)
        actual = sha256(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            print(f"ERROR   checksum mismatch for {remote}: {actual}", file=sys.stderr)
            return 1
        tmp.replace(target)
        print(f"ok      {local} ({target.stat().st_size / 1e6:.1f} MB, sha256 verified)")
    (DEST / "SOURCE.txt").write_text(
        f"{REPO} @ {REVISION}\nLicence: Apache-2.0 (https://huggingface.co/{REPO})\nFiles: {', '.join(FILES)}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
