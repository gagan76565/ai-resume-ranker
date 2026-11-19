"""Record the scoring baseline: run an engine on the bundled sample dataset and save every output.

    python benchmarks/record_baseline.py              # writes benchmarks/baseline/scoring-v2.0-samples.json
    python benchmarks/record_baseline.py --check      # compares the current engine against the saved file

The baseline was recorded BEFORE any Prompt-2 engine change, so later engines can
be compared with the exact original behaviour (scores, ranks, matched skills).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scoring import ScoringConfig  # noqa: E402
from talentlens import samples  # noqa: E402
from talentlens.screening import run_screening  # noqa: E402
from talentlens.uploads import UploadLimits, extract_text  # noqa: E402

OUT = ROOT / "benchmarks" / "baseline" / "scoring-v2.0-samples.json"


def current_output() -> dict:
    manifest = samples.manifest()
    docs = []
    for name, data in samples.sample_files():
        extraction = extract_text(data, UploadLimits())
        docs.append((name, extraction.text, extraction.warnings))
    result = run_screening(manifest["job_title"], manifest["job_description"], manifest["must_have"], docs, ScoringConfig())
    result.pop("created_at")
    return result


def fingerprint(result: dict) -> str:
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare against the saved baseline instead of writing it")
    args = parser.parse_args()
    result = current_output()
    if args.check:
        saved = json.loads(OUT.read_text(encoding="utf-8"))
        same = fingerprint(saved["result"]) == fingerprint(result)
        print(f"baseline {saved['fingerprint'][:16]} vs current {fingerprint(result)[:16]}: {'IDENTICAL' if same else 'DIFFERENT'}")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine": "TalentLens scoring v2.0 (scoring/ package as of commit b16e325)",
        "dataset": "samples/ (6 synthetic resumes + sample job description, must-have: Python, SQL)",
        "fingerprint": fingerprint(result),
        "summary": [
            {"rank": c["rank"], "filename": c["filename"], "final": c["score"], "tfidf": c["breakdown"]["tfidf_score"],
             "coverage": c["breakdown"]["skill_coverage"], "section": c["breakdown"]["section_signal"],
             "capped": c["breakdown"]["capped"], "low_text": c["breakdown"]["low_text"]}
            for c in result["candidates"]
        ],
        "result": result,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} fingerprint {payload['fingerprint'][:16]}")
    for row in payload["summary"]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
