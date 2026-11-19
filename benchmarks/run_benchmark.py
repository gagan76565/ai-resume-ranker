"""Run the TalentLens benchmark and write benchmarks/results/latest.{json,md}.

    python benchmarks/run_benchmark.py                 # v2, v3-lexical, v3-hybrid (if the model is present)
    SEMANTIC_ENABLED=0 python benchmarks/run_benchmark.py   # skip the hybrid engine

Every number in AI_BENCHMARK_REPORT.md comes from this script's output.
"""

from __future__ import annotations

import io
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import numpy as np  # noqa: E402
from dataset import TRAPS, build_dataset, self_check  # noqa: E402

from scoring import rank_candidates  # noqa: E402
from scoring import semantic as semantic_mod  # noqa: E402
from scoring.context import is_negated  # noqa: E402
from scoring.engine_v3 import ConfigV3, ResumeInput, rank_candidates_v3  # noqa: E402
from scoring.sections import detect_sections, heading_label  # noqa: E402
from scoring.skills import load_taxonomy  # noqa: E402
from talentlens.documents import parse_document  # noqa: E402
from talentlens.minipdf import build_pdf  # noqa: E402
from talentlens.uploads import UploadLimits  # noqa: E402

OUT = ROOT / "benchmarks" / "results"
LIMITS = UploadLimits()


def docx_bytes(lines):
    import docx

    document = docx.Document()
    for line in lines:
        document.add_paragraph(line[2:] if line.startswith("# ") else line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def words(text):
    return [w.lower() for w in text.split() if any(ch.isalnum() for ch in w)]


def token_recall(source, extracted):
    src, got = words(source), set(words(extracted))
    return sum(1 for w in src if w in got) / len(src) if src else 1.0


def ndcg_at(labels_in_rank_order, k):
    def dcg(labels):
        return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(labels[:k]))

    ideal = dcg(sorted(labels_in_rank_order, reverse=True))
    return dcg(labels_in_rank_order) / ideal if ideal else 0.0


def pairwise_accuracy(scores, labels):
    good = total = 0.0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if labels[i] == labels[j]:
                continue
            total += 1
            hi, lo = (i, j) if labels[i] > labels[j] else (j, i)
            good += 1.0 if scores[hi] > scores[lo] else 0.5 if scores[hi] == scores[lo] else 0.0
    return good / total if total else 1.0


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> int:
    taxonomy = load_taxonomy()
    dataset = build_dataset()
    problems = self_check(dataset, taxonomy)
    if problems:
        print("Dataset self-check failed:", problems)
        return 1

    # ------------------------------------------------------------ document processing
    docs, doc_stats = {}, {"pdf_ok": 0, "docx_ok": 0, "total": 0, "section_exact": 0, "section_jaccard": [], "with_headings": 0}
    for _role, _spec, resumes in dataset:
        for r in resumes:
            doc_stats["total"] += 1
            pdf = parse_document(r.id + ".pdf", build_pdf(r.lines), LIMITS)
            docs[r.id] = pdf
            if pdf.quality.evaluable and token_recall(r.text, pdf.text) >= 0.95:
                doc_stats["pdf_ok"] += 1
            dx = parse_document(r.id + ".docx", docx_bytes(r.lines), LIMITS)
            if dx.quality.evaluable and token_recall(r.text, dx.text) >= 0.95:
                doc_stats["docx_ok"] += 1
            gold = {heading_label(line[2:]) for line in r.lines[1:] if line.startswith("# ")} - {""}
            if gold:
                doc_stats["with_headings"] += 1
                got = {s.label or s.kind for s in detect_sections(pdf.text)}
                doc_stats["section_exact"] += int(got == gold)
                doc_stats["section_jaccard"].append(len(got & gold) / len(got | gold))

    # ------------------------------------------------------------ skill extraction (v2 extractor vs v3 negation-aware)
    trap_skills = {m.canonical for m in taxonomy.find(" ".join(TRAPS))}
    counts = {"v2": [0, 0, 0], "v3": [0, 0, 0]}
    trap_fp = {"v2": 0, "v3": 0}
    for _role, _spec, resumes in dataset:
        for r in resumes:
            text = docs[r.id].text
            found = taxonomy.find(text)
            v2 = {m.canonical for m in found}
            v3 = {m.canonical for m in found if not is_negated(text, m.start, m.end)}
            for name, pred in (("v2", v2), ("v3", v3)):
                counts[name][0] += len(pred & r.gold_skills)
                counts[name][1] += len(pred - r.gold_skills)
                counts[name][2] += len(r.gold_skills - pred)
                trap_fp[name] += len((pred - r.gold_skills) & trap_skills)

    # ------------------------------------------------------------ engines
    encoder = semantic_mod.get_encoder()
    engines = ["v2", "v3-lexical"] + (["v3-hybrid"] if encoder is not None else [])
    results = {}
    component_rows = []
    for engine in engines:
        per_role, timing, mandatory_hits, mandatory_total, musthave_hits = [], 0.0, 0, 0, 0
        evidence = {"supported": 0, "attributed": 0, "quotes": 0, "bad_quotes": 0, "covered": 0, "fp_covered": 0}
        stable = True
        for _role, spec, resumes in dataset:
            def score(order, _spec=spec, _engine=engine):
                if _engine == "v2":
                    ranked = rank_candidates(_spec["jd"], [(r.id, docs[r.id].text) for r in order], must_have=_spec["must_have"])
                    return {x["name"]: x for x in ranked}
                inputs = [ResumeInput(r.id, docs[r.id].text, docs[r.id].page_at, docs[r.id].quality.score, docs[r.id].quality.to_dict())
                          for r in order]
                enc = encoder if _engine == "v3-hybrid" else None
                ranked, _meta = rank_candidates_v3(_spec["jd"], inputs, _spec["must_have"], ConfigV3.from_env({}), encoder=enc)
                return {x["name"]: x for x in ranked}

            t0 = time.perf_counter()
            out = score(resumes)
            timing += time.perf_counter() - t0
            shuffled = resumes[:]
            random.Random(11).shuffle(shuffled)
            again = score(shuffled)
            stable &= all(out[k]["final_score"] == again[k]["final_score"] and out[k]["rank"] == again[k]["rank"] for k in out)
            ordered = sorted(resumes, key=lambda r: out[r.id]["rank"])
            labels = [r.relevance for r in ordered]
            scores = [out[r.id]["final_score"] for r in resumes]
            per_role.append({
                "role": _role, "ndcg@5": ndcg_at(labels, 5), "p@3": sum(1 for x in labels[:3] if x >= 2) / 3,
                "pairwise": pairwise_accuracy(scores, [r.relevance for r in resumes]),
                "top3": [r.archetype for r in ordered[:3]],
            })
            for r in resumes:
                item = out[r.id]
                met = item["mandatory"]["met"] if engine != "v2" else not item["must_have_missing"]
                mandatory_hits += int(met == r.gold_mandatory_met)
                mandatory_total += 1
                if engine != "v2":  # like-for-like with v2: recruiter must-haves only
                    must = [q for q in item["requirements"] if q["id"] in {x["id"] for x in item["requirements"]} and q["cls"] == "mandatory"
                            and q["source"] is None]
                    met_must = all(q["status"] in {"supported", "unclear"} for q in must)
                    musthave_hits += int(met_must == r.gold_mandatory_met)
                else:
                    musthave_hits += int(met == r.gold_mandatory_met)
                if engine == "v2":
                    continue
                if engine == "v3-lexical":
                    component_rows.append([item["components"][k]["value"] or 0.0 for k in ("required", "experience", "projects", "lexical")])
                text = docs[r.id].text
                for req in item["requirements"]:
                    for ev in req["evidence"]:
                        evidence["quotes"] += 1
                        if text[ev["start"]:ev["end"]] != ev["quote"] or ev["surface"] not in ev["quote"]:
                            evidence["bad_quotes"] += 1
                    if req["status"] in {"supported", "unclear"}:
                        evidence["covered"] += 1
                        if not set(req["skills"]) & r.gold_skills:
                            evidence["fp_covered"] += 1
                    if req["status"] == "supported":
                        gold_sec = next((r.gold_sections[s] for s in req["skills"] if s in r.gold_sections), None)
                        if gold_sec is not None:
                            evidence["supported"] += 1
                            evidence["attributed"] += int(req["evidence"][0]["section"] == gold_sec)
        results[engine] = {
            "ndcg@5": float(np.mean([x["ndcg@5"] for x in per_role])),
            "p@3": float(np.mean([x["p@3"] for x in per_role])),
            "pairwise": float(np.mean([x["pairwise"] for x in per_role])),
            "mandatory_accuracy": mandatory_hits / mandatory_total,
            "musthave_accuracy": musthave_hits / mandatory_total,
            "deterministic_and_order_invariant": stable,
            "ms_per_resume": 1000 * timing / sum(len(r) for _, _, r in dataset),
            "per_role": per_role,
            "evidence": None if engine == "v2" else {
                "attribution_accuracy": evidence["attributed"] / evidence["supported"] if evidence["supported"] else None,
                "supported_with_gold_section": evidence["supported"],
                "unsupported_quote_rate": evidence["bad_quotes"] / evidence["quotes"] if evidence["quotes"] else 0.0,
                "quotes_checked": evidence["quotes"],
                "false_positive_match_rate": evidence["fp_covered"] / evidence["covered"] if evidence["covered"] else 0.0,
                "covered_verdicts": evidence["covered"],
            },
        }

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(np.array(component_rows).T).round(2).tolist() if component_rows else None
    summary = {
        "dataset": {"pairs": sum(len(r) for _, _, r in dataset), "roles": len(dataset), "annotation": "by construction (author), synthetic"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "semantic_model": semantic_mod.status()},
        "documents": {
            "pdf_extraction_success": doc_stats["pdf_ok"] / doc_stats["total"],
            "docx_extraction_success": doc_stats["docx_ok"] / doc_stats["total"],
            "section_set_exact_match": doc_stats["section_exact"] / doc_stats["with_headings"],
            "section_jaccard_mean": float(np.mean(doc_stats["section_jaccard"])),
            "resumes_with_headings": doc_stats["with_headings"],
        },
        "skill_extraction": {
            name: dict(zip(("precision", "recall", "f1"), prf(*c), strict=True)) | {"tp": c[0], "fp": c[1], "fn": c[2], "trap_fp": trap_fp[name]}
            for name, c in counts.items()
        },
        "engines": results,
        "v3_component_correlation": {"components": ["required", "experience", "projects", "lexical"], "matrix": corr},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")

    lines = ["| Engine | NDCG@5 | P@3 | Pairwise acc. | Must-have acc. | Mandatory flag vs must-have gold | Deterministic | ms/resume |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for engine, res in results.items():
        lines.append(f"| {engine} | {res['ndcg@5']:.3f} | {res['p@3']:.3f} | {res['pairwise']:.3f} | "
                     f"{res['musthave_accuracy']:.3f} | {res['mandatory_accuracy']:.3f} | "
                     f"{'yes' if res['deterministic_and_order_invariant'] else 'NO'} | {res['ms_per_resume']:.0f} |")
    (OUT / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "engines"}, indent=1))
    print("\n".join(lines))
    for engine, res in results.items():
        print(engine, "evidence:", res["evidence"])
        for role in res["per_role"]:
            print("  ", engine, role["role"], f"ndcg@5={role['ndcg@5']:.3f} pairwise={role['pairwise']:.3f}", role["top3"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
