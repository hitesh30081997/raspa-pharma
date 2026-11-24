#!/usr/bin/env python3
"""
Evaluate an SLU model on a Household-SLU manifest: scenario / action / intent accuracy, slot precision,
recall and F1 (exact slot type + filler), exact-match rate, per-intent accuracy and the most frequent
intent confusions.

  python scripts/eval_household.py --model nemo_experiments/household_slu_ft.nemo --manifest manifests/test_slu.json
  python scripts/eval_household.py --model slu_conformer_transformer_large_slurp   # zero-shot baseline
  python scripts/eval_household.py --predictions preds.jsonl                        # score saved predictions

For SLURP-F1 with beam search use NeMo's run_speech_intent_slot_eval.py on the same manifest.
Writes <out>/predictions.jsonl and <out>/metrics.json.
"""
import argparse, ast, json, re
from collections import Counter, defaultdict
from pathlib import Path


def parse(s):
    """Semantics string -> (scenario, action, [(type, filler)]); tolerant of malformed output."""
    try:
        d = ast.literal_eval(s)
        ents = [(str(e.get("type", "")).strip(), str(e.get("filler", "")).strip().lower())
                for e in d.get("entities", []) if isinstance(e, dict)]
        return str(d.get("scenario", "")), str(d.get("action", "")), ents
    except Exception:
        sc = re.search(r"'scenario':\s*'([^']*)'", s)
        ac = re.search(r"'action':\s*'([^']*)'", s)
        ents = re.findall(r"'type':\s*'([^']*)',\s*'filler':\s*'([^']*)'", s)
        return (sc.group(1) if sc else ""), (ac.group(1) if ac else ""), [(t, f.lower()) for t, f in ents]


def score(pairs):
    n = len(pairs)
    sc = ac = it = em = 0
    tp = fp = fn = 0
    per_int = defaultdict(lambda: [0, 0])
    conf = Counter()
    slot_tp, slot_fp, slot_fn = Counter(), Counter(), Counter()
    for ref, hyp in pairs:
        rs, ra, re_ = parse(ref)
        hs, ha, he = parse(hyp)
        ri, hi = f"{rs}_{ra}", f"{hs}_{ha}"
        sc += rs == hs; ac += ra == ha; it += ri == hi
        per_int[ri][0] += ri == hi; per_int[ri][1] += 1
        if ri != hi:
            conf[(ri, hi)] += 1
        rc, hc = Counter(re_), Counter(he)
        t = sum((rc & hc).values())
        tp += t; fp += sum(hc.values()) - t; fn += sum(rc.values()) - t
        for k in rc | hc:
            m = min(rc[k], hc[k])
            slot_tp[k[0]] += m; slot_fp[k[0]] += hc[k] - m; slot_fn[k[0]] += rc[k] - m
        em += ri == hi and rc == hc
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    def f(t):
        pp = slot_tp[t] / (slot_tp[t] + slot_fp[t]) if slot_tp[t] + slot_fp[t] else 0.0
        rr = slot_tp[t] / (slot_tp[t] + slot_fn[t]) if slot_tp[t] + slot_fn[t] else 0.0
        return round(2 * pp * rr / (pp + rr), 4) if pp + rr else 0.0

    return {"utterances": n, "scenario_acc": sc / n, "action_acc": ac / n, "intent_acc": it / n,
            "slot_precision": p, "slot_recall": r, "slot_f1": f1, "exact_match": em / n,
            "per_intent_acc": {k: v[0] / v[1] for k, v in sorted(per_int.items())},
            "per_slot_f1": {t: f(t) for t in sorted(set(slot_tp) | set(slot_fn) | set(slot_fp)) if t},
            "top_confusions": [[a, b, c] for (a, b), c in conf.most_common(15)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help=".nemo path or pretrained model name")
    ap.add_argument("--manifest", default="manifests/test_slu.json")
    ap.add_argument("--predictions", help="score an existing predictions.jsonl instead of running a model")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="eval_out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.predictions:
        rows = [json.loads(l) for l in open(a.predictions)]
    else:
        import sys
        import nemo.collections.asr as nemo_asr
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import nemo_compat
        nemo_compat.apply()  # NeMo 2.x: SLURP classifier lacks .mlp
        M = nemo_asr.models.SLUIntentSlotBPEModel
        model = M.restore_from(a.model) if a.model.endswith(".nemo") else M.from_pretrained(model_name=a.model)
        model.eval()
        items = [json.loads(l) for l in open(a.manifest)]
        hyps = model.transcribe([x["audio_filepath"] for x in items], batch_size=a.batch_size)
        if isinstance(hyps, tuple):  # some versions return (best, all)
            hyps = hyps[0]
        hyps = [h.text if hasattr(h, "text") else str(h) for h in hyps]
        rows = [{"audio_filepath": x["audio_filepath"], "reference": x["text"], "prediction": h}
                for x, h in zip(items, hyps)]
        with open(out / "predictions.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    m = score([(r["reference"], r["prediction"]) for r in rows])
    json.dump(m, open(out / "metrics.json", "w"), indent=2)
    print(f"utterances     {m['utterances']}")
    for k in ("scenario_acc", "action_acc", "intent_acc", "slot_precision", "slot_recall", "slot_f1", "exact_match"):
        print(f"{k:15s}{100 * m[k]:6.2f}")
    worst = sorted(m["per_intent_acc"].items(), key=lambda x: x[1])[:5]
    print("weakest intents:", ", ".join(f"{k} {100 * v:.0f}%" for k, v in worst))
    print("top confusions:", m["top_confusions"][:5])
    print("details ->", out / "metrics.json")


if __name__ == "__main__":
    main()
