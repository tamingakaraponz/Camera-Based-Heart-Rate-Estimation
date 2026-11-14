"""Explore a fixed, unverified 34/32 multiplier on reference BPM only.

Does not change workbook timestamps, amplitudes, or the production pipeline.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from benchmark_background_methods import analyze_contact_ppg


ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "outputs/first-three-demo-20261102"
out = folder / "cppg_time_hypothesis_34_over_32.json"
if out.exists():
    raise FileExistsError(f"Refusing to overwrite: {out}")
baseline = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
factor = 34.0 / 32.0
records = []
for old in baseline["records"]:
    code = old["data_id"]
    source = ROOT / "DATAcppg" / (code + ".xlsx")
    before = sha256(source.read_bytes()).hexdigest()
    contact = analyze_contact_ppg(source, 0.6, 2.0, filter_order=3)
    ippg = old["pos_bpm"]
    original = float(contact.heart_rate_bpm)
    hypothetical = original * factor
    records.append({
        "data_id": code, "ippg_bpm": ippg,
        "cppg_original_bpm": original,
        "cppg_hypothetical_bpm": hypothetical,
        "absolute_difference_before_bpm": abs(ippg-original),
        "absolute_difference_after_bpm": abs(ippg-hypothetical),
        "difference_reduction_bpm": abs(ippg-original)-abs(ippg-hypothetical),
        "source_sha256": before,
        "source_unchanged": sha256(source.read_bytes()).hexdigest() == before,
    })
result = {
    "purpose": "exploratory sensitivity check; not accuracy validation",
    "factor": factor, "factor_origin": "Previously proposed, unverified 34 s versus 32 s time-stretch hypothesis; not fitted to these BPM outputs",
    "operation": "multiply calculated full-workbook cPPG BPM by factor only; not raw amplitude and not a rerun on corrected timestamps",
    "reference_status": "reference_time_unverified",
    "scope": "3 recordings of one participant; same iPPG result as saved baseline, independent full-workbook cPPG interval",
    "records": records,
    "mean_absolute_difference_before_bpm": float(np.mean([r["absolute_difference_before_bpm"] for r in records])),
    "mean_absolute_difference_after_bpm": float(np.mean([r["absolute_difference_after_bpm"] for r in records])),
    "caveats": [
        "Closer numerical agreement does not prove the time correction is valid.",
        "No time alignment, trimming, or production config was changed.",
        "Not a recalculated corrected-time cPPG pipeline: band-pass/resampling/peak selection may behave differently if timestamps are changed.",
        "Do not apply this factor to the other 57 recordings as a validated correction.",
    ],
}
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
