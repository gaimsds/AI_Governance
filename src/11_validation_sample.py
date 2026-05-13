"""
Validation sample creation for stance classifier evaluation.

Draws 400 documents across 4 strata (100 each):
  A — Academic, pub_year 2015–2021
  B — Academic, pub_year 2022–2025
  C — Policy, pub_year 2017–2021
  D — Policy, pub_year 2022–2025

Within each stratum: 35 risk_focused, 35 opportunity_focused, 30 balanced.
If a class is short, rebalances proportionally from available rows.

Outputs:
  results/validation/validation_sample.csv          — full internal file
  results/validation/validation_sample_for_coders.csv — coder-facing (no metadata)
"""

import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
RESULTS   = BASE_DIR / "results"
VAL_DIR   = RESULTS / "validation"

# Targets per (stratum × class)
CLASS_TARGETS = {
    "risk_focused":       35,
    "opportunity_focused": 35,
    "balanced":           30,
}
STRATUM_TOTAL = sum(CLASS_TARGETS.values())  # 100


def _sample_stratum(df: pd.DataFrame, stratum_label: str, rng: np.random.Generator) -> pd.DataFrame:
    """Sample CLASS_TARGETS rows per stance class from df; rebalance if short."""
    rows = []
    for stance, target in CLASS_TARGETS.items():
        pool = df[df["predicted_stance"] == stance]
        n = min(target, len(pool))
        if n < target:
            print(f"  WARNING [{stratum_label}] {stance}: only {len(pool)} available, "
                  f"sampling {n} (target was {target})")
        sampled = pool.sample(n=n, random_state=SEED)
        rows.append(sampled)
    return pd.concat(rows, ignore_index=True)


def main():
    print("=" * 65)
    print("VALIDATION SAMPLE CREATION")
    print("=" * 65)

    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load academic corpus ───────────────────────────────────────────
    print("\n[1] Loading academic corpus...")
    acad = pd.read_csv(RESULTS / "governance_papers_stance.csv")
    print(f"  Loaded {len(acad):,} academic papers")

    # Rename for uniform schema
    acad = acad.rename(columns={
        "scopus_id":      "original_id",
        "abstract_clean": "text",
        "stance":         "predicted_stance",
        "cover_date":     "pub_year_raw",
    })
    # pub_year: parse from cover_date or use cover_date directly
    if "pub_year" not in acad.columns:
        acad["pub_year"] = pd.to_datetime(
            acad["pub_year_raw"], errors="coerce"
        ).dt.year
    acad["corpus_type"] = "academic"
    acad["text"] = acad["text"].fillna(acad.get("abstract", ""))

    # ── Load policy corpus ─────────────────────────────────────────────
    print("\n[2] Loading policy corpus...")
    chunks = pd.read_csv(DATA_DIR / "policy_chunks_clean.csv")
    stance = pd.read_csv(RESULTS / "policy_stance_temporal.csv")
    print(f"  Clean chunks: {len(chunks):,}  |  Stance rows: {len(stance):,}")

    policy = chunks.merge(
        stance[["chunk_id", "stance", "stance_confidence",
                "score_risk", "score_opportunity", "score_balanced"]],
        on="chunk_id",
        how="inner",
    )
    print(f"  Joined policy rows: {len(policy):,}")

    policy = policy.rename(columns={
        "chunk_id":   "original_id",
        "chunk_text": "text",
        "stance":     "predicted_stance",
    })
    policy["corpus_type"] = "policy"
    # pub_year already present from policy_chunks_clean.csv

    # ── Build strata ───────────────────────────────────────────────────
    strata = {
        "A": acad[(acad["pub_year"] >= 2015) & (acad["pub_year"] <= 2021)].copy(),
        "B": acad[(acad["pub_year"] >= 2022) & (acad["pub_year"] <= 2025)].copy(),
        "C": policy[(policy["pub_year"] >= 2017) & (policy["pub_year"] <= 2021)].copy(),
        "D": policy[(policy["pub_year"] >= 2022) & (policy["pub_year"] <= 2025)].copy(),
    }

    print("\n[3] Stratum sizes:")
    for label, df in strata.items():
        print(f"  Stratum {label}: {len(df):,} total")
        for stance_class in CLASS_TARGETS:
            n = (df["predicted_stance"] == stance_class).sum()
            print(f"    {stance_class}: {n}")

    # ── Sample ────────────────────────────────────────────────────────
    print("\n[4] Sampling...")
    rng = np.random.default_rng(SEED)
    sampled_parts = []

    for label, df in strata.items():
        part = _sample_stratum(df, label, rng)
        part["stratum"] = label
        sampled_parts.append(part)
        print(f"  Stratum {label}: sampled {len(part)} rows")

    sample = pd.concat(sampled_parts, ignore_index=True)

    # ── Shuffle and assign coding IDs ─────────────────────────────────
    print("\n[5] Shuffling and assigning coding IDs...")
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample["coding_id"] = [f"V{i+1:04d}" for i in range(len(sample))]

    # Verify shuffle: coding_ids should not be stratum-clustered
    first_10_strata = sample["stratum"].head(10).tolist()
    print(f"  First 10 strata (should be mixed): {first_10_strata}")
    unique_in_first_10 = len(set(first_10_strata))
    print(f"  Unique strata in first 10: {unique_in_first_10} (expect >1 for good shuffle)")

    # ── Verification prints ───────────────────────────────────────────
    print("\n[6] Verification:")
    print(f"\n  Total sampled: {len(sample)}")

    print("\n  Count per stratum:")
    print(sample.groupby("stratum").size().rename("n").to_string())

    print("\n  Count per stratum × class:")
    cross = (sample.groupby(["stratum", "predicted_stance"])
             .size().rename("n").reset_index()
             .sort_values(["stratum", "predicted_stance"]))
    print(cross.to_string(index=False))

    print("\n  5 random examples:")
    examples = sample.sample(5, random_state=SEED)[
        ["coding_id", "stratum", "predicted_stance", "corpus_type", "pub_year"]
    ]
    print(examples.to_string(index=False))

    # ── Save internal file ────────────────────────────────────────────
    internal_cols = [
        "coding_id", "text", "original_id", "stratum",
        "predicted_stance", "pub_year", "corpus_type",
        "stance_confidence",
    ]
    # stance_confidence may not exist for academic rows (different column name)
    if "stance_confidence" not in sample.columns:
        sample["stance_confidence"] = np.nan

    internal = sample[[c for c in internal_cols if c in sample.columns]].copy()
    internal.to_csv(VAL_DIR / "validation_sample.csv", index=False)
    print(f"\n  Saved: results/validation/validation_sample.csv ({len(internal):,} rows)")

    # ── Save coder-facing file ────────────────────────────────────────
    coder = sample[["coding_id", "text"]].copy()
    coder["coder_label"] = ""
    coder["coder_notes"] = ""
    coder.to_csv(VAL_DIR / "validation_sample_for_coders.csv", index=False)
    print(f"  Saved: results/validation/validation_sample_for_coders.csv ({len(coder):,} rows)")

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
