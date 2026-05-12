"""
Policy Stance Temporal Analysis — Pre vs Post-ChatGPT (2022 boundary)
Extracts text from the 35 policy PDFs, chunks them (500 words / 50-word overlap),
assigns each chunk to pre- or post-ChatGPT based on the parent document's publication year,
then runs facebook/bart-large-mnli with the same labels and 0.45 threshold used on the
academic corpus (4d_sentiment_analysis.py).
"""
import re
import warnings
import pandas as pd
import pdfplumber
from pathlib import Path
from transformers import pipeline
import torch

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent.parent
POLICY_DIR  = BASE_DIR / "data_raw" / "policy_frameworks"
RESULTS_DIR = BASE_DIR / "results"

# --- exact same config as 4d_sentiment_analysis.py ---
ZS_MODEL   = "facebook/bart-large-mnli"
STANCE_LABELS = [
    "risk-focused: this text emphasises dangers, threats, harms, and concerns about AI",
    "opportunity-focused: this text emphasises benefits, potential, innovation, and positive outcomes of AI",
    "balanced: this text discusses both risks and opportunities of AI without clear emphasis",
]
LABEL_MAP = {
    "risk-focused: this text emphasises dangers, threats, harms, and concerns about AI":        "risk_focused",
    "opportunity-focused: this text emphasises benefits, potential, innovation, and positive outcomes of AI": "opportunity_focused",
    "balanced: this text discusses both risks and opportunities of AI without clear emphasis":   "balanced",
}
CONFIDENCE_THRESHOLD = 0.45
BATCH_SIZE           = 8

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


# ── Text extraction ────────────────────────────────────────────────────────────
def extract_text(pdf_path: Path) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
        return "\n\n".join(pages)
    except Exception:
        return ""


def clean_text(text: str) -> str:
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, doc_id: str) -> list[dict]:
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [{"chunk_id": f"{doc_id}_chunk_0", "text": text}]
    chunks = []
    start, num = 0, 0
    while start < len(words):
        end        = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        if len(chunk_text.split()) >= 50:
            chunks.append({"chunk_id": f"{doc_id}_chunk_{num}", "text": chunk_text})
        num   += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Build chunk dataframe from PDFs + manifest ────────────────────────────────
def build_chunks() -> pd.DataFrame:
    manifest = pd.read_csv(POLICY_DIR / "policy_manifest.csv", dtype=str)
    manifest = manifest[manifest["status"].isin(["downloaded", "already_exists"])].copy()
    manifest["year"] = pd.to_numeric(manifest["year"], errors="coerce")

    rows = []
    for _, doc in manifest.iterrows():
        pdf_path = POLICY_DIR / str(doc["filename"]).strip()
        if not pdf_path.exists():
            print(f"  MISSING: {doc['filename']}")
            continue
        raw  = extract_text(pdf_path)
        text = clean_text(raw)
        if len(text) < 100:
            print(f"  LOW TEXT: {doc['filename']} ({len(text)} chars)")
            continue
        doc_id = str(doc["filename"]).replace(".pdf", "")
        for ch in chunk_text(text, doc_id):
            ch.update({
                "doc_id":  doc_id,
                "country": doc.get("country", ""),
                "region":  doc.get("region", ""),
                "year":    doc["year"],
            })
            rows.append(ch)

    df = pd.DataFrame(rows)
    df["period"] = df["year"].apply(
        lambda y: "pre_chatgpt"  if pd.notna(y) and int(y) <= 2021 else
                  "post_chatgpt" if pd.notna(y) and int(y) >= 2022 else
                  "unknown"
    )
    print(f"\nTotal chunks: {len(df)}")
    print(df["period"].value_counts().to_string())
    return df.reset_index(drop=True)


# ── Classifier ────────────────────────────────────────────────────────────────
def load_classifier():
    if torch.backends.mps.is_available():
        device = 0
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = 0
        print("Using CUDA")
    else:
        device = -1
        print("Using CPU")
    return pipeline("zero-shot-classification", model=ZS_MODEL, device=device)


def classify_batch(classifier, texts: list[str]) -> list[dict]:
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        out   = classifier(batch, candidate_labels=STANCE_LABELS, multi_label=False)
        if isinstance(out, dict):
            out = [out]
        for r in out:
            top_label = r["labels"][0]
            top_score = r["scores"][0]
            scores    = dict(zip(r["labels"], r["scores"]))
            if top_score < CONFIDENCE_THRESHOLD:
                stance = "balanced"
            else:
                stance = LABEL_MAP[top_label]
            results.append({
                "stance":             stance,
                "stance_confidence":  round(top_score, 4),
                "score_risk":         round(scores.get(STANCE_LABELS[0], 0), 4),
                "score_opportunity":  round(scores.get(STANCE_LABELS[1], 0), 4),
                "score_balanced":     round(scores.get(STANCE_LABELS[2], 0), 4),
            })
        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"  Classified {min(i + BATCH_SIZE, len(texts)):,} / {len(texts):,} chunks")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("POLICY STANCE TEMPORAL ANALYSIS")
    print("=" * 60)

    print("\n[1/3] Extracting and chunking policy documents...")
    chunks = build_chunks()

    print(f"\n[2/3] Loading classifier: {ZS_MODEL}")
    classifier = load_classifier()

    print(f"\nRunning zero-shot classification on {len(chunks):,} chunks...")
    stance_rows = classify_batch(classifier, chunks["text"].tolist())
    stance_df   = pd.DataFrame(stance_rows)
    chunks      = pd.concat([chunks.reset_index(drop=True), stance_df], axis=1)

    # Save full results
    out_path = RESULTS_DIR / "policy_stance_temporal.csv"
    chunks.drop(columns=["text"]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[3/3] Results")
    print("=" * 60)

    periods = {"pre_chatgpt": "Pre-ChatGPT (≤2021)", "post_chatgpt": "Post-ChatGPT (≥2022)"}
    summary_rows = []

    for period_key, period_label in periods.items():
        subset = chunks[chunks["period"] == period_key]
        total  = len(subset)
        if total == 0:
            continue
        counts = subset["stance"].value_counts()
        risk   = counts.get("risk_focused", 0)
        opp    = counts.get("opportunity_focused", 0)
        bal    = counts.get("balanced", 0)

        print(f"\n{period_label}  (n={total:,} chunks)")
        print(f"  Risk-focused:        {risk:4d}  ({risk/total*100:.1f}%)")
        print(f"  Opportunity-focused: {opp:4d}  ({opp/total*100:.1f}%)")
        print(f"  Balanced:            {bal:4d}  ({bal/total*100:.1f}%)")

        docs = subset["doc_id"].nunique()
        print(f"  Documents covered:   {docs}")
        print(f"  Avg confidence:      {subset['stance_confidence'].mean():.3f}")
        low_conf = (subset["stance_confidence"] < CONFIDENCE_THRESHOLD).sum()
        print(f"  Low-conf (→balanced):{low_conf}  ({low_conf/total*100:.1f}%)")

        summary_rows.append({
            "period":              period_label,
            "n_chunks":            total,
            "n_docs":              docs,
            "risk_focused_n":      risk,
            "opportunity_focused_n": opp,
            "balanced_n":          bal,
            "risk_focused_pct":    round(risk/total*100, 1),
            "opportunity_focused_pct": round(opp/total*100, 1),
            "balanced_pct":        round(bal/total*100, 1),
            "avg_confidence":      round(subset["stance_confidence"].mean(), 3),
            "low_conf_pct":        round(low_conf/total*100, 1),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "policy_stance_temporal_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    # Pre vs post shift
    if len(summary_rows) == 2:
        pre  = summary_rows[0]
        post = summary_rows[1]
        print("\n── Shift (post minus pre) ──────────────────────────────")
        print(f"  Risk shift:        {post['risk_focused_pct'] - pre['risk_focused_pct']:+.1f} pp")
        print(f"  Opportunity shift: {post['opportunity_focused_pct'] - pre['opportunity_focused_pct']:+.1f} pp")
        print(f"  Balanced shift:    {post['balanced_pct'] - pre['balanced_pct']:+.1f} pp")

    print("\nDone.")


if __name__ == "__main__":
    main()
