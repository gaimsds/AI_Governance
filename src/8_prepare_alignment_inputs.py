"""
Prepares the four input files required by the symmetric alignment pipeline:

  data/policy_chunks.csv            — 1,557 rows: chunk_id, doc_id, chunk_text,
                                       country, region, doc_type, pub_year
  data/academic_governance_papers.csv — 3,186 rows: paper_id, abstract, topic_id,
                                       topic_label, pub_year, region
  data/governance_topic_ids.csv     — 21 governance topic IDs and labels
  models/academic_topic_model/      — copy of the saved BERTopic model

Policy chunk text is re-extracted from PDFs (gitignored from original run).
Academic data is re-shaped from results/governance_papers_stance.csv.
"""
import re
import shutil
import pdfplumber
import pandas as pd
from pathlib import Path

SEED       = 42
BASE_DIR   = Path(__file__).parent.parent
DATA_OUT   = BASE_DIR / "data"
RESULTS    = BASE_DIR / "results"
MODELS     = BASE_DIR / "models"
POLICY_DIR = BASE_DIR / "data_raw" / "policy_frameworks"

DATA_OUT.mkdir(exist_ok=True)

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


# ── 1. Policy chunks ──────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: Path) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
        text = "\n\n".join(pages)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    except Exception:
        return ""


def chunk_text(text: str, doc_id: str) -> list[dict]:
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [{"chunk_id": f"{doc_id}_chunk_0", "chunk_text": text}]
    chunks, start, num = [], 0, 0
    while start < len(words):
        end  = min(start + CHUNK_SIZE, len(words))
        blob = " ".join(words[start:end])
        if len(blob.split()) >= 50:
            chunks.append({"chunk_id": f"{doc_id}_chunk_{num}", "chunk_text": blob})
        num   += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_policy_chunks() -> pd.DataFrame:
    manifest  = pd.read_csv(POLICY_DIR / "policy_manifest.csv", dtype=str)
    manifest  = manifest[manifest["status"].isin(["downloaded", "already_exists"])].copy()
    manifest["pub_year"] = pd.to_numeric(manifest["year"], errors="coerce").astype("Int64")

    rows = []
    missing, low_text = 0, 0
    for _, doc in manifest.iterrows():
        pdf_path = POLICY_DIR / str(doc["filename"]).strip()
        if not pdf_path.exists():
            print(f"  MISSING PDF: {doc['filename']}")
            missing += 1
            continue
        text = extract_pdf_text(pdf_path)
        if len(text) < 100:
            print(f"  LOW TEXT:   {doc['filename']} ({len(text)} chars)")
            low_text += 1
            continue
        doc_id = str(doc["filename"]).replace(".pdf", "")
        for ch in chunk_text(text, doc_id):
            rows.append({
                "chunk_id": ch["chunk_id"],
                "doc_id":   doc_id,
                "chunk_text": ch["chunk_text"],
                "country":  doc.get("country", ""),
                "region":   doc.get("region", ""),
                "doc_type": doc.get("doc_type", ""),
                "pub_year": doc["pub_year"],
            })

    df = pd.DataFrame(rows)
    print(f"  Policy chunks built: {len(df):,} rows "
          f"({missing} PDFs missing, {low_text} low-text)")
    return df


# ── 2. Academic governance papers ─────────────────────────────────────────────
def build_academic_papers() -> pd.DataFrame:
    src = pd.read_csv(RESULTS / "governance_papers_stance.csv", dtype=str)

    # Extract year from cover_date (YYYY-MM-DD)
    src["pub_year"] = pd.to_numeric(
        src["cover_date"].str[:4], errors="coerce"
    ).astype("Int64")

    df = pd.DataFrame({
        "paper_id":    src["scopus_id"],
        "abstract":    src["abstract"],
        "topic_id":    pd.to_numeric(src["topic_id_finetuned"], errors="coerce").astype("Int64"),
        "topic_label": src["topic_label_finetuned"],
        "pub_year":    src["pub_year"],
        "region":      src["region"],
    })
    print(f"  Academic papers built: {len(df):,} rows")
    return df


# ── 3. Governance topic IDs ───────────────────────────────────────────────────
def build_governance_topic_ids() -> pd.DataFrame:
    src = pd.read_csv(RESULTS / "governance_topics.csv")
    df  = src[["topic_id", "topic_label", "governance_score", "paper_count"]].copy()
    print(f"  Governance topic IDs built: {len(df)} topics")
    return df


# ── 4. Copy academic model ────────────────────────────────────────────────────
def copy_academic_model():
    src = MODELS / "final_bertopic_model"
    dst = MODELS / "academic_topic_model"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  Academic model copied: {src.name} → {dst.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BUILDING ALIGNMENT INPUT FILES")
    print("=" * 60)

    print("\n[1/4] Building policy_chunks.csv...")
    policy_chunks = build_policy_chunks()
    policy_chunks.to_csv(DATA_OUT / "policy_chunks.csv", index=False)

    print("\n[2/4] Building academic_governance_papers.csv...")
    academic_papers = build_academic_papers()
    academic_papers.to_csv(DATA_OUT / "academic_governance_papers.csv", index=False)

    print("\n[3/4] Building governance_topic_ids.csv...")
    topic_ids = build_governance_topic_ids()
    topic_ids.to_csv(DATA_OUT / "governance_topic_ids.csv", index=False)

    print("\n[4/4] Copying academic_topic_model...")
    copy_academic_model()

    # ── Verification ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    checks = [
        ("data/policy_chunks.csv",              1557, ["chunk_id","doc_id","chunk_text","country","region","doc_type","pub_year"]),
        ("data/academic_governance_papers.csv", 3186, ["paper_id","abstract","topic_id","topic_label","pub_year","region"]),
        ("data/governance_topic_ids.csv",         21, ["topic_id","topic_label","governance_score","paper_count"]),
    ]

    all_ok = True
    for rel_path, expected_rows, expected_cols in checks:
        path = BASE_DIR / rel_path
        df   = pd.read_csv(path)
        row_ok = len(df) >= expected_rows * 0.85  # allow variance from direct PDF extraction
        col_ok = all(c in df.columns for c in expected_cols)
        status = "OK" if (row_ok and col_ok) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  [{status}] {rel_path}: {len(df):,} rows, cols={df.columns.tolist()}")
        if not col_ok:
            missing = [c for c in expected_cols if c not in df.columns]
            print(f"       Missing columns: {missing}")

    model_ok = (BASE_DIR / "models/academic_topic_model").exists()
    print(f"  [{'OK' if model_ok else 'FAIL'}] models/academic_topic_model/ exists")
    if not model_ok:
        all_ok = False

    print()
    print("All checks passed." if all_ok else "SOME CHECKS FAILED — review above.")


if __name__ == "__main__":
    main()
