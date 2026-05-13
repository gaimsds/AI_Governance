"""
Policy corpus cleaning pipeline.

Steps:
  1. Classify each document as English-source or non-English-source
  2. Drop non-English-source documents entirely (log them)
  3. Apply OCR-artifact heuristics per chunk on retained documents
  4. Tag every retained chunk with original_language and source_language_match
  5. Save cleaned corpus to data/policy_chunks_clean.csv
  6. Print cleaning summary report

Non-English sources dropped (per user instruction, regardless of detection):
  France (fr), Brazil (pt), Colombia (es), Mexico (es), Chile (machine-translated)

For all other documents: chunk-level OCR heuristics applied.
"""

import re
import unicodedata
import pandas as pd
from pathlib import Path
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
RESULTS   = BASE_DIR / "results" / "alignment"

# ── Documents explicitly classified by source language ────────────────────────
# original_language = language of the authoritative source document
# english_source = True if an official English version exists
DOC_LANGUAGE_MAP = {
    # Non-English source — DROP
    "France_AI_Strategy_2021":                    {"original_language": "fr", "english_source": False},
    "Brazil_National_AI_Plan_PBIA_2024":           {"original_language": "pt", "english_source": False},
    "Colombia_National_AI_Policy_CONPES4144_2025": {"original_language": "es", "english_source": False},
    "Mexico_AI_National_Agenda_2018":              {"original_language": "es", "english_source": False},
    "Chile_National_AI_Policy_2021":               {"original_language": "es", "english_source": False,
                                                    "note": "PDF is machine-translated from Spanish — not an official English version"},
    # English source — KEEP
    "China_New_Generation_AI_Development_Plan_2017": {"original_language": "zh", "english_source": True,
                                                       "note": "CSET Georgetown official English translation"},
    "China_Interim_Measures_Generative_AI_2023":     {"original_language": "zh", "english_source": True,
                                                       "note": "USAF CASI official English translation"},
    "SouthKorea_AI_Basic_Act_2025":                  {"original_language": "ko", "english_source": True,
                                                       "note": "CSET Georgetown official English translation"},
    "Japan_AI_Strategy_2022":                        {"original_language": "ja", "english_source": True,
                                                       "note": "Cabinet Office Japan official English version"},
}

# Reversed English words — used to detect mirror-text OCR artefacts
REVERSED_COMMON = {
    "dna", "eht", "rof", "fo", "ot", "ni", "ia", "si", "sa", "yb",
    "sti", "na", "ro", "ta", "ew", "era", "saw", "ton", "tub", "htiw",
    "morf", "lacol", "tnempoleved", "lanoitan", "ecnanrevog", "ytiruces",
    "ycilopecnanrevog", "ygolocnhcet", "tnemssessa", "noitartsinimda",
}


# ── OCR artifact heuristics ───────────────────────────────────────────────────
def non_alpha_ratio(text: str) -> float:
    """Proportion of characters that are not letters and not whitespace."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    non_alpha = sum(1 for c in chars if not c.isalpha())
    return non_alpha / len(chars)


def avg_word_length(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def numeric_token_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    numeric = sum(1 for t in tokens if re.fullmatch(r"[\d,.]+", t))
    return numeric / len(tokens)


def reversed_text_score(text: str) -> float:
    """Fraction of 2–6 char lowercase tokens that are reversed common English words."""
    tokens = [t.lower() for t in text.split() if 2 <= len(t) <= 6 and t.isalpha()]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in REVERSED_COMMON)
    return hits / len(tokens)


def utf8_sanity(text: str) -> bool:
    """Return False if text contains a high proportion of replacement characters."""
    replacements = text.count("�")
    return replacements / max(len(text), 1) < 0.02


def standalone_al_ratio(text: str) -> float:
    """Fraction of tokens that are exactly 'al' — detects I→l font encoding artifacts
    where 'AI' is rendered as 'Al' (e.g. ASEAN PDFs with ligature substitution)."""
    tokens = text.split()
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t.lower() == "al")
    return count / len(tokens)


def is_ocr_artifact(text: str) -> tuple[bool, str]:
    """
    Returns (is_artifact, reason). Checks five heuristics:
      - non_alpha_ratio > 0.20
      - avg_word_length < 2.5 or > 12
      - numeric_token_ratio > 0.25
      - reversed_text_score > 0.15
      - utf8 sanity fail
      - standalone_al_ratio > 0.04  (I→l font encoding artifact)
    """
    nar = non_alpha_ratio(text)
    if nar > 0.20:
        return True, f"non_alpha_ratio={nar:.2f}"

    awl = avg_word_length(text)
    if awl < 2.5:
        return True, f"avg_word_len={awl:.2f} (<2.5)"
    if awl > 12:
        return True, f"avg_word_len={awl:.2f} (>12)"

    ntr = numeric_token_ratio(text)
    if ntr > 0.25:
        return True, f"numeric_token_ratio={ntr:.2f}"

    rts = reversed_text_score(text)
    if rts > 0.15:
        return True, f"reversed_text_score={rts:.2f}"

    if not utf8_sanity(text):
        return True, "utf8_replacement_chars"

    sal = standalone_al_ratio(text)
    if sal > 0.04:
        return True, f"al_as_ai_ratio={sal:.2f}"

    return False, ""


def detect_language_safe(text: str) -> str:
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "unknown"


# ── Main cleaning pipeline ────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("POLICY CORPUS CLEANING PIPELINE")
    print("=" * 65)

    chunks = pd.read_csv(DATA_DIR / "policy_chunks.csv")
    total_input = len(chunks)
    print(f"\nInput: {total_input:,} chunks from {chunks['doc_id'].nunique()} documents")

    # ── Step 1: Classify each document ────────────────────────────────────────
    print("\n[Step 1] Classifying documents by source language...")

    drop_docs   = []
    keep_docs   = []
    doc_meta    = {}

    for doc_id, grp in chunks.groupby("doc_id"):
        meta = DOC_LANGUAGE_MAP.get(doc_id, {})

        if meta.get("english_source") is False:
            # Explicitly non-English or machine-translated: drop
            drop_docs.append({
                "doc_id":            doc_id,
                "country":           grp["country"].iloc[0],
                "original_language": meta.get("original_language", "?"),
                "reason":            meta.get("note", "non-English source"),
                "n_chunks":          len(grp),
            })
            doc_meta[doc_id] = {
                "original_language":   meta.get("original_language", "?"),
                "english_source":      False,
                "source_language_match": False,
            }
        else:
            # English source (official or detected)
            orig_lang = meta.get("original_language", "en")
            keep_docs.append(doc_id)
            doc_meta[doc_id] = {
                "original_language":   orig_lang,
                "english_source":      True,
                "source_language_match": orig_lang == "en",
            }

    print(f"  Documents to DROP: {len(drop_docs)}")
    for d in drop_docs:
        print(f"    {d['doc_id']:<55}  lang={d['original_language']}  "
              f"({d['n_chunks']} chunks)  [{d['reason']}]")
    print(f"  Documents to KEEP: {len(keep_docs)}")

    # Save dropped document log
    drop_log = pd.DataFrame(drop_docs)
    drop_log.to_csv(RESULTS / "dropped_documents.csv", index=False)
    print(f"  Saved: results/alignment/dropped_documents.csv")

    # ── Step 2: Filter to kept documents ──────────────────────────────────────
    retained = chunks[chunks["doc_id"].isin(keep_docs)].copy()
    print(f"\n[Step 2] After document-level drop: {len(retained):,} chunks")

    # ── Step 3: OCR artifact heuristics per chunk ──────────────────────────────
    print("\n[Step 3] Applying OCR artifact heuristics...")

    artifact_flags = []
    artifact_reasons = []
    chunk_langs = []

    for _, row in retained.iterrows():
        text = str(row["chunk_text"])
        is_art, reason = is_ocr_artifact(text)

        # Secondary check: language detection on retained-doc chunks
        if not is_art:
            lang = detect_language_safe(text)
            chunk_langs.append(lang)
            if lang not in ("en", "unknown") and len(text.split()) > 30:
                is_art  = True
                reason  = f"chunk_lang_detected={lang}"
        else:
            chunk_langs.append("not_checked")

        artifact_flags.append(is_art)
        artifact_reasons.append(reason)

    retained = retained.copy()
    retained["_is_artifact"] = artifact_flags
    retained["_artifact_reason"] = artifact_reasons
    retained["_chunk_lang"] = chunk_langs

    n_artifacts = sum(artifact_flags)
    print(f"  OCR artifact chunks flagged: {n_artifacts}")

    if n_artifacts > 0:
        art_summary = (retained[retained["_is_artifact"]]
                       .groupby(["doc_id", "_artifact_reason"])
                       .size().reset_index(name="n")
                       .sort_values("n", ascending=False))
        print(art_summary.to_string(index=False))

    # Save artifact log
    artifact_log = retained[retained["_is_artifact"]][
        ["chunk_id", "doc_id", "country", "_artifact_reason"]
    ].rename(columns={"_artifact_reason": "reason"})
    artifact_log.to_csv(RESULTS / "dropped_chunks_ocr.csv", index=False)
    print(f"  Saved: results/alignment/dropped_chunks_ocr.csv")

    # ── Step 4: Build final cleaned corpus ────────────────────────────────────
    clean = retained[~retained["_is_artifact"]].copy()

    # Add traceability tags
    clean["original_language"]    = clean["doc_id"].map(
        lambda d: doc_meta[d]["original_language"])
    clean["source_language_match"] = clean["doc_id"].map(
        lambda d: doc_meta[d]["source_language_match"])

    clean = clean.drop(columns=["_is_artifact", "_artifact_reason", "_chunk_lang"])
    clean = clean.reset_index(drop=True)

    clean.to_csv(DATA_DIR / "policy_chunks_clean.csv", index=False)
    print(f"\n[Step 4] Cleaned corpus saved: data/policy_chunks_clean.csv")

    # ── Cleaning summary report ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  CLEANING SUMMARY")
    print(f"{'='*65}")
    print(f"  Input chunks:                  {total_input:>6,}")
    print(f"  Dropped (non-English docs):    {total_input - len(retained):>6,}  "
          f"({(total_input - len(retained))/total_input*100:.1f}%)")
    print(f"  Dropped (OCR artifacts):       {n_artifacts:>6,}  "
          f"({n_artifacts/len(retained)*100:.1f}% of retained)")
    print(f"  Final clean chunks:            {len(clean):>6,}")
    print(f"  Documents retained:            {clean['doc_id'].nunique():>6}")
    print(f"  Documents dropped:             {len(drop_docs):>6}")

    print(f"\n  Chunks per region (clean corpus):")
    print(clean.groupby("region").size().reset_index(name="chunk_count").to_string(index=False))

    print(f"\n  Chunks per document (clean corpus, sorted):")
    doc_counts = (clean.groupby(["doc_id", "country", "pub_year", "original_language"])
                  .size().reset_index(name="chunk_count")
                  .sort_values("chunk_count", ascending=False))
    print(doc_counts.to_string(index=False))

    return clean


if __name__ == "__main__":
    main()
