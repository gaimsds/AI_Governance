"""
========================================================================================================================
POLICY CORPUS TEXT EXTRACTION
Project: Uneven Science–Policy Translation Shapes Global AI Governance
Authors: Tambudzai G. Charumbira & Joshua Gray
Institution: George Washington University, CCAS | M.S. Data Science | Spring 2026
Date: March 2026

DESCRIPTION:
This script extracted and cleaned text from the 35 policy framework PDFs collected in
1_data_collection_policyframeworks.py. The output was a structured CSV mirroring the format of the academic
corpus to enable consistent BERTopic processing across both corpora.

Extraction used pdfplumber as the primary method, with pypdf as fallback for documents where pdfplumber yielded
insufficient text. Language was detected automatically (langdetect) with manual overrides for known non-English
documents. English documents received full NLP cleaning (stopwords, lemmatization); non-English documents
(French, Spanish, Portuguese) received light cleaning only to preserve accented characters for multilingual
processing.

Extraction quality was assessed per document using a words-per-page metric: ≥500 wpp = good, 100–500 = moderate,
<100 = poor (likely scanned). The EU AI Act was the largest document at 91,316 words.

OUTPUT FILES (saved to data_clean/):
- policy_corpus.csv              → All 35 documents with raw and cleaned text
- policy_corpus_english_only.csv → English-language subset for primary BERTopic run
- policy_extraction_report.txt   → Per-document extraction quality report
========================================================================================================================
"""
import re
import logging
import pandas as pd
import pdfplumber
from pathlib import Path
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    from langdetect import detect as langdetect_detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# ======================================================================================================================
#  SECTION 2: FOLDER SETUP & LOGGING
# ======================================================================================================================
BASE_DIR    = Path(__file__).parent.parent
POLICY_DIR  = BASE_DIR / "data_raw"  / "policy_frameworks"
DATA_CLEAN  = BASE_DIR / "data_clean"
DATA_CLEAN.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(DATA_CLEAN / "policy_extraction.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ======================================================================================================================
#  SECTION 3: NLTK SETUP
#  NLTK resources were downloaded for tokenization, stopword removal, and lemmatization. Domain-specific stopwords common
#  in policy documents (e.g., "shall", "pursuant", "accordance", "annex") were added to the standard English stopword list
#  to reduce noise without removing substantive governance vocabulary.
# ======================================================================================================================
def setup_nltk():
    for resource in ["punkt", "stopwords", "wordnet", "omw-1.4",
                     "punkt_tab", "averaged_perceptron_tagger"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

setup_nltk()

ENGLISH_STOPWORDS = set(stopwords.words("english"))

DOMAIN_STOPWORDS  = {
    "shall", "may", "must", "article", "section", "paragraph",
    "member", "state", "states", "government", "national",
    "international", "including", "pursuant", "accordance",
    "relevant", "ensure", "within", "provide", "regard",
    "also", "however", "therefore", "thus", "whereas",
    "annex", "chapter", "part", "ibid", "et", "al",
}
ENGLISH_STOPWORDS.update(DOMAIN_STOPWORDS)
LEMMATIZER = WordNetLemmatizer()

# ======================================================================================================================
#  SECTION 4: KNOWN LANGUAGE MAP
#  Non-English documents were pre-assigned language codes to avoid misdetection on short or mixed-language content. Four
#  documents were flagged: France (French), Brazil (Portuguese), Colombia (Spanish), and Mexico (English version available).
# ======================================================================================================================
KNOWN_LANGUAGES = {
    "France_AI_Strategy_2021.pdf":                   "fr",
    "Brazil_National_AI_Plan_PBIA_2024.pdf":         "pt",
    "Colombia_National_AI_Policy_CONPES4144_2025.pdf": "es",
    "Mexico_AI_National_Agenda_2018.pdf":            "en",
}

NON_ENGLISH = {"fr", "es", "pt"}

# ======================================================================================================================
#  SECTION 5: TEXT EXTRACTION
#  A two-tier extraction strategy was implemented. pdfplumber was used first for its superior layout handling. If
#  pdfplumber yielded fewer than 200 characters, pypdf was attempted as fallback. The extraction method used was recorded
#  for each document to support quality auditing.
# ======================================================================================================================
def extract_text_pdfplumber(pdf_path: Path) -> tuple[str, int]:
    """Extract text using pdfplumber. Returns (text, page_count)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages), len(pdf.pages)
    except Exception as e:
        log.warning(f"   pdfplumber failed: {e}")
        return "", 0

def extract_text_pypdf(pdf_path: Path) -> tuple[str, int]:
    """Fallback extraction using pypdf. Returns (text, page_count)."""
    if not HAS_PYPDF:
        return "", 0
    try:
        reader = PdfReader(str(pdf_path))
        pages  = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages), len(reader.pages)
    except Exception as e:
        log.warning(f"   pypdf fallback failed: {e}")
        return "", 0

def extract_text(pdf_path: Path) -> tuple[str, int, str]:
    """
    Try pdfplumber first, fall back to pypdf.
    Returns (text, page_count, method_used).
    """
    text, pages = extract_text_pdfplumber(pdf_path)

    if len(text.strip()) > 200:
        return text, pages, "pdfplumber"

    log.info("   pdfplumber yielded little text — trying pypdf fallback...")
    text_fb, pages_fb = extract_text_pypdf(pdf_path)

    if len(text_fb.strip()) > len(text.strip()):
        return text_fb, pages_fb, "pypdf_fallback"

    return text, pages, "pdfplumber_limited"

# ======================================================================================================================
#  SECTION 6: TEXT CLEANING
#  Three cleaning levels were applied depending on document language:
#    - Basic (all documents): page number removal, URL stripping, whitespace normalization
#    - Full NLP (English): stopword removal and lemmatization, matching the academic corpus pipeline
#    - Light multilingual (non-English): lowercase and whitespace only, preserving accented characters
# ======================================================================================================================
def clean_raw_text(text: str) -> str:
    """Basic cleaning applied to ALL documents regardless of language."""
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n\u00C0-\u024F\u0400-\u04FF]", " ", text)
    return text.strip()

def clean_english_nlp(text: str) -> str:
    """
    Full NLP cleaning for English documents.
    Lowercase → tokenise → remove stopwords → lemmatize.
    Mirrors the cleaning in 2_data_cleaning_scopus.py.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in ENGLISH_STOPWORDS and len(t) > 2
    ]
    return " ".join(tokens)

def clean_multilingual_light(text: str) -> str:
    """
    Light cleaning for non-English documents.
    Lowercase + whitespace normalisation only.
    Preserves accented characters for multilingual BERTopic.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def detect_language(text: str, filename: str) -> str:
    """Detect document language. Uses known map first, then langdetect."""
    if filename in KNOWN_LANGUAGES:
        return KNOWN_LANGUAGES[filename]
    if not HAS_LANGDETECT:
        return "en"
    try:
        sample = text[:3000]
        return langdetect_detect(sample)
    except Exception:
        return "en"

def assess_quality(text: str, page_count: int) -> str:
    """
    Assign a quality flag to the extraction.
    Good: >500 words per page average.
    Moderate: 100-500 words per page.
    Poor: <100 words per page (likely scanned/image PDF).
    """
    word_count = len(text.split())
    if page_count == 0:
        return "failed"
    wpp = word_count / page_count
    if wpp >= 500:
        return "good"
    elif wpp >= 100:
        return "moderate"
    else:
        return "poor_likely_scanned"

# ======================================================================================================================
#  SECTION 7: MANIFEST LOADING
#  The policy manifest generated by 1_data_collection_policy frameworks.py was loaded to identify which documents were
#  successfully downloaded and ready for extraction. Documents with status "downloaded" or "already_exists" were
#  processed; others were skipped.
# ======================================================================================================================
def load_manifest() -> pd.DataFrame:
    manifest_path = POLICY_DIR / "policy_manifest.csv"
    if not manifest_path.exists():
        log.error(f" policy_manifest.csv not found at {manifest_path}")
        log.error("   Run 1_data_collection_policyframeworks.py first.")
        raise FileNotFoundError(manifest_path)
    df = pd.read_csv(manifest_path, dtype=str)
    log.info(f"   Loaded manifest: {len(df)} documents")
    return df

# ======================================================================================================================
#  SECTION 8: MAIN PIPELINE
#  The pipeline iterated through all downloaded policy documents and for each:
#    1. Extracted raw text via pdfplumber (with pypdf fallback)
#    2. Applied basic cleaning to all documents
#    3. Detected language (automatic + manual overrides)
#    4. Applied full NLP or light cleaning based on language
#    5. Assessed extraction quality via words-per-page metric
#    6. Saved the full corpus CSV, English-only subset, and extraction quality report
# ======================================================================================================================
def run_extraction_pipeline():

    log.info("=" * 65)
    log.info("  POLICY FRAMEWORKS TEXT EXTRACTION PIPELINE")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 65)

    manifest = load_manifest()
    downloaded = manifest[manifest["status"].isin(["downloaded", "already_exists"])]
    log.info(f"   Documents to extract: {len(downloaded)} / {len(manifest)} total")
    log.info(f"   Skipping:             {len(manifest) - len(downloaded)} (not downloaded)")

    results = []
    success = 0
    partial = 0
    failed  = 0

    for i, (_, row) in enumerate(downloaded.iterrows(), 1):
        filename = str(row.get("filename", "")).strip()
        filepath = POLICY_DIR / filename
        log.info("")
        log.info(f"[{i:02d}/{len(downloaded):02d}] {row.get('country')} — {row.get('doc_name')}")

        if not filepath.exists():
            log.warning(f"     File not found: {filename} — skipping")
            failed += 1
            continue

        if filepath.stat().st_size < 1000:
            log.warning(f"     File too small ({filepath.stat().st_size} bytes) — skipping")
            failed += 1
            continue

        # ── ExtractING raw text ──────────────────────────────────────────────────
        text_raw, page_count, method = extract_text(filepath)
        if not text_raw.strip():
            log.warning(f"   ❌ No text extracted — file may be image-only PDF")
            failed += 1
            results.append(_build_row(row, filename, "", "", 0, page_count,
                                       "unknown", "failed", method))
            continue

        # ── Basic cleaning (all languages) ────────────────────────────────────
        text_raw   = clean_raw_text(text_raw)
        word_count = len(text_raw.split())

        # ── Language detection ────────────────────────────────────────────────
        language   = detect_language(text_raw, filename)
        is_english = language not in NON_ENGLISH

        # ── NLP cleaning ──────────────────────────────────────────────────────
        if is_english:
            text_clean = clean_english_nlp(text_raw)
            log.info(f"    Extracted {word_count:,} words | {page_count} pages | "
                     f"lang={language} | {method}")
        else:
            text_clean = clean_multilingual_light(text_raw)
            log.info(f"    Extracted {word_count:,} words | {page_count} pages | "
                     f"lang={language} (non-English — light cleaning) | {method}")

        # ── Quality assessment ────────────────────────────────────────────────
        quality = assess_quality(text_raw, page_count)
        if quality == "poor_likely_scanned":
            log.warning(f"     Low word density — may be partially scanned")
            partial += 1
        else:
            success += 1

        results.append(_build_row(
            row, filename, text_raw, text_clean,
            word_count, page_count, language, quality, method
        ))

    # ── Build output dataframe ────────────────────────────────────────────────
    corpus_df = pd.DataFrame(results)

    # ── Save outputs ──────────────────────────────────────────────────────────
    corpus_path = DATA_CLEAN / "policy_corpus.csv"
    corpus_df.to_csv(corpus_path, index=False, encoding="utf-8")
    log.info(f"\n Corpus saved: {corpus_path.name}  ({len(corpus_df)} documents)")

    english_df = corpus_df[corpus_df["language"] == "en"]
    english_path = DATA_CLEAN / "policy_corpus_english_only.csv"
    english_df.to_csv(english_path, index=False, encoding="utf-8")
    log.info(f" English-only corpus saved: {english_path.name}  ({len(english_df)} documents)")
    _save_report(corpus_df, success, partial, failed)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info("  EXTRACTION SUMMARY")
    log.info("=" * 65)
    log.info(f"  Total documents processed:  {len(results)}")
    log.info(f"  Good extraction:            {success}")
    log.info(f"  Partial (low density):      {partial}")
    log.info(f"  Failed:                     {failed}")
    log.info("")
    log.info("  Language breakdown:")
    for lang, count in corpus_df["language"].value_counts().items():
        label = "English" if lang == "en" else \
                "French"  if lang == "fr" else \
                "Spanish" if lang == "es" else \
                "Portuguese" if lang == "pt" else lang
        log.info(f"    {label:<15} ({lang})  {count} documents")
    log.info("")
    log.info("  Region breakdown:")
    for region, count in corpus_df["region"].value_counts().items():
        total_words = corpus_df[corpus_df["region"] == region]["word_count"].sum()
        log.info(f"    {region:<25} {count} docs  {total_words:>8,} words")
    log.info("")
    log.info("  Top 5 documents by word count:")
    top5 = corpus_df.nlargest(5, "word_count")[["doc_name", "country", "word_count"]]
    for _, r in top5.iterrows():
        log.info(f"    {r['country']:<20} {int(r['word_count']):>8,} words  {r['doc_name'][:50]}")
    log.info("")
    log.info(f"  Output files in data_clean/:")
    log.info(f"    policy_corpus.csv              — all documents (all languages)")
    log.info(f"    policy_corpus_english_only.csv — English documents only")
    log.info(f"    policy_extraction_report.txt   — extraction quality report")
    log.info("=" * 65)
    log.info("🎉 Policy text extraction complete!")
    log.info("")

# ======================================================================================================================
#  SECTION 9: HELPER FUNCTIONS
#  Utility functions for building output rows and generating the extraction quality report.
# ======================================================================================================================
def _build_row(row, filename, text_raw, text_clean,
               word_count, page_count, language, quality, method):
    return {
        "doc_id":             filename.replace(".pdf", ""),
        "filename":           filename,
        "country":            row.get("country", ""),
        "region":             row.get("region", ""),
        "issuer":             row.get("issuer", ""),
        "doc_name":           row.get("doc_name", ""),
        "doc_type":           row.get("doc_type", ""),
        "year":               row.get("year", ""),
        "language":           language,
        "word_count":         word_count,
        "page_count":         page_count,
        "extraction_method":  method,
        "extraction_quality": quality,
        "text_raw":           text_raw,
        "text_clean":         text_clean,
        "notes":              row.get("notes", ""),
    }

def _save_report(df: pd.DataFrame, success: int, partial: int, failed: int):
    report_path = DATA_CLEAN / "policy_extraction_report.txt"
    with open(report_path, "w") as f:
        f.write("POLICY FRAMEWORKS — TEXT EXTRACTION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Total documents processed:  {len(df)}\n")
        f.write(f"Good extraction:            {success}\n")
        f.write(f"Partial (low density):      {partial}\n")
        f.write(f"Failed:                     {failed}\n\n")

        f.write("DOCUMENT DETAILS:\n")
        f.write("-" * 65 + "\n")
        for _, row in df.iterrows():
            f.write(f"\n{row['country']} — {row['doc_name']}\n")
            f.write(f"  Language:  {row['language']}\n")
            f.write(f"  Words:     {int(row['word_count']):,}\n")
            f.write(f"  Pages:     {row['page_count']}\n")
            f.write(f"  Quality:   {row['extraction_quality']}\n")
            f.write(f"  Method:    {row['extraction_method']}\n")

        f.write("\n\nMETHODOLOGY NOTE:\n")
        f.write("Non-English documents (French, Spanish, Portuguese) received\n")
        f.write("light cleaning only (lowercase + whitespace normalisation).\n")
        f.write("They are included in policy_corpus.csv for multilingual\n")
        f.write("BERTopic analysis but excluded from policy_corpus_english_only.csv.\n")
        f.write("See 3_modelling.py for both English-only and multilingual runs.\n")

    log.info(f" Report saved:  policy_extraction_report.txt")

# ======================================================================================================================
#  ENTRY POINT
# ======================================================================================================================

if __name__ == "__main__":
    run_extraction_pipeline()