# =============================================================================
#  DATA CLEANING SCRIPT
#  Project: Mapping Global AI Governance Narratives
#  Authors: Tambudzai Gundani & Joshua Gray
#  Date:    March 2026
# =============================================================================
#
#  WHAT THIS SCRIPT DOES:
#  ----------------------
#  Takes the raw Scopus corpus (41,893 papers) and produces a clean,
#  analysis-ready dataset by applying QUALITY-ONLY filters.
#
#  IMPORTANT METHODOLOGICAL NOTE:
#  --------------------------------
#  This script applies NO thematic filtering. We do not filter by governance
#  keywords, topic terms, or any content-based criteria. The purpose of using
#  BERTopic (unsupervised topic modelling) is to let the algorithm discover
#  what AI discourse is actually about — not to pre-impose themes we expect.
#  Only papers that are technically unusable for NLP are removed.
#
#  FILTERS APPLIED (quality only):
#  ---------------------------------
#  1. Exact duplicates (by DOI, then by title)
#  2. Empty or missing abstracts
#  3. Abstracts under 50 words (too short for meaningful NLP)
#  4. Non-English papers that slipped through language filter
#  5. Standardise country names (e.g. "USA" → "United States")
#  6. Standardise year column to clean integers
#  7. Add permanent region column
#
#  TEXT PREPROCESSING (for NLP pipeline):
#  ----------------------------------------
#  Full preprocessing on abstract text:
#  - Lowercase
#  - Remove special characters and encoding artifacts
#  - Remove stopwords
#  - Lemmatize
#  Stored in a NEW column 'abstract_clean' — original abstract preserved
#
#  OUTPUT FILES (saved to data_clean/ folder):
#  --------------------------------------------
#  - scopus_cleaned.csv            — full cleaned dataset, all columns
#  - scopus_cleaned.xlsx           — same as Excel for browsing
#  - scopus_abstracts_nlp.csv      — abstracts only, ready for BERTopic
#  - scopus_cleaning_report.txt    — summary of what was removed and why
# =============================================================================


# -----------------------------------------------------------------------------
#  SECTION 1: IMPORTS
# -----------------------------------------------------------------------------

import os
import re
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

# NLP libraries for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only downloads if not already present)
# These are small files downloaded once and cached locally
nltk.download('stopwords',       quiet=True)
nltk.download('wordnet',         quiet=True)
nltk.download('punkt',           quiet=True)
nltk.download('punkt_tab',       quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4',         quiet=True)


# -----------------------------------------------------------------------------
#  SECTION 2: FOLDER SETUP
# -----------------------------------------------------------------------------

DATA_RAW   = Path(__file__).parent.parent / "data_raw"
DATA_CLEAN = Path(__file__).parent.parent / "data_clean"
DATA_CLEAN.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
#  SECTION 3: LOGGING SETUP
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(DATA_CLEAN / "scopus_cleaning.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#  SECTION 4: COUNTRY NAME STANDARDISATION MAP
#  Scopus returns country names inconsistently across papers.
#  This map normalises the most common variations to a single standard form.
# -----------------------------------------------------------------------------

COUNTRY_CORRECTIONS = {
    # United States variations
    "USA": "United States", "U.S.A": "United States", "U.S.": "United States",
    "US": "United States", "United States of America": "United States",

    # United Kingdom variations
    "UK": "United Kingdom", "England": "United Kingdom",
    "Scotland": "United Kingdom", "Wales": "United Kingdom",
    "Great Britain": "United Kingdom",

    # China variations
    "Peoples R China": "China", "People's Republic of China": "China",
    "P.R. China": "China", "PR China": "China",

    # South Korea variations
    "Korea": "South Korea", "Republic of Korea": "South Korea",
    "South Korea": "South Korea",

    # Other common variations
    "Taiwan (Republic of China)": "Taiwan",
    "Hong Kong SAR": "Hong Kong",
    "Macao": "Macau",
    "Czech Republic": "Czechia",
    "Russian Federation": "Russia",
    "Iran (Islamic Republic of)": "Iran",
    "Viet Nam": "Vietnam",
    "Syrian Arab Republic": "Syria",
    "United Arab Emirates": "UAE",
}


# -----------------------------------------------------------------------------
#  SECTION 5: REGION MAP
#  Maps each country to one of 5 regions from your proposal's stratification.
#  This becomes a permanent column in the cleaned dataset.
# -----------------------------------------------------------------------------

REGION_MAP = {
    # North America
    "United States": "North America", "Canada": "North America",

    # Europe
    "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
    "Netherlands": "Europe", "Italy": "Europe", "Spain": "Europe",
    "Sweden": "Europe", "Switzerland": "Europe", "Norway": "Europe",
    "Denmark": "Europe", "Finland": "Europe", "Belgium": "Europe",
    "Austria": "Europe", "Poland": "Europe", "Portugal": "Europe",
    "Ireland": "Europe", "Greece": "Europe", "Czechia": "Europe",
    "Czech Republic": "Europe", "Hungary": "Europe", "Romania": "Europe",
    "Russia": "Europe", "Turkey": "Europe", "Ukraine": "Europe",
    "Croatia": "Europe", "Serbia": "Europe", "Slovenia": "Europe",
    "Slovakia": "Europe", "Luxembourg": "Europe", "Estonia": "Europe",
    "Latvia": "Europe", "Lithuania": "Europe", "Malta": "Europe",
    "Cyprus": "Europe", "Iceland": "Europe",

    # Asia-Pacific
    "China": "Asia-Pacific", "India": "Asia-Pacific",
    "Japan": "Asia-Pacific", "South Korea": "Asia-Pacific",
    "Australia": "Asia-Pacific", "Singapore": "Asia-Pacific",
    "New Zealand": "Asia-Pacific", "Taiwan": "Asia-Pacific",
    "Hong Kong": "Asia-Pacific", "Malaysia": "Asia-Pacific",
    "Indonesia": "Asia-Pacific", "Thailand": "Asia-Pacific",
    "Vietnam": "Asia-Pacific", "Philippines": "Asia-Pacific",
    "Pakistan": "Asia-Pacific", "Bangladesh": "Asia-Pacific",
    "Sri Lanka": "Asia-Pacific", "Nepal": "Asia-Pacific",
    "Myanmar": "Asia-Pacific", "Cambodia": "Asia-Pacific",
    "Macau": "Asia-Pacific",

    # Latin America
    "Brazil": "Latin America", "Mexico": "Latin America",
    "Argentina": "Latin America", "Colombia": "Latin America",
    "Chile": "Latin America", "Peru": "Latin America",
    "Venezuela": "Latin America", "Ecuador": "Latin America",
    "Bolivia": "Latin America", "Uruguay": "Latin America",
    "Costa Rica": "Latin America", "Cuba": "Latin America",
    "Panama": "Latin America", "Paraguay": "Latin America",
    "Guatemala": "Latin America", "Honduras": "Latin America",

    # Africa & Middle East
    "South Africa": "Africa & Middle East",
    "Nigeria": "Africa & Middle East",
    "Kenya": "Africa & Middle East",
    "Egypt": "Africa & Middle East",
    "Ethiopia": "Africa & Middle East",
    "Ghana": "Africa & Middle East",
    "Tanzania": "Africa & Middle East",
    "Uganda": "Africa & Middle East",
    "Rwanda": "Africa & Middle East",
    "Cameroon": "Africa & Middle East",
    "Tunisia": "Africa & Middle East",
    "Morocco": "Africa & Middle East",
    "Algeria": "Africa & Middle East",
    "Israel": "Africa & Middle East",
    "Saudi Arabia": "Africa & Middle East",
    "UAE": "Africa & Middle East",
    "Qatar": "Africa & Middle East",
    "Jordan": "Africa & Middle East",
    "Lebanon": "Africa & Middle East",
    "Iran": "Africa & Middle East",
    "Iraq": "Africa & Middle East",
    "Kuwait": "Africa & Middle East",
    "Oman": "Africa & Middle East",
    "Bahrain": "Africa & Middle East",
    "Zimbabwe": "Africa & Middle East",
    "Zambia": "Africa & Middle East",
    "Senegal": "Africa & Middle East",
    "Ivory Coast": "Africa & Middle East",
}


# -----------------------------------------------------------------------------
#  SECTION 6: TEXT PREPROCESSING FUNCTION
#  Cleans abstract text for BERTopic input.
#  Stores result in 'abstract_clean' — NEVER overwrites original 'abstract'.
#
#  Steps:
#  1. Fix encoding artifacts (e.g. â€™ → ', √ú → ü)
#  2. Lowercase everything
#  3. Remove URLs, email addresses, numbers, special characters
#  4. Tokenize (split into individual words)
#  5. Remove stopwords (common words with no meaning: "the", "and", "is")
#  6. Lemmatize (reduce words to root: "governing" → "govern")
# -----------------------------------------------------------------------------

# Initialise lemmatizer and stopwords once — reused for every paper
lemmatizer = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words('english'))

# Add domain-specific stopwords that add noise without meaning for NLP
# These are very common in academic abstracts but carry no thematic signal
CUSTOM_STOPWORDS = {
    "paper", "study", "research", "article", "work", "propose", "proposed",
    "present", "presented", "show", "shown", "result", "results", "finding",
    "findings", "also", "using", "used", "use", "based", "approach",
    "method", "methods", "methodology", "analysis", "analyze", "analyses",
    "data", "dataset", "model", "models", "framework", "review", "however",
    "therefore", "thus", "furthermore", "moreover", "although", "despite",
    "conclusion", "conclusions", "suggest", "suggests", "indicate", "indicates",
    "provide", "provides", "examine", "examines", "investigate", "investigates",
    "al", "et", "fig", "table", "section", "journal", "volume", "issue",
}
STOP_WORDS.update(CUSTOM_STOPWORDS)


def clean_encoding(text: str) -> str:
    """Fix common encoding artifacts from PDF/database extraction."""
    # Common Scopus encoding issues
    replacements = {
        "â€™": "'", "â€œ": '"', "â€": '"', "â€˜": "'",
        "√ú": "ü", "√∂": "ö", "√§": "ä", "√±": "ñ",
        "√©": "é", "√®": "è", "√ª": "ú", "Å": "A",
        "‚Äì": "-", "‚Äî": "—", "‚Äú": '"', "‚Äù": '"',
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "—",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def preprocess_abstract(text: str) -> str:
    """
    Full NLP preprocessing pipeline for a single abstract.
    Returns cleaned text ready for BERTopic.
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Fix encoding artifacts
    text = clean_encoding(text)

    # Step 2: Lowercase
    text = text.lower()

    # Step 3: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 4: Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Step 5: Remove numbers and special characters
    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Step 6: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 7: Tokenize
    tokens = word_tokenize(text)

    # Step 8: Remove stopwords and very short tokens (under 3 characters)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # Step 9: Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


# -----------------------------------------------------------------------------
#  SECTION 7: CLEANING PIPELINE FUNCTION
# -----------------------------------------------------------------------------

def run_cleaning_pipeline():
    """
    Main cleaning pipeline. Runs all quality filters and preprocessing.
    """

    log.info("=" * 60)
    log.info("  DATA CLEANING PIPELINE STARTING")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    # ── Load raw data ──────────────────────────────────────────────────────────
    input_file = DATA_RAW / "scopus_combined.csv"

    if not input_file.exists():
        log.error(f"❌ Input file not found: {input_file}")
        log.error("   Make sure you have run 1_data_collection_scopus.py first.")
        return

    log.info(f"📂 Loading: {input_file.name}")
    df = pd.read_csv(input_file, dtype=str, low_memory=False)
    log.info(f"   Loaded {len(df):,} papers with {len(df.columns)} columns")

    # Tracking removed papers for the cleaning report
    removal_log = []
    original_count = len(df)

    # ── FILTER 1: Exact duplicates by DOI ─────────────────────────────────────
    log.info("")
    log.info("🔍 Filter 1: Removing duplicate DOIs...")
    has_doi    = df[df["doi"].notna() & (df["doi"] != "")].copy()
    no_doi     = df[df["doi"].isna()  | (df["doi"] == "")].copy()
    before     = len(has_doi)
    has_doi    = has_doi.drop_duplicates(subset=["doi"], keep="first")
    removed    = before - len(has_doi)
    df         = pd.concat([has_doi, no_doi], ignore_index=True)
    removal_log.append(("Duplicate DOIs", removed))
    log.info(f"   Removed {removed:,} duplicate DOIs")

    # ── FILTER 2: Duplicate titles ─────────────────────────────────────────────
    log.info("🔍 Filter 2: Removing duplicate titles...")
    before  = len(df)
    df["_title_norm"] = df["title"].fillna("").str.lower().str.strip()
    df      = df.drop_duplicates(subset=["_title_norm"], keep="first")
    df      = df.drop(columns=["_title_norm"])
    removed = before - len(df)
    removal_log.append(("Duplicate titles", removed))
    log.info(f"   Removed {removed:,} duplicate titles")

    # ── FILTER 3: Missing abstracts ────────────────────────────────────────────
    log.info("🔍 Filter 3: Removing papers with missing abstracts...")
    before  = len(df)
    df      = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]
    removed = before - len(df)
    removal_log.append(("Missing abstracts", removed))
    log.info(f"   Removed {removed:,} papers with no abstract")

    # ── FILTER 4: Short abstracts ──────────────────────────────────────────────
    # Under 50 words = not enough text for BERTopic to extract meaningful topics
    log.info("🔍 Filter 4: Removing abstracts under 50 words...")
    before      = len(df)
    df["_wc"]   = df["abstract"].str.split().str.len()
    df          = df[df["_wc"] >= 50]
    df          = df.drop(columns=["_wc"])
    removed     = before - len(df)
    removal_log.append(("Abstracts under 50 words", removed))
    log.info(f"   Removed {removed:,} papers with abstracts under 50 words")

    # ── FILTER 5: Non-English abstracts ───────────────────────────────────────
    # Simple heuristic: if over 30% of words are not ASCII letters,
    # the abstract is likely not in English
    log.info("🔍 Filter 5: Flagging likely non-English abstracts...")
    before = len(df)

    def is_english(text):
        if not isinstance(text, str):
            return False
        words  = text.split()
        if not words:
            return False
        ascii_words = sum(1 for w in words if w.isascii())
        return (ascii_words / len(words)) >= 0.70  # 70% ASCII threshold

    df          = df[df["abstract"].apply(is_english)]
    removed     = before - len(df)
    removal_log.append(("Likely non-English abstracts", removed))
    log.info(f"   Removed {removed:,} likely non-English abstracts")

    # ── STANDARDISE: Country names ─────────────────────────────────────────────
    log.info("")
    log.info("🌍 Standardising country names...")
    df["primary_country"] = (
        df["primary_country"]
        .fillna("")
        .str.strip()
        .replace(COUNTRY_CORRECTIONS)
    )
    log.info("   Country names standardised")

    # ── STANDARDISE: Year column ───────────────────────────────────────────────
    log.info("📅 Standardising year column...")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # Keep only papers within our intended range
    before  = len(df)
    df      = df[df["year"].between(2015, 2025)]
    removed = before - len(df)
    if removed > 0:
        removal_log.append(("Year outside 2015-2025", removed))
        log.info(f"   Removed {removed:,} papers outside 2015-2025 range")
    df["year"] = df["year"].astype(int)

    # ── ADD: Permanent region column ──────────────────────────────────────────
    log.info("🗺️  Adding region column...")
    df["region"] = df["primary_country"].map(REGION_MAP).fillna("Other / Unclassified")

    # ── STANDARDISE: cited_by_count ───────────────────────────────────────────
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce").fillna(0).astype(int)

    # ── TEXT PREPROCESSING ────────────────────────────────────────────────────
    log.info("")
    log.info("🔤 Running full text preprocessing on abstracts...")
    log.info("   (lowercase → remove special chars → remove stopwords → lemmatize)")
    log.info("   This may take a few minutes for 40,000+ papers...")

    df["abstract_clean"] = df["abstract"].apply(preprocess_abstract)

    # Remove papers where preprocessing left an empty string
    before  = len(df)
    df      = df[df["abstract_clean"].str.strip() != ""]
    removed = before - len(df)
    if removed > 0:
        removal_log.append(("Empty after preprocessing", removed))
        log.info(f"   Removed {removed:,} papers with empty text after preprocessing")

    log.info("   Text preprocessing complete")

    # ── REORDER COLUMNS ───────────────────────────────────────────────────────
    # Put the most important columns first for easy browsing
    priority_cols = [
        "scopus_id", "doi", "title", "abstract", "abstract_clean",
        "year", "period", "region", "primary_country", "all_countries",
        "authors", "creator", "author_count", "affiliations",
        "affiliation_city", "journal", "cited_by_count",
        "open_access", "fund_sponsor", "keywords",
        "document_type", "subtype", "source",
    ]
    # Add any remaining columns not in priority list
    remaining = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + remaining]

    # ── PRINT CLEANING SUMMARY ────────────────────────────────────────────────
    total_removed = original_count - len(df)
    retention_pct = (len(df) / original_count * 100)

    log.info("")
    log.info("=" * 60)
    log.info("  CLEANING SUMMARY")
    log.info("=" * 60)
    log.info(f"  Original papers:     {original_count:,}")
    log.info(f"  Papers removed:      {total_removed:,}")
    log.info(f"  Papers retained:     {len(df):,}  ({retention_pct:.1f}%)")
    log.info("")
    log.info("  Removal breakdown:")
    for reason, count in removal_log:
        log.info(f"    {reason:<35} {count:>6,}")
    log.info("")
    log.info(f"  Pre-ChatGPT  (2015–2021): {(df['period'] == 'pre_chatgpt').sum():,}")
    log.info(f"  Post-ChatGPT (2022–2025): {(df['period'] == 'post_chatgpt').sum():,}")
    log.info(f"  Unique countries:          {df['primary_country'].replace('', pd.NA).nunique()}")
    log.info(f"  Unique regions:            {df['region'].nunique()}")
    log.info("")

    # Regional breakdown
    region_counts = df["region"].value_counts()
    log.info("  Regional distribution after cleaning:")
    log.info("  " + "-" * 45)
    for region in ["North America", "Europe", "Asia-Pacific",
                   "Latin America", "Africa & Middle East", "Other / Unclassified"]:
        count = region_counts.get(region, 0)
        pct   = (count / len(df) * 100) if len(df) > 0 else 0
        flag  = "  ⚠️  LOW" if pct < 5 else ""
        log.info(f"    {region:<25} {count:>6,}  ({pct:4.1f}%){flag}")
    log.info("  " + "-" * 45)
    log.info("=" * 60)

    # ── SAVE OUTPUTS ──────────────────────────────────────────────────────────
    log.info("")
    log.info("💾 Saving outputs...")

    # 1. Full cleaned dataset — CSV
    clean_csv = DATA_CLEAN / "scopus_cleaned.csv"
    df.to_csv(clean_csv, index=False, encoding="utf-8")
    log.info(f"   ✅ Full cleaned CSV:      {clean_csv.name}")

    # 2. Full cleaned dataset — Excel
    clean_xlsx = DATA_CLEAN / "scopus_cleaned.xlsx"
    df.to_excel(clean_xlsx, index=False)
    log.info(f"   ✅ Full cleaned Excel:    {clean_xlsx.name}")

    # 3. Abstracts-only CSV for BERTopic
    # Contains only what the NLP pipeline needs
    abstracts_df = df[[
        "scopus_id", "doi", "title",
        "abstract_clean",   # preprocessed text → feed this to BERTopic
        "abstract",         # original text → keep for reference
        "year", "period", "region", "primary_country",
        "journal", "cited_by_count", "keywords",
    ]].copy()

    abstracts_csv = DATA_CLEAN / "scopus_abstracts_nlp.csv"
    abstracts_df.to_csv(abstracts_csv, index=False, encoding="utf-8")
    log.info(f"   ✅ Abstracts NLP CSV:     {abstracts_csv.name}")

    # 4. Cleaning report — plain text for your methods section
    report_path = DATA_CLEAN / "scopus_cleaning_report.txt"
    with open(report_path, "w") as f:
        f.write("SCOPUS DATA CLEANING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("METHODOLOGICAL NOTE:\n")
        f.write("Quality-only filtering was applied. No thematic or keyword-based\n")
        f.write("filtering was used, in order to preserve the unsupervised nature\n")
        f.write("of the BERTopic analysis that follows.\n\n")
        f.write("FILTERING STEPS:\n")
        f.write(f"  Original corpus size:     {original_count:,}\n")
        for reason, count in removal_log:
            f.write(f"  Removed ({reason}): {count:,}\n")
        f.write(f"  Final corpus size:        {len(df):,}\n")
        f.write(f"  Retention rate:           {retention_pct:.1f}%\n\n")
        f.write("REGIONAL DISTRIBUTION:\n")
        for region in ["North America", "Europe", "Asia-Pacific",
                       "Latin America", "Africa & Middle East", "Other / Unclassified"]:
            count = region_counts.get(region, 0)
            pct   = (count / len(df) * 100) if len(df) > 0 else 0
            f.write(f"  {region:<25} {count:>6,}  ({pct:.1f}%)\n")
        f.write("\nTEXT PREPROCESSING APPLIED:\n")
        f.write("  - Encoding artifact removal\n")
        f.write("  - Lowercase conversion\n")
        f.write("  - URL and email removal\n")
        f.write("  - Special character removal\n")
        f.write("  - Stopword removal (NLTK English + domain-specific)\n")
        f.write("  - Lemmatization (NLTK WordNetLemmatizer)\n")
        f.write("  - Results stored in 'abstract_clean' column\n")
        f.write("  - Original abstract preserved in 'abstract' column\n")

    log.info(f"   ✅ Cleaning report:       {report_path.name}")
    log.info("")
    log.info("🎉 Cleaning pipeline complete!")
    log.info(f"   Open your data_clean/ folder to see the results.")
    log.info("")

    return df


# -----------------------------------------------------------------------------
#  SECTION 8: ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_cleaning_pipeline()