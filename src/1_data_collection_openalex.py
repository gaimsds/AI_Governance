"""
AI Research Data Collection Script
====================================
Collects AI research papers from OpenAlex, filters to US-based corresponding authors,
retrieves full text, cleans it, validates the dataset, and prepares it for NLP analysis.

Requirements:
    pip install requests pandas pymupdf tqdm
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import time
import re
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from io import BytesIO

load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("collection_log.txt"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

EMAIL = os.getenv("EMAIL_J")          # Required by OpenAlex for polite pool access

# AI-related subject categories on OpenAlex
AI_CONCEPTS = {
    "Artificial Intelligence": "C154945302"
    #"Machine Learning":        "C119857082",
    #"Natural Language Processing": "C2522767166",
    #"Computer Vision":         "C31972630",
}

YEAR_START = 2020
YEAR_END = 2025
PER_PAGE = 200                            # Max allowed by OpenAlex per page is 200
MAX_FILES = 200000                        # Max number of files
REQUEST_DELAY = 1                         # Seconds between API calls (polite crawling)
PDF_TIMEOUT = 30                          # Seconds before PDF download times out
MEATDATA_FILE = "data_raw/open_alex_metadata.json"
OUTPUT_FILE = "data_raw/open_alex_abs_data.parquet"
CHECKPOINT_FILE = "oa_checkpoint.json"
FALLBACK_ABSTRACT_COUNT = 0              # Tracker for fallback to abstract

def save_checkpoint(all_works: list, concept_progress: dict):
    """Save current progress to checkpoint file."""
    checkpoint = {
        "all_works": all_works,
        "concept_progress": concept_progress
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)
    log.info(f"Checkpoint saved. {len(all_works)} works collected so far.")


def load_checkpoint() -> tuple[list, dict]:
    """Load progress from checkpoint file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        log.info(f"Checkpoint found. Resuming from {len(checkpoint['all_works'])} works.")
        return checkpoint["all_works"], checkpoint["concept_progress"]
    return [], {}

# ─────────────────────────────────────────────
# STEP 1: QUERY OPENALEX API
# Retrieve all open access AI research works from the past 5 years
# where at least one author is US-based.
# ─────────────────────────────────────────────

def build_query_url(concept_id: str, cursor: str = "*") -> str:
    """Construct a paginated OpenAlex API query URL for a given concept."""
    filters = ",".join([
        f"concepts.id:{concept_id}",
        f"publication_year:{YEAR_START}-{YEAR_END}",
        "authorships.institutions.country_code:US",
        "open_access.is_oa:true",
        "language:en"
    ])
    fields = ",".join([
        "id", "title", "doi", "publication_date",
        "open_access", "authorships", "abstract_inverted_index", "locations"
    ])
    return (
        f"https://api.openalex.org/works"
        f"?filter={filters}"
        f"&select={fields}"
        f"&per-page={PER_PAGE}"
        f"&cursor={cursor}"
        f"&sort=publication_date:desc"
        f"&mailto={EMAIL}"
    )


def fetch_all_works() -> list[dict]:
    """Page through all OpenAlex results with checkpointing to allow resume on failure."""
    all_works, concept_progress = load_checkpoint()
    seen_ids = set(work["id"] for work in all_works)

    for concept_name, concept_id in AI_CONCEPTS.items():

        # Skip concepts that were fully completed in a previous run
        if concept_progress.get(concept_name) == "complete":
            log.info(f"Skipping {concept_name} — already completed in previous run.")
            continue

        # Resume from saved cursor if available, otherwise start fresh
        cursor = concept_progress.get(concept_name, "*")
        log.info(f"Querying concept: {concept_name} | Resuming from cursor: {cursor}")
        page = 1

        while True:
            url = build_query_url(concept_id, cursor)
            response = requests.get(url)

            if response.status_code != 200:
                log.warning(f"Non-200 response on page {page} for {concept_name}: {response.status_code}")
                break

            data = response.json()
            results = data.get("results", [])

            if not results:
                log.info(f"No more results for {concept_name}.")
                concept_progress[concept_name] = "complete"
                save_checkpoint(all_works, concept_progress)
                break

            new_works = 0
            for work in results:
                work_id = work.get("id")
                if work_id and work_id not in seen_ids:
                    seen_ids.add(work_id)
                    all_works.append(work)
                    new_works += 1

            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")

            log.info(f"  {concept_name} | Page {page} | New works: {new_works} | Total: {len(all_works)}")

            if len(all_works) >= MAX_FILES:
                log.info(f"Max number of works retrieved: {len(all_works)}")
                return all_works

            # Save checkpoint after every page
            concept_progress[concept_name] = cursor
            save_checkpoint(all_works, concept_progress)

            if not cursor:
                concept_progress[concept_name] = "complete"
                save_checkpoint(all_works, concept_progress)
                break

            page += 1
            time.sleep(REQUEST_DELAY)

    log.info(f"Step 1 complete. Total unique works retrieved: {len(all_works)}")
    return all_works



# ─────────────────────────────────────────────
# STEP 2: FILTER TO US-BASED CORRESPONDING AUTHOR
# Keep only works where the corresponding author's institution
# is US-based. Fall back to majority-US-author rule if the
# corresponding author field is missing.
# ─────────────────────────────────────────────

def is_us_institution(affiliations: list[dict]) -> bool:
    """Return True if any affiliation in the list is US-based."""
    for affil in affiliations:
        country = affil.get("country_code", "")
        if country:
            if country.upper() == "US":
                return True
    return False


def majority_us_authors(authorships: list[dict]) -> bool:
    """Return True if more than half of authors have a US affiliation."""
    us_count = 0
    for authorship in authorships:
        institutions = authorship.get("institutions", [])
        if is_us_institution(institutions):
            us_count += 1
    return us_count > len(authorships) / 2


def filter_to_us_corresponding(works: list[dict]) -> list[dict]:
    """Filter works to those with a US-based corresponding author."""
    global FALLBACK_ABSTRACT_COUNT
    filtered = []
    us_found = False
    corr_count = 0
    foreign_corr_count = 0
    fallback_count = 0
    no_corresponding_count = 0

    for work in works:
        authorships = work.get("authorships", [])
        corresponding_authors = [a for a in authorships if a.get("is_corresponding")]

        if corresponding_authors:
            # Check if any corresponding author is US-based
            for author in corresponding_authors:
                institutions = author.get("institutions", [])
                if is_us_institution(institutions):
                    filtered.append(work)
                    corr_count += 1
                    us_found = True
                    break
            if not us_found:
                foreign_corr_count += 1
        else:
            # Fallback: use majority US authors rule
            no_corresponding_count += 1
            if majority_us_authors(authorships):
                work["_used_fallback"] = True
                filtered.append(work)
                fallback_count += 1

    log.info(f"Step 2 complete.")
    log.info(f"  Works passing US corresponding author filter: {corr_count}")
    log.info(f"  Works failing US corresponding author filter: {foreign_corr_count}")
    log.info(f"  Works missing corresponding author field: {no_corresponding_count}")
    log.info(f"  Works kept via majority-US fallback: {fallback_count}")
    return filtered



# ─────────────────────────────────────────────
# STEP 3: EXTRACT AND STORE METADATA
# Pull structured metadata from OpenAlex results before
# attempting any PDF downloads.
# ─────────────────────────────────────────────

def reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def extract_metadata(works: list[dict]) -> pd.DataFrame:
    """Extract relevant metadata fields from OpenAlex work objects into a DataFrame."""
    records = []

    for work in works:
        authorships = work.get("authorships", [])

        # Collect all unique institutions and countries across all authors
        institutions = []
        countries = []
        for authorship in authorships:
            for inst in authorship.get("institutions", []):
                inst_name = inst.get("display_name", "")
                country = inst.get("country_code", "")
                if inst_name:
                    institutions.append(inst_name)
                if country:
                    countries.append(country)

        # Get best open access URL
        oa_info = work.get("open_access", {})
        oa_url = oa_info.get("oa_url", "")

        # Fallback URL from locations if oa_url is missing
        if not oa_url:
            for loc in work.get("locations", []):
                if loc.get("pdf_url"):
                    oa_url = loc["pdf_url"]
                    break
                elif loc.get("landing_page_url"):
                    oa_url = loc["landing_page_url"]
                    break

        records.append({
            "openalex_id": work.get("id", ""),
            "title": work.get("title", ""),
            "doi": work.get("doi", ""),
            "publication_date": work.get("publication_date", ""),
            "institutions": "; ".join(set(institutions)),
            "countries": "; ".join(set(countries)),
            "oa_url": oa_url,
            "abstract": reconstruct_abstract(work.get("abstract_inverted_index", {})),
            "abstract_cleaned": ""
        })

    df = pd.DataFrame(records)
    log.info(f"Step 3 complete. Metadata extracted for {len(df)} works.")
    return df



# ─────────────────────────────────────────────
# STEP 4: RETRIEVE AND EXTRACT FULL TEXT
# Download PDFs using the OA URL and extract text with PyMuPDF.
# Fall back to the abstract if the download fails.
# ─────────────────────────────────────────────

def download_and_extract_pdf(url: str) -> str | None:
    """Download a PDF from a URL and extract its plain text."""
    try:
        response = requests.get(url, timeout=PDF_TIMEOUT, headers={"User-Agent": f"CapstoneResearch/{EMAIL}"})
        if response.status_code != 200:
            return None
        if "application/pdf" not in response.headers.get("Content-Type", ""):
            return None
        pdf_bytes = BytesIO(response.content)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text if text.strip() else None
    except Exception as e:
        log.debug(f"PDF extraction failed for {url}: {e}")
        return None


def retrieve_full_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Iterate through works and attempt to retrieve full text from PDF URLs."""
    pdf_success = 0
    abstract_fallback = 0
    total_failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading PDFs"):
        url = row["oa_url"]

        if url:
            text = download_and_extract_pdf(url)
            if text:
                df.at[idx, "full_text"] = text
                df.at[idx, "full_text_source"] = "pdf"
                pdf_success += 1
            else:
                # Fallback to abstract
                df.at[idx, "full_text"] = row["abstract"]
                df.at[idx, "full_text_source"] = "abstract_fallback"
                abstract_fallback += 1
                log.debug(f"PDF failed, using abstract for: {row['title']}")
        else:
            # No URL available at all
            df.at[idx, "full_text"] = row["abstract"]
            df.at[idx, "full_text_source"] = "abstract_fallback_no_url"
            abstract_fallback += 1
            total_failed += 1

        time.sleep(0.5)  # Be polite to host servers

    log.info(f"Step 4 complete.")
    log.info(f"  Successful PDF extractions: {pdf_success}")
    log.info(f"  Fell back to abstract: {abstract_fallback}")
    log.info(f"  No URL available: {total_failed}")
    return df



# ─────────────────────────────────────────────
# STEP 5: CLEAN THE TEXT
# Remove formatting artifacts, excessive whitespace,
# garbled characters, and citation noise from extracted text.
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean extracted PDF or abstract text for NLP analysis."""
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove citation markers like [1], [2,3], (Smith et al., 2020)
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\([A-Z][a-z]+ et al\.,?\s*\d{4}\)", "", text)

    # Remove LaTeX-style math expressions
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)

    # Remove lines that are just numbers (page numbers, table data)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove excessive whitespace and normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove non-ASCII characters (garbled PDF artifacts)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    return text.strip()


def clean_all_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to all abstract entries in the DataFrame."""
    df["abstract_cleaned"] = df["abstract"].apply(clean_text)
    empty_after_clean = (df["abstract_cleaned"].str.strip() == "").sum()
    log.info(f"Step 5 complete. Text cleaned for {len(df)} works.")
    log.info(f"  Works with empty text after cleaning: {empty_after_clean}")
    return df



# ─────────────────────────────────────────────
# STEP 6: VALIDATE THE DATASET
# Check for duplicates, empty text, missing metadata,
# and summarize how many records passed each filtering step.
# ─────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run quality checks and report on dataset integrity."""
    log.info("Step 6: Validating dataset...")

    initial_count = len(df)

    # Remove duplicates by OpenAlex ID
    df = df.drop_duplicates(subset="openalex_id")
    duplicates_removed = initial_count - len(df)

    # Flag records with no usable text
    no_text = df["abstract_cleaned"].str.strip().eq("").sum()

    # Flag records missing key metadata
    missing_date = df["publication_date"].eq("").sum()
    missing_institution = df["institutions"].eq("").sum()
    missing_title = df["title"].eq("").sum()

    # Summary report
    log.info(f"  Total records: {len(df)}")
    log.info(f"  Duplicates removed: {duplicates_removed}")
    log.info(f"  Records with no usable text: {no_text}")
    log.info(f"  Records missing publication date: {missing_date}")
    log.info(f"  Records missing institution: {missing_institution}")
    log.info(f"  Records missing title: {missing_title}")
    #log.info(f"  Full text from PDF: {(df['abstract_source'] == 'pdf').sum()}")
    #log.info(f"  Full text from abstract fallback: {df['abstract_source'].str.contains('abstract').sum()}")
    #log.info(f"  Records using majority-US fallback filter: {df['used_fallback_filter'].sum()}")

    log.info("Step 6 complete. Dataset is ready for NLP analysis.")
    return df

# ─────────────────────────────────────────────
# STEP 7: SAVE DATASET FOR NLP ANALYSIS
# Export the cleaned, validated dataset to CSV.
# The full_text_cleaned column is ready for NLP input.
# ─────────────────────────────────────────────

def save_data_json(works: list):
    with open(MEATDATA_FILE, "w") as f:
        json.dump(works, f, indent=2)
    log.info(f"Results saved to '{MEATDATA_FILE}'.")



# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────


log.info("Starting AI research data collection pipeline.")

#%% Step 1
works = fetch_all_works()
save_data_json(works)

#%% Step 2
filtered_works = filter_to_us_corresponding(works)

#%% Step 3
df = extract_metadata(filtered_works)
df.replace('', np.nan, inplace=True)
print(f"Missing abstracts from {df['abstract'].isnull().sum()} entries: {df['abstract'].count()} entries remaining.")
df = df.dropna(subset='abstract').reset_index(drop=True)
print("New dataset")
print(df.info())

#%% Step 4
#df = retrieve_full_texts(df)
#df.to_parquet("data_raw/meta_data_step4.parquet", index=False)

#%% Step 5
df = clean_all_texts(df)
log.info(f"Step 5 Complete. Text cleaned for {len(df)} works.")

#%% Step 6
df = validate_dataset(df)
df.replace('', np.nan, inplace=True)
print(f"Missing abstracts from {df['abstract_cleaned'].isnull().sum()} entries: {df['abstract_cleaned'].count()} entries remaining.")
df = df.dropna(subset='abstract_cleaned').reset_index(drop=True)
df.to_parquet(OUTPUT_FILE, index=False)

log.info(f"Step 6 complete. Dataset saved to '{OUTPUT_FILE}' with {len(df)} records.")
log.info("The 'full_text_cleaned' column is ready for NLP analysis.")
log.info("Pipeline complete.")

#%%
print(df.info())


