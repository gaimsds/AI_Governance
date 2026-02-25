"""
  ======================================================================================================================
  SCOPUS DATA COLLECTION SCRIPT
  Project: Global AI Governance Narratives
  Authors: Tambudzai Gundani & Joshua Gray
  Date:    February 2026

   This script connects to the Scopus database using the API key, searches for academic papers about AI governance
   published between 2015 and 2025, and saves the results to your data_raw/ folder.

   OUTPUT FILES (saved to data_raw/ folder):

   - scopus_pre_chatgpt_raw.csv     (papers from 2015-2021)
   - scopus_post_chatgpt_raw.csv    (papers from 2022-2025)
   - scopus_combined.csv            (all papers merged together)
   - scopus_combined.xlsx           (same, as an Excel file)
   - scopus_geo_summary.csv         (paper counts by country)
  ======================================================================================================================
"""
import os
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pybliometrics.scopus import ScopusSearch, AbstractRetrieval

load_dotenv()
API_KEY = os.getenv("SCOPUS_API_KEY")
DATA_RAW = Path(__file__).parent.parent / "data_raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# ======================================================================================================================
#  SECTION 1: LOGGING SETUP
#  Setting up a diary of everything the script does and keeping track for what happened if something goes wrong.
# ======================================================================================================================
logging.basicConfig(
    level=logging.INFO,                                          # INFO "show me normal updates" (not just errors)
    format="%(asctime)s  %(message)s",                           # Each log line starts with the time
    datefmt="%H:%M:%S",                                          # Time format: hours:minutes:seconds
    handlers=[

        logging.FileHandler(DATA_RAW / "scopus_collection.log"), # Writing logs to a file in data_raw/
        logging.StreamHandler(),                                 # Additional print logs to the PyCharm console
    ],
)
log = logging.getLogger(__name__)                                # Creating a logger

# ======================================================================================================================
#  SECTION 2: SEARCHING CONFIGURATION
#  Defining WHAT to search for and WHEN. Scopus Special query language.
#  TITLE-ABS-KEY(...)  = search in titles, abstracts, AND keywords
#  AND                 = both terms must appear
#  OR                  = either term can appear
#  W/3                 = words must appear within 3 words of each other
# The main search query — finds papers about AI + governance themes
# ======================================================================================================================
SEARCH_QUERY = (
    'TITLE-ABS-KEY('
    '("artificial intelligence" OR "machine learning" OR "algorithmic system*")'
    ' AND '
    '("governance" OR "regulation" OR "policy" OR "ethics" OR '
    '"accountability" OR "oversight" OR "AI governance" OR "AI ethics")'
    ')'
)
DATE_RANGES = {                                                     # Date ranges — ChatGPT launch (November 2022)
    "pre_chatgpt": {
        "label":      "Pre-ChatGPT (2015-2021)",                    # PUBYEAR > X "published after year X"
        "date_range": "PUBYEAR > 2014 AND PUBYEAR < 2022",          # 2015 to 2021
    },
    "post_chatgpt": {
        "label":      "Post-ChatGPT (2022-2025)",
        "date_range": "PUBYEAR > 2021 AND PUBYEAR < 2026",           # 2022 to 2025
    },
}
DOCTYPE_FILTER = "DOCTYPE(ar) OR DOCTYPE(re)"                        # Document types (ar = article, re = review paper)
                                                                     # Excluded book chapters, editorials, letters etc.
MAX_RESULTS = 2000                                                   # Maximum to collect per period to stay within rate limits

# ======================================================================================================================
#  SECTION 3: THE SEARCH FUNCTION
#  This function sends our query to Scopus and gets back a list of papers. We defined it once here and then call it twice
#  below (once for each time period).
# ======================================================================================================================
def search_scopus(period_key: str, config: dict) -> list:
    """
    Search Scopus for papers matching our query within a given date range.
    Parameters:
        period_key : short name like "pre_chatgpt" or "post_chatgpt"
        config     : dictionary containing label and date_range

    Returns:
        A list of Scopus result objects (raw search results)
    """
    full_query = f"{SEARCH_QUERY} AND {config['date_range']} AND ({DOCTYPE_FILTER})"
    log.info(f"")
    log.info(f"📅 Searching: {config['label']}")
    log.info(f"   Query: {full_query[:120]}...")

    try:
        results = ScopusSearch(
            query=full_query,
            view="STANDARD",
            count=MAX_RESULTS,
            download=True,
        )
        papers = results.results

        if papers is None:
            log.warning(f"   No results returned for {config['label']}")
            return []

        log.info(f"   Found {len(papers):,} papers")
        return papers

    except Exception as e:
        log.error(f"    Search failed for {config['label']}: {e}")
        return []

# ======================================================================================================================
#  SECTION 4: THE PARSING FUNCTION
#  Raw Scopus results come back as complex objects with lots of fields. This function takes each paper and extracts just
#  the fields we need,turning them into a simple flat row suitable for a spreadsheet.
# ======================================================================================================================
def parse_paper(paper, period: str) -> dict:
    """
    Extract the fields we need from a single Scopus paper result.
    Parameters:
        paper  : a single raw Scopus result object
        period : "pre_chatgpt" or "post_chatgpt"
    Returns:
        A dictionary (like one row of a spreadsheet) with all our fields
    """
    def safe(value):
        if value is None:
            return ""
        return str(value).strip()

    affiliations_text = ""
    all_countries = []
    primary_country = ""

    try:
        if paper.affiliation:
            affil_parts = []
            for affil in paper.affiliation:
                name    = getattr(affil, "affilname",      "") or ""
                country = getattr(affil, "affiliation_country", "") or ""

                if name:
                    affil_parts.append(name)
                if country and country not in all_countries:
                    all_countries.append(country)

            affiliations_text = "; ".join(affil_parts)
            if all_countries:
                primary_country = all_countries[0]

    except Exception:
        pass
    return {
        "scopus_id":       safe(paper.eid),                             # Scopus unique ID
        "doi":             safe(paper.doi),                             # Digital Object Identifier
        "title":           safe(paper.title),                           # Paper title
        "abstract":        safe(paper.description),                     # Abstract text
        "authors":         safe(paper.author_names),                    # All author names
        "affiliations":    affiliations_text,                           # Institution names
        "primary_country": primary_country,                             # First author's country ← key field
        "all_countries":   "; ".join(all_countries),                    # All countries in paper
        "journal":         safe(paper.publicationName),                 # Journal name
        "year":            safe(paper.coverDate[:4]) if paper.coverDate else "",
        "cited_by_count":  safe(paper.citedby_count),                   # Number of citations
        "keywords":        safe(paper.authkeywords),                    # Author-assigned keywords
        "document_type":   safe(paper.subtypeDescription),              # Article, Review, etc.
        "period":          period,                                      # "pre_chatgpt" or "post_chatgpt"
        "source":          "Scopus",                                    # which database this came from
    }

# ======================================================================================================================
#  SECTION 5: THE DEDUPLICATION FUNCTION
#  If the same paper appears in both time periods, or Scopus returns duplicates. This function removes them.
# ======================================================================================================================
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate papers from the combined dataset.
    Uses DOI as the primary key (since every paper has a unique DOI).
    Falls back to title matching for papers without a DOI.
    """
    original_count = len(df)
    has_doi = df[df["doi"] != ""].copy()                                  # Split into papers that have a DOI and those that don't
    no_doi  = df[df["doi"] == ""].copy()
    has_doi_clean = has_doi.drop_duplicates(subset=["doi"], keep="first") # Deduplicate by DOI — if same DOI appears twice, keep first occurrence

    no_doi["_title_norm"] = no_doi["title"].str.lower().str.strip()       # Deduplicate by title (normalized to lowercase) for papers without DOI
    no_doi_clean = no_doi.drop_duplicates(subset=["_title_norm"], keep="first")
    no_doi_clean = no_doi_clean.drop(columns=["_title_norm"])              # removing temp column

    result = pd.concat([has_doi_clean, no_doi_clean], ignore_index=True)   # Putting the two halves back together
    removed = original_count - len(result)
    log.info(f"    Deduplication: {original_count:,} → {len(result):,} papers "
             f"({removed:,} duplicates removed)")

    return result
# ======================================================================================================================
#  SECTION 6: THE SUMMARY FUNCTION
#  After collecting data, this prints a readable summary in the console to immediately verify output
# ======================================================================================================================
def print_summary(df: pd.DataFrame):
    """Print a human-readable summary of the collected data."""

    log.info("")
    log.info("=" * 55)
    log.info("   COLLECTION SUMMARY")
    log.info("=" * 55)
    log.info(f"  Total papers collected:    {len(df):,}")
    log.info(f"  Pre-ChatGPT  (2015–2021): "
             f"{(df['period'] == 'pre_chatgpt').sum():,}")
    log.info(f"  Post-ChatGPT (2022–2025): "
             f"{(df['period'] == 'post_chatgpt').sum():,}")
    log.info(f"  Papers with country data:  "
             f"{(df['primary_country'] != '').sum():,}")
    log.info(f"  Papers with abstract:      "
             f"{(df['abstract'] != '').sum():,}")
    log.info(f"  Unique countries:          "
             f"{df['primary_country'].replace('', pd.NA).nunique()}")

    top_countries = (
        df[df["primary_country"] != ""]["primary_country"]
        .value_counts()
        .head(10)
    )
    if not top_countries.empty:
        log.info("")
        log.info("   Top 10 countries (by first-author affiliation):")
        for country, count in top_countries.items():

            bar = "█" * min(int(count * 25 / top_countries.iloc[0]), 25)
            log.info(f"    {country:<25} {count:>5,}  {bar}")
    log.info("=" * 55)
    log.info("")

# ======================================================================================================================
#  SECTION 7: THE MAIN PIPELINE
# ======================================================================================================================
def run_pipeline():
    """
    Main function — runs the full Scopus data collection pipeline:
    1. Search for pre-ChatGPT papers (2015–2021)
    2. Search for post-ChatGPT papers (2022–2025)
    3. Parse and clean all results
    4. Combine and deduplicate
    5. Save outputs to data_raw/
    """
    log.info("=" * 55)
    log.info("   Scopus Data Collection Pipeline Starting")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 55)

    if not API_KEY:
        log.error(" SCOPUS_API_KEY not found in your .env file.")
        log.error("   Open your .env file and add:  SCOPUS_API_KEY=your_key_here")
        return

    log.info(f"   API key loaded successfully")
    log.info(f"   Output folder: {DATA_RAW}")
    log.info("")
    all_dataframes = []

    for period_key, config in DATE_RANGES.items():
        raw_papers = search_scopus(period_key, config)

        if not raw_papers:
            log.warning(f"    Skipping {period_key} — no results")
            continue

        log.info(f"    Parsing {len(raw_papers):,} papers...")
        rows = []
        failed = 0

        for i, paper in enumerate(raw_papers):
            try:
                row = parse_paper(paper, period=period_key)
                rows.append(row)
            except Exception as e:
                failed += 1
                continue

            if (i + 1) % 200 == 0:
                log.info(f"   ... parsed {i + 1:,} / {len(raw_papers):,} papers")

        if failed > 0:
            log.warning(f"     {failed} papers could not be parsed and were skipped")

        df_period = pd.DataFrame(rows)
        log.info(f"    Parsed {len(df_period):,} papers for {config['label']}")
        csv_filename = f"scopus_{period_key}_raw.csv"
        csv_path = DATA_RAW / csv_filename
        df_period.to_csv(csv_path, index=False, encoding="utf-8")
        log.info(f"    Saved: {csv_path.name}")
        all_dataframes.append(df_period)
        time.sleep(2)

    if not all_dataframes:
        log.error(" No data collected from any period. Check your query and API key.")
        return

    log.info("")
    log.info(" Combining all periods...")
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    df_combined = deduplicate(df_combined)
    print_summary(df_combined)

    combined_csv = DATA_RAW / "scopus_combined.csv"
    df_combined.to_csv(combined_csv, index=False, encoding="utf-8")
    log.info(f" Combined CSV saved:   {combined_csv.name}")

    combined_xlsx = DATA_RAW / "scopus_combined.xlsx"
    df_combined.to_excel(combined_xlsx, index=False)
    log.info(f" Combined Excel saved: {combined_xlsx.name}")

    geo_summary = (
        df_combined[df_combined["primary_country"] != ""]
        .groupby(["primary_country", "period"])
        .agg(
            paper_count   = ("scopus_id",       "count"),
            avg_citations = ("cited_by_count",  lambda x:
                             pd.to_numeric(x, errors="coerce").mean().round(2)),
        )
        .reset_index()
        .sort_values("paper_count", ascending=False)
    )
    geo_path = DATA_RAW / "scopus_geo_summary.csv"
    geo_summary.to_csv(geo_path, index=False)
    log.info(f" Geo summary saved:    {geo_path.name}")

    log.info("")
    log.info(" Pipeline complete! Open your data_raw/ folder to see the results.")
    log.info("")

    return df_combined

# ======================================================================================================================
#  SECTION 8: ENTRY POINT
# ======================================================================================================================
if __name__ == "__main__":
    run_pipeline()