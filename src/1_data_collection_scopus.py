"""
========================================================================================================================
SCOPUS ACADEMIC CORPUS COLLECTION
Project: Uneven Science–Policy Translation Shapes Global AI Governance
Authors: Tambudzai G. Charumbira & Joshua Gray
Institution: George Washington University, CCAS | M.S. Data Science | Spring 2026
Date: February 2026

DESCRIPTION:
This script collected the primary academic corpus from the Scopus database via the Elsevier API. The search targeted
peer-reviewed AI publications at the intersection of governance, ethics, regulation, and policy — published between
2015 and 2025 — and stratified the results into pre-ChatGPT (2015–2021) and post-ChatGPT (2022–2025) periods.

Scopus was selected over Web of Science and Google Scholar for its broader coverage of non-English and Global South
journals (27,000+ journals vs WoS's ~21,000), which was essential for a study examining spatial distribution and
regional variation in AI governance discourse. The pybliometrics Python library was used for API interaction, with
cursor-based pagination to retrieve complete result sets within Elsevier's rate limits.

The collection yielded 46,583 raw records across both periods, which were subsequently deduplicated and cleaned in
the data cleaning stage (2_data_cleaning_scopus.py) to produce the final corpus of 41,067 papers.

SEARCH STRATEGY:
- AI terms were required in the TITLE field (not just abstract or keywords) to ensure papers were fundamentally
  about AI rather than merely mentioning it peripherally
- Subject area filtering restricted results to Social Sciences, Law, Multidisciplinary, Business, Arts & Humanities,
  Decision Sciences, Economics, and Psychology — excluding purely technical domains (e.g., Computer Science only)
  to focus on governance-relevant discourse
- Document types were limited to articles and review papers (excluding editorials, book chapters, letters)
- Language was restricted to English, acknowledged as a limitation in the paper (Section 5.5)

OUTPUT FILES (saved to data_raw/):
- scopus_pre_chatgpt_raw.csv     → 8,904 papers from 2015–2021
- scopus_post_chatgpt_raw.csv    → 37,679 papers from 2022–2025
- scopus_combined.csv            → All papers merged and deduplicated
- scopus_combined.xlsx           → Same dataset in Excel format for manual inspection
- scopus_geo_summary.csv         → Paper counts by country and period for initial geographic audit
- scopus_collection.log          → Full execution log with timestamps and API quota tracking
========================================================================================================================
"""
import os
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus import init

load_dotenv()
API_KEY = os.getenv("SCOPUS_API_KEY")
init(keys=[API_KEY])
DATA_RAW = Path(__file__).parent.parent / "data_raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# ======================================================================================================================
#  SECTION 1: LOGGING SETUP
#  A structured logging configuration was established to create a full audit trail of the collection process. All
#  API interactions, result counts, errors, and quota checks were logged to both a persistent file
#  (data_raw/scopus_collection.log) and the console. This dual-output approach ensured that the collection process
#  was fully reproducible and that any API failures or rate limit issues could be diagnosed after the fact.
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
#  SECTION 2: SEARCH CONFIGURATION
#  The search strategy was designed to balance comprehensiveness with precision. AI-related terms were required in the
#  TITLE field — a deliberate choice to ensure every paper in the corpus was fundamentally about artificial intelligence,
#  rather than papers that mention AI tangentially in an abstract about an unrelated topic.
#
#  Subject area filtering was applied to focus on governance-relevant disciplines. The Scopus subject area codes used:
#    - SOCI (Social Sciences)       - LAWS (Law)
#    - MULT (Multidisciplinary)     - BUSI (Business & Management)
#    - ARTS (Arts & Humanities)     - DECI (Decision Sciences)
#    - ECON (Economics)             - PSYC (Psychology)
#
#  This cross-disciplinary scope was essential because AI governance scholarship spans multiple fields, a purely
#  computer science query would miss legal, ethical, and policy-oriented research.
#
#  The corpus was temporally stratified at the ChatGPT release boundary (November 2022). Because Scopus metadata
#  records publication year rather than month, papers from 2022 were assigned to the post-ChatGPT period — a
#  conservative choice that slightly inflates the post-ChatGPT count but avoids arbitrary month-level splits.
#
#  Language was restricted to English. This is acknowledged as a limitation: governance discourse in Chinese, Arabic,
#  French, and other languages may frame concerns differently and is not captured in this corpus.
# ======================================================================================================================
SEARCH_QUERY = (
    'TITLE('
    '"artificial intelligence" OR "machine learning" OR '
    '"algorithmic system*" OR "automated decision*" OR "AI system*"'
    ')'
    
    ' AND SUBJAREA('
    'SOCI OR LAWS OR MULT OR BUSI OR ARTS OR DECI OR ECON OR PSYC'
    ')'
    ' AND LANGUAGE(english)'
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
                                                                     # Excluded book chapters, editorials, letters, etc.
MAX_RESULTS = 99999                                                  # Maximum to collect per period to stay within rate

# ======================================================================================================================
#  SECTION 3: SCOPUS API SEARCH FUNCTION
#  This function executed the Scopus search for a given time period. The pybliometrics library's ScopusSearch class
#  was used with the COMPLETE view (which returns full metadata including affiliations and abstracts) and cursor-based
#  pagination to automatically retrieve all results beyond Scopus's 25-result-per-page limit.
#
#  After each search, the remaining API quota was checked via a lightweight probe request and logged. This was necessary
#  because the Elsevier API enforces rate limits (typically 5,000 requests per week for institutional keys).The function
#  returned raw Scopus result objects, which were subsequently parsed in Section 4.
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
    log.info(f" Searching: {config['label']}")
    log.info(f"   Query: {full_query[:120]}...")
    try:
        results = ScopusSearch(
            query=full_query,
            view="COMPLETE",
            count=25,                                                # COMPLETE view hard limit per request
            cursor=True,                                             # Enabling automatic pagination through all results
            download=True,
        )
                                                                     # Checking remaining quota
        import requests
        r = requests.get(
            "https://api.elsevier.com/content/search/scopus",
            params={"query": "test", "count": "1"},
            headers={"X-ELS-APIKey": API_KEY}
        )
        log.info(f"    Quota Limit:     {r.headers.get('X-RateLimit-Limit', 'N/A')}")
        log.info(f"    Quota Remaining: {r.headers.get('X-RateLimit-Remaining', 'N/A')}")
        log.info(f"    Quota Resets:    {r.headers.get('X-RateLimit-Reset', 'N/A')}")

        papers = results.results
        if papers is None:
            log.warning(f"   No results returned for {config['label']}")
            return []
        if len(papers) > MAX_RESULTS:
            log.info(f"   Found {len(papers):,} papers — capping at {MAX_RESULTS:,}")
            papers = papers[:MAX_RESULTS]
        else:
            log.info(f"   Found {len(papers):,} papers")
        return papers
    except Exception as e:
        log.error(f"    Search failed for {config['label']}: {e}")
        return []

# ======================================================================================================================
#  SECTION 4: PAPER PARSING FUNCTION
#  Each raw Scopus result object was parsed into a flat dictionary containing 32 fields organized into seven
#  categories: identifiers (EID, DOI), content (title, abstract, keywords), author metadata (names, IDs, counts),
#  geographic attribution (affiliations, countries), publication details (journal, date, volume), impact metrics
#  (citation count, open access status), and research metadata (period label, source tag).
#
#  Geographic attribution required special handling. The affiliation fields in Scopus results can contain multiple
#  institutions separated by semicolons (for multi-affiliation authors). The parser extracted all affiliated countries
#  into an 'all_countries' field and assigned the first-listed country as 'primary_country' — following the standard
#  bibliometric convention of first-author affiliation (Waltman, 2016). This primary country field was later used
#  for geographic attribution in the analysis, while the full country list was used for co-authorship network
#  construction (2c_coauthorship_edges_scopus.py).
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
        raw_name = getattr(paper, "affilname", None)
        raw_country = getattr(paper, "affiliation_country", None)
        raw_city = getattr(paper, "affiliation_city", None)

        if raw_name:
            affiliations_text = str(raw_name).strip()

        if raw_country:
            countries = [c.strip() for c in str(raw_country).split(";") if c.strip()]
            all_countries = list(dict.fromkeys(countries))
            primary_country = all_countries[0] if all_countries else ""

    except Exception as e:
        log.debug(f"Affiliation parsing error for {getattr(paper, 'eid', 'unknown')}: {e}")
        pass

    return {
        # ── Identifiers ───────────────────────────────────────────
        "scopus_id": safe(paper.eid),
        "doi": safe(paper.doi),
        "pubmed_id": safe(paper.pubmed_id),
        "source_id": safe(paper.source_id),

        # ── Content ───────────────────────────────────────────────
        "title": safe(paper.title),
        "abstract": safe(paper.description),
        "keywords": safe(paper.authkeywords),

        # ── Authors ───────────────────────────────────────────────
        "creator": safe(paper.creator),                                         # first author
        "authors": safe(paper.author_names),                                    # all authors
        "author_count": safe(paper.author_count),
        "author_ids": safe(paper.author_ids),
        "author_afids": safe(paper.author_afids),                               # per-author affiliation IDs

        # ── Geography & Affiliation ────────────────────────────────
        "afid": safe(paper.afid),                                               # institution ID
        "affiliations": affiliations_text,
        "affiliation_city": safe(paper.affiliation_city),
        "primary_country": primary_country,
        "all_countries": "; ".join(all_countries),

        # ── Publication ───────────────────────────────────────────
        "journal": safe(paper.publicationName),
        "issn": safe(paper.issn),
        "eissn": safe(paper.eIssn),
        "volume": safe(paper.volume),
        "issue": safe(paper.issueIdentifier),
        "page_range": safe(paper.pageRange),
        "cover_date": safe(paper.coverDate),
        "year": safe(paper.coverDate[:4]) if paper.coverDate else "",
        "aggregation_type": safe(paper.aggregationType),                         # Journal vs Book Series

        # ── Impact & Access ───────────────────────────────────────
        "cited_by_count": safe(paper.citedby_count),
        "open_access": safe(paper.openaccess),
        "open_access_label": safe(paper.freetoreadLabel),

        # ── Funding ───────────────────────────────────────────────
        "fund_sponsor": safe(paper.fund_sponsor),
        "fund_acronym": safe(paper.fund_acr),
        "fund_number": safe(paper.fund_no),

        # ── Document Type ─────────────────────────────────────────
        "subtype": safe(paper.subtype),
        "document_type": safe(paper.subtypeDescription),

        # ── Research Metadata ─────────────────────────────────────
        "period": period,
        "source": "Scopus",
    }

# ======================================================================================================================
#  SECTION 5: DEDUPLICATION
#  Duplicate records were removed using a two-tier strategy. Papers with DOIs were deduplicated on the DOI field
#  (each published paper has a unique DOI). Papers lacking DOIs — a small minority — were deduplicated on
#  normalized title strings (lowercased, stripped of whitespace). In both cases, the first occurrence was retained.
#
#  Deduplication was necessary because the two temporal queries (pre- and post-ChatGPT) could return overlapping
#  results for papers published near the boundary, and because Scopus occasionally returns duplicate entries for
#  papers indexed under multiple source IDs.
# ======================================================================================================================
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate papers from the combined dataset.
    Uses DOI as the primary key (since every paper has a unique DOI).
    Falls back to title matching for papers without a DOI.
    """
    original_count = len(df)
    has_doi = df[df["doi"] != ""].copy()
    no_doi  = df[df["doi"] == ""].copy()
    has_doi_clean = has_doi.drop_duplicates(subset=["doi"], keep="first")

    no_doi["_title_norm"] = no_doi["title"].str.lower().str.strip()
    no_doi_clean = no_doi.drop_duplicates(subset=["_title_norm"], keep="first")
    no_doi_clean = no_doi_clean.drop(columns=["_title_norm"])

    result = pd.concat([has_doi_clean, no_doi_clean], ignore_index=True)
    removed = original_count - len(result)
    log.info(f"    Deduplication: {original_count:,} → {len(result):,} papers "
             f"({removed:,} duplicates removed)")

    return result
# ======================================================================================================================
#  SECTION 6: COLLECTION SUMMARY AND REGIONAL AUDIT
#  After collection and deduplication, a structured summary was printed and logged. This included total paper counts,
#  period splits, geographic coverage statistics, and the top 10 countries by first-author affiliation.
#
#  A regional coverage audit was performed by mapping each country to one of five analytical regions (North America,
#  Europe, Asia-Pacific, Latin America, Africa & Middle East) plus an "Other / Unclassified" residual category.
#  Regions representing less than 5% of the corpus were flagged for discussion in the methodology section, and
#  regions below 2% were flagged as potential limitations. This audit was designed to surface geographic biases
#  in the corpus at the earliest possible stage, before downstream analysis.
#
#  The regional underrepresentation of certain areas (particularly Africa and Latin America) was itself identified
#  as a substantive finding about the structure of global AI governance research production.
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
        log.info("  Top 10 countries (by first-author affiliation):")
        for country, count in top_countries.items():
            bar = "█" * min(int(count * 25 / top_countries.iloc[0]), 25)
            log.info(f"    {country:<25} {count:>5,}  {bar}")

    region_map = {
        "United States": "North America", "Canada": "North America",                # North America

        "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",        # Europe
        "Netherlands": "Europe", "Italy": "Europe", "Spain": "Europe",
        "Sweden": "Europe", "Switzerland": "Europe", "Norway": "Europe",
        "Denmark": "Europe", "Finland": "Europe", "Belgium": "Europe",
        "Austria": "Europe", "Poland": "Europe", "Portugal": "Europe",
        "Ireland": "Europe", "Greece": "Europe", "Czech Republic": "Europe",
        "Hungary": "Europe", "Romania": "Europe",

        "China": "Asia-Pacific", "India": "Asia-Pacific",                            # Asia-Pacific
        "Japan": "Asia-Pacific", "South Korea": "Asia-Pacific",
        "Australia": "Asia-Pacific", "Singapore": "Asia-Pacific",
        "New Zealand": "Asia-Pacific", "Taiwan": "Asia-Pacific",
        "Hong Kong": "Asia-Pacific", "Malaysia": "Asia-Pacific",
        "Indonesia": "Asia-Pacific", "Thailand": "Asia-Pacific",
        "Vietnam": "Asia-Pacific", "Philippines": "Asia-Pacific",
        "Pakistan": "Asia-Pacific", "Bangladesh": "Asia-Pacific",
        "Sri Lanka": "Asia-Pacific",

        "Brazil": "Latin America", "Mexico": "Latin America",                        # Latin America
        "Argentina": "Latin America", "Colombia": "Latin America",
        "Chile": "Latin America", "Peru": "Latin America",
        "Venezuela": "Latin America", "Ecuador": "Latin America",
        "Bolivia": "Latin America", "Uruguay": "Latin America",
        "Costa Rica": "Latin America",

        "South Africa": "Africa & Middle East",                                      # Africa & Middle East
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
        "Israel": "Africa & Middle East",
        "Saudi Arabia": "Africa & Middle East",
        "UAE": "Africa & Middle East",
        "Qatar": "Africa & Middle East",
        "Jordan": "Africa & Middle East",
        "Lebanon": "Africa & Middle East",
    }

    df_with_country = df[df["primary_country"] != ""].copy()
    df_with_country["region"] = df_with_country["primary_country"].map(
        region_map
    ).fillna("Other / Unclassified")

    region_counts = df_with_country["region"].value_counts()

    log.info("")
    log.info(" Regional Coverage Audit (per stratification):")
    log.info("  " + "-" * 45)

    for region in ["North America", "Europe", "Asia-Pacific",
                   "Latin America", "Africa & Middle East",
                   "Other / Unclassified"]:
        count = region_counts.get(region, 0)
        total = len(df_with_country)
        pct = (count / total * 100) if total > 0 else 0

        if pct < 2:                                                                 # Flag low-representation regions
            flag = " VERY LOW — flag as limitation"
        elif pct < 5:
            flag = "  LOW — note in methodology"
        else:
            flag = ""

        log.info(f"    {region:<25} {count:>5,}  ({pct:4.1f}%){flag}")

    log.info("  " + "-" * 45)
    log.info("      Regions below 5% should be discussed as limitations.")
    log.info("      This is expected — it is itself a key finding for the project.")
    log.info("=" * 55)
    log.info("")

# ======================================================================================================================
#  SECTION 7: MAIN COLLECTION PIPELINE
#  The pipeline executed the following steps in sequence:
#    1. Validated the Scopus API key from the .env file
#    2. Searched for pre-ChatGPT papers (2015–2021) and parsed results
#    3. Saved period-specific CSV to data_raw/
#    4. Searched for post-ChatGPT papers (2022–2025) and parsed results
#    5. Saved period-specific CSV to data_raw/
#    6. Combined both periods and deduplicated
#    7. Generated and logged the collection summary and regional audit
#    8. Saved combined outputs in CSV, Excel, and geographic summary formats
#
#  A 2-second delay was inserted between period searches to respect Elsevier's API rate limits. Failed paper parses were
#  counted and logged but did not halt the pipeline — ensuring that individual malformed records did not prevent the
#  collection of the remaining corpus.
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
        log.error(" SCOPUS_API_KEY not found in .env file.")
        log.error("   Open .env file and add:  SCOPUS_API_KEY=put_key_here")
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
        log.error(" No data collected from any period. Check query and API key.")
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
    log.info(" Pipeline complete! Open data_raw/ folder to see the results.")
    log.info("")

    return df_combined

# ======================================================================================================================
#  SECTION 8: ENTRY POINT
#  Standard Python entry point. When executed directly (python 1_data_collection_scopus.py), the full collection
#  pipeline ran and saved all outputs to the data_raw/ directory. The script was designed to be idempotent — running
#  it again would overwrite previous outputs with a fresh collection.
# ======================================================================================================================
if __name__ == "__main__":
    run_pipeline()