"""
========================================================================================================================
INSTITUTION GEOCODING
Project: Uneven Science–Policy Translation Shapes Global AI Governance
Authors: Tambudzai G. Charumbira & Joshua Gray
Institution: George Washington University, CCAS | M.S. Data Science | Spring 2026
Date: March 2026

DESCRIPTION:
This script geocoded all 40,968 unique institutions in the Scopus corpus using OpenStreetMap Nominatim, achieving
99.96% coverage (18 failures). Each institution was resolved to latitude/longitude coordinates, enabling the
spatial analysis central to RO2.

Geocoding used a cascading query strategy: "Institution, City, Country" first, falling back to "City, Country",
then "Institution, Country", then country alone. Results were cached to a JSON file — re-runs skip previously
geocoded institutions, making the script idempotent. At Nominatim's required 1-second rate limit, initial
geocoding of ~8,000 unique institutions took approximately 2.5 hours; subsequent runs completed in seconds
from cache.

OUTPUT FILES (saved to data_clean/):
- scopus_institutions.csv    → One row per unique institution with coordinates and paper count
- scopus_geo_enriched.csv    → Full corpus with latitude/longitude columns added
- geocoding_cache.json       → Persistent cache of all geocoding results
- geocoding_failures.csv     → 18 institutions that could not be resolved
========================================================================================================================
"""
# ======================================================================================================================
#  SECTION 1: IMPORTS
# ======================================================================================================================
import os
import re
import json
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
# geopy handles geocoding via Nominatim (OpenStreetMap)
# Install with: pip install geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ======================================================================================================================
#  SECTION 2: FOLDER SETUP
# ======================================================================================================================
DATA_CLEAN = Path(__file__).parent.parent / "data_clean"
DATA_CLEAN.mkdir(parents=True, exist_ok=True)

# ======================================================================================================================
#  SECTION 3: LOGGING
# ======================================================================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(DATA_CLEAN / "geocoding.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ======================================================================================================================
#  SECTION 4: GEOCODER SETUP
#  Nominatim (OpenStreetMap) was selected over the Google Maps Geocoding API for being open-source and free at scale —
#  essential when processing 40,968 institutions without API cost constraints. The user_agent string identifies the
#  project to Nominatim as required by their terms of service.
# ======================================================================================================================
geocoder = Nominatim(
    user_agent="GWU_AI_Governance_Capstone_Gundani_Gray_2026",
    timeout=10
)
CACHE_FILE = DATA_CLEAN / "geocoding_cache.json"

# ======================================================================================================================
#  SECTION 5: CACHE FUNCTIONS
#  A JSON cache stored every geocoding result keyed by "Institution | City | Country". This ensured each institution was
#  queried only once across multiple script runs, respecting Nominatim's rate limits and allowing interrupted runs to
#  resume without redundant API calls.
# ======================================================================================================================
def load_cache() -> dict:
    """Load existing geocoding cache from disk."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        log.info(f"   Loaded {len(cache):,} cached geocoding results")
        return cache
    return {}

def save_cache(cache: dict):
    """Save geocoding cache to disk."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# ======================================================================================================================
#  SECTION 6: INSTITUTION PARSING
#  The Scopus affiliations column contains semicolon-separated institution names, and affiliation_city contains
#  corresponding city names. This function paired them by position, associating each institution with its city and the
#  paper's primary country.
# ======================================================================================================================
def parse_institutions(affiliations_str: str, cities_str: str, country: str) -> list:
    """
    Parse the affiliations and cities fields into a list of institution records.
    Example input:
        affiliations_str = "MIT; Harvard University; Stanford University"
        cities_str       = "Cambridge; Cambridge; Stanford"
        country          = "United States"
    Example output:
        [
            {"institution": "MIT",                "city": "Cambridge", "country": "United States"},
            {"institution": "Harvard University", "city": "Cambridge", "country": "United States"},
            {"institution": "Stanford University","city": "Stanford",  "country": "United States"},
        ]
    """
    records = []

    if not affiliations_str or not isinstance(affiliations_str, str):
        return records

    institutions = [i.strip() for i in affiliations_str.split(";") if i.strip()]
    cities       = [c.strip() for c in cities_str.split(";")] if isinstance(cities_str, str) else []

    for i, institution in enumerate(institutions):
        city = cities[i] if i < len(cities) else ""

        records.append({
            "institution": institution,
            "city":        city,
            "country":     country if isinstance(country, str) else "",
        })
    return records

# ======================================================================================================================
#  SECTION 7: GEOCODING FUNCTION
#  Each institution was geocoded using a cascading query strategy, trying progressively less specific queries until
#  coordinates were found:
#    1. "Institution, City, Country" (most specific)
#    2. "City, Country"
#    3. "Institution, Country"
#    4. Country alone (least specific)
#  Failed lookups were cached as None to prevent redundant retries on subsequent runs. The cache was saved to disk every
#  100 new lookups to protect against interruptions.
# ======================================================================================================================
def geocode_institution(institution: str, city: str, country: str,
                         cache: dict) -> dict:
    """
    Get latitude, longitude and state for an institution.
    Uses cache first — only calls Nominatim if not already cached.
    Returns dict with lat, lon, state, display_name
    Returns None if geocoding fails completely.
    """

    cache_key = f"{institution} | {city} | {country}"
    if cache_key in cache:
        return cache[cache_key]
    queries = []

    if institution and city and country:
        queries.append(f"{institution}, {city}, {country}")
    if city and country:
        queries.append(f"{city}, {country}")
    if institution and country:
        queries.append(f"{institution}, {country}")
    if country:
        queries.append(country)

    result = None

    for query in queries:
        try:
            time.sleep(1.1)
            location = geocoder.geocode(query, addressdetails=True, language="en")

            if location:
                address = location.raw.get("address", {})
                state   = (
                    address.get("state") or
                    address.get("province") or
                    address.get("region") or
                    address.get("county") or
                    ""
                )

                result = {
                    "lat":          round(location.latitude,  5),
                    "lon":          round(location.longitude, 5),
                    "state":        state,
                    "display_name": location.address,
                    "query_used":   query,
                }
                break

        except GeocoderTimedOut:
            time.sleep(3)
            continue
        except GeocoderServiceError as e:
            log.warning(f"   Geocoder service error for '{query}': {e}")
            continue
        except Exception as e:
            log.warning(f"   Unexpected error for '{query}': {e}")
            continue

    cache[cache_key] = result
    if len(cache) % 100 == 0:
        save_cache(cache)

    return result

# ======================================================================================================================
#  SECTION 8: MAIN PIPELINE
#  The pipeline executed the following steps:
#    1. Loaded the cleaned Scopus corpus and geocoding cache
#    2. Parsed all institutions from every paper's affiliation field
#    3. Counted papers per unique institution
#    4. Geocoded each unique institution (using cache where available)
#    5. Saved the institution summary CSV with coordinates and paper counts
#    6. Merged coordinates back into the main dataset via first-author institution lookup
#    7. Saved the geo-enriched dataset and geocoding failures log
# ======================================================================================================================
def run_geocoding_pipeline():
    """
    Full institution geographic enrichment pipeline:
    1. Load cleaned Scopus data
    2. Parse all institutions from every paper
    3. Count how many papers each institution appears in
    4. Geocode each unique institution (using cache)
    5. Save institution summary CSV
    6. Merge coordinates back into the main dataset
    7. Save enriched dataset
    """

    log.info("=" * 60)
    log.info("  INSTITUTION GEOCODING PIPELINE STARTING")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    # ── Loading cleaned data ─────────────────────────────────────────────────────
    input_file = DATA_CLEAN / "scopus_cleaned.csv"
    if not input_file.exists():
        input_file = Path(__file__).parent.parent / "data_raw" / "scopus_combined.csv"
        log.warning("   scopus_cleaned.csv not found — using scopus_combined.csv")
    if not input_file.exists():
        log.error(" No input file found. Run the collection and cleaning scripts first.")
        return
    log.info(f" Loading: {input_file.name}")
    df = pd.read_csv(input_file, dtype=str, low_memory=False)
    log.info(f"   Loaded {len(df):,} papers")

    # ── Loading geocoding cache ──────────────────────────────────────────────────
    log.info("")
    log.info(" Loading geocoding cache...")
    cache = load_cache()
    # ── Parsing institutions from every paper ───────────────────────────────────
    log.info("")
    log.info(" Parsing institutions from all papers...")
    institution_papers = defaultdict(set)
    institution_info   = {}
    papers_with_affiliations = 0
    papers_without           = 0

    for _, row in df.iterrows():
        scopus_id    = str(row.get("scopus_id", ""))
        affiliations = str(row.get("affiliations", "")) if pd.notna(row.get("affiliations")) else ""
        cities       = str(row.get("affiliation_city", "")) if pd.notna(row.get("affiliation_city")) else ""
        country      = str(row.get("primary_country", "")) if pd.notna(row.get("primary_country")) else ""

        if not affiliations:
            papers_without += 1
            continue
        papers_with_affiliations += 1
        records = parse_institutions(affiliations, cities, country)

        for rec in records:
            key = f"{rec['institution']} | {rec['city']} | {rec['country']}"
            institution_papers[key].add(scopus_id)
            institution_info[key] = rec

    log.info(f"   Papers with affiliation data:    {papers_with_affiliations:,}")
    log.info(f"   Papers without affiliation data: {papers_without:,}")
    log.info(f"   Unique institution records:      {len(institution_info):,}")

    # ── Geocoded all unique institutions ───────────────────────────────────────
    log.info("")
    log.info("   Geocoding institutions...")
    log.info("   (Using Nominatim/OpenStreetMap — 1 second pause between requests)")
    log.info("   Cached results load instantly. New lookups take ~1 second each.")

    needs_geocoding = sum(
        1 for key in institution_info
        if key not in cache or cache[key] is None
    )
    log.info(f"   Already cached: {len(institution_info) - needs_geocoding:,}")
    log.info(f"   Needs geocoding: {needs_geocoding:,}")

    if needs_geocoding > 0:
        est_minutes = round(needs_geocoding * 1.1 / 60, 1)
        log.info(f"   Estimated time for new lookups: ~{est_minutes} minutes")
    geocoded    = 0
    failed      = 0
    failures    = []

    for i, (key, info) in enumerate(institution_info.items()):
        result = geocode_institution(
            institution = info["institution"],
            city        = info["city"],
            country     = info["country"],
            cache       = cache
        )

        if result:
            geocoded += 1
        else:
            failed += 1
            failures.append({
                "key":         key,
                "institution": info["institution"],
                "city":        info["city"],
                "country":     info["country"],
            })

        if (i + 1) % 500 == 0:
            log.info(f"   ... processed {i + 1:,} / {len(institution_info):,} institutions "
                     f"({geocoded:,} geocoded, {failed:,} failed)")

    save_cache(cache)

    log.info(f"    Geocoded successfully: {geocoded:,}")
    log.info(f"    Could not geocode:     {failed:,}")

    # ── Building institution summary dataframe ───────────────────────────────────
    log.info("")
    log.info(" Building institution summary...")
    rows = []
    for key, info in institution_info.items():
        geo    = cache.get(key)
        count  = len(institution_papers[key])

        rows.append({
            "institution":  info["institution"],
            "city":         info["city"],
            "state":        geo["state"]        if geo else "",
            "country":      info["country"],
            "latitude":     geo["lat"]          if geo else None,
            "longitude":    geo["lon"]          if geo else None,
            "paper_count":  count,
            "geocoded":     bool(geo),
            "display_name": geo["display_name"] if geo else "",
        })

    inst_df = pd.DataFrame(rows)
    inst_df = inst_df.sort_values("paper_count", ascending=False).reset_index(drop=True)

    log.info(f"   Total institutions:    {len(inst_df):,}")
    log.info(f"   With coordinates:      {inst_df['geocoded'].sum():,}")
    log.info(f"   Without coordinates:   {(~inst_df['geocoded']).sum():,}")
    log.info("")
    log.info("   Top 10 institutions by paper count:")
    for _, row in inst_df.head(10).iterrows():
        log.info(f"    {row['institution'][:45]:<45} {row['paper_count']:>5,} papers  "
                 f"{row['country']}")

    # ── Saving institution summary CSV ──────────────────────────────────────────
    inst_path = DATA_CLEAN / "scopus_institutions.csv"
    inst_df.to_csv(inst_path, index=False, encoding="utf-8")
    log.info(f"\n Institution summary saved: {inst_path.name}")

    # ── Saving geocoding failures ───────────────────────────────────────────────
    if failures:
        fail_df   = pd.DataFrame(failures)
        fail_path = DATA_CLEAN / "geocoding_failures.csv"
        fail_df.to_csv(fail_path, index=False)
        log.info(f"  Failures saved:           {fail_path.name}  ({len(failures):,} rows)")

    # ── Merging coordinates back into main dataset ──────────────────────────────
    log.info("")
    log.info(" Merging coordinates into main dataset...")
    geo_lookup = {}
    for key, geo in cache.items():
        if geo:
            geo_lookup[key] = geo

    def get_primary_key(row):
        affiliations = str(row.get("affiliations", "")) if pd.notna(row.get("affiliations")) else ""
        cities       = str(row.get("affiliation_city", "")) if pd.notna(row.get("affiliation_city")) else ""
        country      = str(row.get("primary_country", "")) if pd.notna(row.get("primary_country")) else ""

        if not affiliations:
            return None, None, None, None, None

        first_inst = affiliations.split(";")[0].strip()
        first_city = cities.split(";")[0].strip() if cities else ""
        key        = f"{first_inst} | {first_city} | {country}"
        geo        = geo_lookup.get(key)

        if geo:
            return (
                first_inst,
                first_city,
                geo.get("state", ""),
                geo.get("lat"),
                geo.get("lon"),
            )
        return first_inst, first_city, "", None, None

    df[["primary_institution", "primary_city", "primary_state",
        "latitude", "longitude"]] = df.apply(
        lambda row: pd.Series(get_primary_key(row)), axis=1
    )

    has_coords = df["latitude"].notna().sum()
    log.info(f"   Papers with coordinates: {has_coords:,} / {len(df):,} "
             f"({has_coords/len(df)*100:.1f}%)")

    # ── Saving enriched dataset ─────────────────────────────────────────────────
    enriched_path = DATA_CLEAN / "scopus_geo_enriched.csv"
    df.to_csv(enriched_path, index=False, encoding="utf-8")
    log.info(f" Geo-enriched dataset saved: {enriched_path.name}")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("  GEOCODING SUMMARY")
    log.info("=" * 60)
    log.info(f"  Unique institutions found:    {len(inst_df):,}")
    log.info(f"  Successfully geocoded:        {geocoded:,}")
    log.info(f"  Failed to geocode:            {failed:,}")
    log.info(f"  Papers with coordinates:      {has_coords:,} ({has_coords/len(df)*100:.1f}%)")
    log.info("")
    log.info("  Output files in data_clean/:")
    log.info("    scopus_institutions.csv   — institution summary with coordinates")
    log.info("    scopus_geo_enriched.csv   — full dataset with lat/lon columns")
    log.info("    geocoding_cache.json      — cached results for future runs")
    if failures:
        log.info("    geocoding_failures.csv    — institutions that could not be geocoded")
    log.info("=" * 60)
    log.info("")
    log.info(" Pipeline complete!")
    log.info("")

    return inst_df, df

# ======================================================================================================================
#  SECTION 9: ENTRY POINT
# ======================================================================================================================
if __name__ == "__main__":
    run_geocoding_pipeline()