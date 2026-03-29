# =============================================================================
#  INSTITUTION GEOGRAPHIC ENRICHMENT SCRIPT
#  Project: Mapping Global AI Governance Narratives
#  Authors: Tambudzai Gundani & Joshua Gray
#  Date:    March 2026
# =============================================================================
#
#  WHAT THIS SCRIPT DOES:
#  ----------------------
#  Takes the cleaned Scopus dataset and extracts institution-level geographic
#  data for every paper. For each institution found it derives:
#
#    - institution name   (parsed from affiliations column)
#    - city               (parsed from affiliation_city column)
#    - country            (from primary_country column)
#    - state/province     (derived via geocoding where available)
#    - latitude           (geocoded from institution + city + country)
#    - longitude          (geocoded from institution + city + country)
#    - paper_count        (how many papers list this institution)
#
#  HOW GEOCODING WORKS:
#  --------------------
#  Geocoding = converting a place name into coordinates (lat/lon).
#  This script uses Nominatim (OpenStreetMap) via the geopy library.
#  It sends queries like "MIT, Cambridge, United States" and gets back
#  the coordinates of that location.
#
#  CACHING:
#  --------
#  Each unique institution is only geocoded ONCE. Results are saved to a
#  cache file. If you re-run the script, cached results are used instantly
#  without making new API calls. This saves time and respects rate limits.
#
#  RATE LIMITING:
#  --------------
#  Nominatim requires a 1 second pause between requests. With ~5,000-8,000
#  unique institutions this will take 1.5-2 hours to complete fully.
#  The script saves progress as it goes — if interrupted, re-running will
#  skip already-geocoded institutions and continue from where it left off.
#
#  OUTPUT FILES (saved to data_clean/ folder):
#  --------------------------------------------
#  - scopus_institutions.csv      — one row per unique institution
#                                   with count + coordinates
#  - scopus_geo_enriched.csv      — full dataset with lat/lon columns added
#  - geocoding_cache.json         — cache of all geocoded results
#  - geocoding_failures.csv       — institutions that could not be geocoded
# =============================================================================


# -----------------------------------------------------------------------------
#  SECTION 1: IMPORTS
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
#  SECTION 2: FOLDER SETUP
# -----------------------------------------------------------------------------

DATA_CLEAN = Path(__file__).parent.parent / "data_clean"
DATA_CLEAN.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
#  SECTION 3: LOGGING
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
#  SECTION 4: GEOCODER SETUP
#  user_agent identifies your app to Nominatim — required by their terms
#  of service. Using your project name here is correct practice.
# -----------------------------------------------------------------------------

geocoder = Nominatim(
    user_agent="GWU_AI_Governance_Capstone_Gundani_Gray_2026",
    timeout=10
)

CACHE_FILE = DATA_CLEAN / "geocoding_cache.json"


# -----------------------------------------------------------------------------
#  SECTION 5: CACHE FUNCTIONS
#  The cache is a JSON file that stores every geocoding result we've
#  ever retrieved. Key = "Institution Name | City | Country"
#  Value = {"lat": ..., "lon": ..., "state": ..., "display_name": ...}
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
#  SECTION 6: INSTITUTION PARSING FUNCTION
#  The affiliations column contains semicolon-separated institution names.
#  The affiliation_city column contains semicolon-separated city names.
#  This function pairs them up correctly.
# -----------------------------------------------------------------------------

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

    # Handle empty/missing values
    if not affiliations_str or not isinstance(affiliations_str, str):
        return records

    # Split on semicolons
    institutions = [i.strip() for i in affiliations_str.split(";") if i.strip()]
    cities       = [c.strip() for c in cities_str.split(";")] if isinstance(cities_str, str) else []

    for i, institution in enumerate(institutions):
        # Match city to institution by position — if no city at that index use empty string
        city = cities[i] if i < len(cities) else ""

        records.append({
            "institution": institution,
            "city":        city,
            "country":     country if isinstance(country, str) else "",
        })

    return records


# -----------------------------------------------------------------------------
#  SECTION 7: GEOCODING FUNCTION
#  Tries multiple query strategies in order of specificity.
#  Falls back to less specific queries if the precise one fails.
# -----------------------------------------------------------------------------

def geocode_institution(institution: str, city: str, country: str,
                         cache: dict) -> dict:
    """
    Get latitude, longitude and state for an institution.
    Uses cache first — only calls Nominatim if not already cached.

    Returns dict with lat, lon, state, display_name
    Returns None if geocoding fails completely.
    """

    # Build cache key from the three components
    cache_key = f"{institution} | {city} | {country}"

    # Return cached result if available
    if cache_key in cache:
        return cache[cache_key]

    # Try queries from most specific to least specific
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
            # Pause 1 second between requests — Nominatim rate limit requirement
            time.sleep(1.1)

            location = geocoder.geocode(query, addressdetails=True, language="en")

            if location:
                # Extract state/province from address details if available
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
                break  # Stop trying once we get a result

        except GeocoderTimedOut:
            time.sleep(3)  # Wait longer if timeout
            continue
        except GeocoderServiceError as e:
            log.warning(f"   Geocoder service error for '{query}': {e}")
            continue
        except Exception as e:
            log.warning(f"   Unexpected error for '{query}': {e}")
            continue

    # Cache the result (even if None — so we don't retry failed ones)
    cache[cache_key] = result

    # Save cache every 100 new lookups to protect against interruptions
    if len(cache) % 100 == 0:
        save_cache(cache)

    return result


# -----------------------------------------------------------------------------
#  SECTION 8: MAIN PIPELINE
# -----------------------------------------------------------------------------

def run_geocoding_pipeline():
    """
    Full institution geographic enrichment pipeline:
    1. Load cleaned Scopus data
    2. Parse all institutions from every paper
    3. Count how many papers each institution appears in
    4. Geocode each unique institution (using cache)
    5. Save institution summary CSV
    6. Merge coordinates back into main dataset
    7. Save enriched dataset
    """

    log.info("=" * 60)
    log.info("  INSTITUTION GEOCODING PIPELINE STARTING")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    # ── Load cleaned data ─────────────────────────────────────────────────────
    # Use cleaned dataset if available, otherwise fall back to raw
    input_file = DATA_CLEAN / "scopus_cleaned.csv"
    if not input_file.exists():
        input_file = Path(__file__).parent.parent / "data_raw" / "scopus_combined.csv"
        log.warning("   scopus_cleaned.csv not found — using scopus_combined.csv")

    if not input_file.exists():
        log.error("❌ No input file found. Run the collection and cleaning scripts first.")
        return

    log.info(f"📂 Loading: {input_file.name}")
    df = pd.read_csv(input_file, dtype=str, low_memory=False)
    log.info(f"   Loaded {len(df):,} papers")

    # ── Load geocoding cache ──────────────────────────────────────────────────
    log.info("")
    log.info("📦 Loading geocoding cache...")
    cache = load_cache()

    # ── Parse institutions from every paper ───────────────────────────────────
    log.info("")
    log.info("🔍 Parsing institutions from all papers...")

    # institution_papers maps institution_key → list of scopus_ids
    # institution_info  maps institution_key → {institution, city, country}
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
            # Create a unique key for each institution + city + country combo
            key = f"{rec['institution']} | {rec['city']} | {rec['country']}"
            institution_papers[key].add(scopus_id)
            institution_info[key] = rec

    log.info(f"   Papers with affiliation data:    {papers_with_affiliations:,}")
    log.info(f"   Papers without affiliation data: {papers_without:,}")
    log.info(f"   Unique institution records:      {len(institution_info):,}")

    # ── Geocode all unique institutions ───────────────────────────────────────
    log.info("")
    log.info("🌍 Geocoding institutions...")
    log.info("   (Using Nominatim/OpenStreetMap — 1 second pause between requests)")
    log.info("   Cached results load instantly. New lookups take ~1 second each.")

    # Figure out how many need geocoding vs are already cached
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

        # Progress update every 500 institutions
        if (i + 1) % 500 == 0:
            log.info(f"   ... processed {i + 1:,} / {len(institution_info):,} institutions "
                     f"({geocoded:,} geocoded, {failed:,} failed)")

    # Final cache save
    save_cache(cache)

    log.info(f"   ✅ Geocoded successfully: {geocoded:,}")
    log.info(f"   ❌ Could not geocode:     {failed:,}")

    # ── Build institution summary dataframe ───────────────────────────────────
    log.info("")
    log.info("📊 Building institution summary...")

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

    # Sort by paper count descending — most prolific institutions first
    inst_df = inst_df.sort_values("paper_count", ascending=False).reset_index(drop=True)

    log.info(f"   Total institutions:    {len(inst_df):,}")
    log.info(f"   With coordinates:      {inst_df['geocoded'].sum():,}")
    log.info(f"   Without coordinates:   {(~inst_df['geocoded']).sum():,}")
    log.info("")
    log.info("   Top 10 institutions by paper count:")
    for _, row in inst_df.head(10).iterrows():
        log.info(f"    {row['institution'][:45]:<45} {row['paper_count']:>5,} papers  "
                 f"{row['country']}")

    # ── Save institution summary CSV ──────────────────────────────────────────
    inst_path = DATA_CLEAN / "scopus_institutions.csv"
    inst_df.to_csv(inst_path, index=False, encoding="utf-8")
    log.info(f"\n✅ Institution summary saved: {inst_path.name}")

    # ── Save geocoding failures ───────────────────────────────────────────────
    if failures:
        fail_df   = pd.DataFrame(failures)
        fail_path = DATA_CLEAN / "geocoding_failures.csv"
        fail_df.to_csv(fail_path, index=False)
        log.info(f"⚠️  Failures saved:           {fail_path.name}  ({len(failures):,} rows)")

    # ── Merge coordinates back into main dataset ──────────────────────────────
    log.info("")
    log.info("🔗 Merging coordinates into main dataset...")

    # Build lookup: institution_key → coordinates
    # We use PRIMARY institution only for the main dataset merge
    # (first institution listed = first author's institution)
    geo_lookup = {}
    for key, geo in cache.items():
        if geo:
            geo_lookup[key] = geo

    # Extract primary institution key per paper
    def get_primary_key(row):
        affiliations = str(row.get("affiliations", "")) if pd.notna(row.get("affiliations")) else ""
        cities       = str(row.get("affiliation_city", "")) if pd.notna(row.get("affiliation_city")) else ""
        country      = str(row.get("primary_country", "")) if pd.notna(row.get("primary_country")) else ""

        if not affiliations:
            return None, None, None, None, None

        # Take first institution and first city
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

    # Apply to every row
    df[["primary_institution", "primary_city", "primary_state",
        "latitude", "longitude"]] = df.apply(
        lambda row: pd.Series(get_primary_key(row)), axis=1
    )

    # Coverage report
    has_coords = df["latitude"].notna().sum()
    log.info(f"   Papers with coordinates: {has_coords:,} / {len(df):,} "
             f"({has_coords/len(df)*100:.1f}%)")

    # ── Save enriched dataset ─────────────────────────────────────────────────
    enriched_path = DATA_CLEAN / "scopus_geo_enriched.csv"
    df.to_csv(enriched_path, index=False, encoding="utf-8")
    log.info(f"✅ Geo-enriched dataset saved: {enriched_path.name}")

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
    log.info("🎉 Pipeline complete!")
    log.info("")

    return inst_df, df


# -----------------------------------------------------------------------------
#  SECTION 9: ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_geocoding_pipeline()