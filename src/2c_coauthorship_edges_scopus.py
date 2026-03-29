# =============================================================================
#  COAUTHORSHIP EDGES SCRIPT
#  Project: Mapping Global AI Governance Narratives
#  Authors: Tambudzai Gundani & Joshua Gray
#  Date:    March 2026
# =============================================================================
#
#  WHAT THIS SCRIPT DOES:
#  ----------------------
#  Takes the geocoded institution data and builds an EDGE LIST —
#  a file that records every pair of institutions that appear together
#  on the same paper. This edge list is what powers the network map
#  showing collaboration connections between countries and institutions.
#
#  HOW EDGES WORK:
#  ---------------
#  If a paper has 3 institutions: MIT, Oxford, University of Nairobi
#  That creates 3 edges:
#    MIT ←→ Oxford
#    MIT ←→ University of Nairobi
#    Oxford ←→ University of Nairobi
#
#  OUTPUTS (saved to data_clean/):
#  ---------------------------------
#  1. scopus_institution_edges.csv
#     One row per institution PAIR per paper.
#     institution_a, country_a, lat_a, lon_a,
#     institution_b, country_b, lat_b, lon_b,
#     paper_count (how many papers share this pair)
#
#  2. scopus_country_edges.csv
#     Aggregated to COUNTRY level — one row per country pair.
#     country_a, country_b, paper_count, region_a, region_b
#     This is what you feed directly into your network map.
#
#  3. scopus_country_nodes.csv
#     One row per country with total paper count and coordinates.
#     Used as the node layer in your network map.
#
#  WHY TWO LEVELS:
#  ---------------
#  Institution edges → detailed academic network analysis
#  Country edges     → spatial network map (lines between countries)
#  Both are needed for your RO2 analysis.
# =============================================================================


# -----------------------------------------------------------------------------
#  SECTION 1: IMPORTS
# -----------------------------------------------------------------------------

import logging
import pandas as pd
import itertools
from pathlib import Path
from datetime import datetime
from collections import defaultdict


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
        logging.FileHandler(DATA_CLEAN / "coauthorship.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#  SECTION 4: REGION MAP
#  Same map used in cleaning and geocoding — kept consistent
# -----------------------------------------------------------------------------

REGION_MAP = {
    "United States": "North America", "Canada": "North America",
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
    "Latvia": "Europe", "Lithuania": "Europe",
    "China": "Asia-Pacific", "India": "Asia-Pacific",
    "Japan": "Asia-Pacific", "South Korea": "Asia-Pacific",
    "Australia": "Asia-Pacific", "Singapore": "Asia-Pacific",
    "New Zealand": "Asia-Pacific", "Taiwan": "Asia-Pacific",
    "Hong Kong": "Asia-Pacific", "Malaysia": "Asia-Pacific",
    "Indonesia": "Asia-Pacific", "Thailand": "Asia-Pacific",
    "Vietnam": "Asia-Pacific", "Philippines": "Asia-Pacific",
    "Pakistan": "Asia-Pacific", "Bangladesh": "Asia-Pacific",
    "Sri Lanka": "Asia-Pacific", "Nepal": "Asia-Pacific",
    "Macau": "Asia-Pacific",
    "Brazil": "Latin America", "Mexico": "Latin America",
    "Argentina": "Latin America", "Colombia": "Latin America",
    "Chile": "Latin America", "Peru": "Latin America",
    "Venezuela": "Latin America", "Ecuador": "Latin America",
    "Bolivia": "Latin America", "Uruguay": "Latin America",
    "Costa Rica": "Latin America", "Cuba": "Latin America",
    "Panama": "Latin America", "Paraguay": "Latin America",
    "Guatemala": "Latin America", "Honduras": "Latin America",
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
}


# -----------------------------------------------------------------------------
#  SECTION 5: MAIN PIPELINE
# -----------------------------------------------------------------------------

def run_coauthorship_pipeline():

    log.info("=" * 60)
    log.info("  COAUTHORSHIP EDGES PIPELINE STARTING")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    # ── Load geocoded institution data ─────────────────────────────────────────
    # We use the geocoded institutions file as our lookup
    # and the cleaned Scopus data for the paper-institution relationships
    inst_file   = DATA_CLEAN / "scopus_institutions.csv"
    papers_file = DATA_CLEAN / "scopus_cleaned.csv"

    if not inst_file.exists():
        log.error("❌ scopus_institutions.csv not found. Run 2b_institution_geocoding_scopus.py first.")
        return
    if not papers_file.exists():
        log.error("❌ scopus_cleaned.csv not found. Run 2_data_cleaning_scopus.py first.")
        return

    log.info("📂 Loading institution data...")
    inst_df = pd.read_csv(inst_file, dtype=str, low_memory=False)

    # Convert numeric columns back to float
    inst_df["latitude"]    = pd.to_numeric(inst_df["latitude"],    errors="coerce")
    inst_df["longitude"]   = pd.to_numeric(inst_df["longitude"],   errors="coerce")
    inst_df["paper_count"] = pd.to_numeric(inst_df["paper_count"], errors="coerce")

    log.info(f"   Loaded {len(inst_df):,} institutions")

    log.info("📂 Loading cleaned papers...")
    papers_df = pd.read_csv(papers_file, dtype=str, low_memory=False)
    log.info(f"   Loaded {len(papers_df):,} papers")

    # ── Build institution lookup dictionary ────────────────────────────────────
    # Key: institution name (lowercased for matching)
    # Value: {country, lat, lon}
    log.info("")
    log.info("🔧 Building institution lookup...")

    inst_lookup = {}
    for _, row in inst_df.iterrows():
        name    = str(row.get("institution", "")).strip()
        country = str(row.get("country", "")).strip()
        lat     = row.get("latitude")
        lon     = row.get("longitude")

        if name and country and pd.notna(lat) and pd.notna(lon):
            inst_lookup[name.lower()] = {
                "institution": name,
                "country":     country,
                "lat":         float(lat),
                "lon":         float(lon),
            }

    log.info(f"   {len(inst_lookup):,} institutions with coordinates in lookup")

    # ── Build institution edges ────────────────────────────────────────────────
    log.info("")
    log.info("🔗 Building institution-level coauthorship edges...")
    log.info("   (Creating pairs of institutions per paper — this may take a few minutes)")

    # institution_edge_counts maps (inst_a, inst_b) → {paper_count, period info}
    institution_edges = defaultdict(lambda: {"paper_count": 0, "pre": 0, "post": 0})

    papers_with_multiple = 0
    papers_single        = 0
    papers_no_affil      = 0

    for i, (_, row) in enumerate(papers_df.iterrows()):

        affiliations_str = str(row.get("affiliations", "")) if pd.notna(row.get("affiliations")) else ""
        period           = str(row.get("period", ""))

        if not affiliations_str:
            papers_no_affil += 1
            continue

        # Split affiliations into list of institution names
        institutions = [a.strip() for a in affiliations_str.split(";") if a.strip()]

        # Deduplicate — same institution appearing twice on one paper = one node
        institutions = list(dict.fromkeys(institutions))

        if len(institutions) < 2:
            papers_single += 1
            continue

        papers_with_multiple += 1

        # Limit to first 10 institutions per paper to avoid combinatorial explosion
        # on papers with very large author lists (some have 50+ institutions)
        institutions = institutions[:10]

        # Generate all pairs
        for inst_a, inst_b in itertools.combinations(institutions, 2):

            # Look up both institutions — skip if either has no coordinates
            info_a = inst_lookup.get(inst_a.lower())
            info_b = inst_lookup.get(inst_b.lower())

            if not info_a or not info_b:
                continue

            # Skip self-loops (same institution appearing under two names)
            if info_a["country"] == info_b["country"] and inst_a.lower() == inst_b.lower():
                continue

            # Normalise key so (A,B) and (B,A) are the same edge
            key = tuple(sorted([inst_a, inst_b]))

            institution_edges[key]["paper_count"] += 1
            if "pre" in period:
                institution_edges[key]["pre"] += 1
            else:
                institution_edges[key]["post"] += 1

        # Progress update every 5000 papers
        if (i + 1) % 5000 == 0:
            log.info(f"   ... processed {i + 1:,} / {len(papers_df):,} papers")

    log.info(f"   Papers with multiple institutions:  {papers_with_multiple:,}")
    log.info(f"   Papers with single institution:     {papers_single:,}")
    log.info(f"   Papers with no affiliation data:    {papers_no_affil:,}")
    log.info(f"   Unique institution edges found:     {len(institution_edges):,}")

    # ── Build institution edge dataframe ──────────────────────────────────────
    log.info("")
    log.info("📊 Building institution edge dataframe...")

    inst_edge_rows = []
    for (inst_a, inst_b), counts in institution_edges.items():
        info_a = inst_lookup.get(inst_a.lower())
        info_b = inst_lookup.get(inst_b.lower())
        if not info_a or not info_b:
            continue

        inst_edge_rows.append({
            "institution_a":   info_a["institution"],
            "country_a":       info_a["country"],
            "region_a":        REGION_MAP.get(info_a["country"], "Other"),
            "lat_a":           info_a["lat"],
            "lon_a":           info_a["lon"],
            "institution_b":   info_b["institution"],
            "country_b":       info_b["country"],
            "region_b":        REGION_MAP.get(info_b["country"], "Other"),
            "lat_b":           info_b["lat"],
            "lon_b":           info_b["lon"],
            "paper_count":     counts["paper_count"],
            "pre_chatgpt":     counts["pre"],
            "post_chatgpt":    counts["post"],
            # Is this a cross-country collaboration?
            "cross_country":   info_a["country"] != info_b["country"],
            # Is this a cross-region collaboration?
            "cross_region":    REGION_MAP.get(info_a["country"], "Other") != REGION_MAP.get(info_b["country"], "Other"),
        })

    inst_edges_df = pd.DataFrame(inst_edge_rows)
    inst_edges_df = inst_edges_df.sort_values("paper_count", ascending=False).reset_index(drop=True)

    log.info(f"   Total institution edges:            {len(inst_edges_df):,}")
    log.info(f"   Cross-country edges:                {inst_edges_df['cross_country'].sum():,}")
    log.info(f"   Cross-region edges:                 {inst_edges_df['cross_region'].sum():,}")

    # Save institution edges
    inst_edge_path = DATA_CLEAN / "scopus_institution_edges.csv"
    inst_edges_df.to_csv(inst_edge_path, index=False, encoding="utf-8")
    log.info(f"   ✅ Saved: {inst_edge_path.name}")

    # ── Build COUNTRY-level edges ──────────────────────────────────────────────
    # Aggregate institution edges up to country level
    # This is the file that directly feeds your network map
    log.info("")
    log.info("🌍 Aggregating to country-level edges...")

    country_edges = defaultdict(lambda: {"paper_count": 0, "pre": 0, "post": 0})

    for _, row in inst_edges_df.iterrows():
        country_a = row["country_a"]
        country_b = row["country_b"]

        # Skip same-country edges — only keep international collaborations
        if country_a == country_b:
            continue

        # Normalise so (A,B) and (B,A) are the same edge
        key = tuple(sorted([country_a, country_b]))
        country_edges[key]["paper_count"] += int(row["paper_count"])
        country_edges[key]["pre"]         += int(row["pre_chatgpt"])
        country_edges[key]["post"]        += int(row["post_chatgpt"])

    # Build country edge dataframe
    # For coordinates we use the centroid of each country
    # (average lat/lon of all institutions in that country)
    log.info("   Computing country centroids from institution coordinates...")

    country_coords = (
        inst_df[inst_df["latitude"].notna()]
        .assign(
            latitude  = lambda d: pd.to_numeric(d["latitude"],  errors="coerce"),
            longitude = lambda d: pd.to_numeric(d["longitude"], errors="coerce"),
        )
        .groupby("country")
        .agg(
            lat = ("latitude",  "mean"),
            lon = ("longitude", "mean"),
        )
        .reset_index()
    )
    coords_lookup = {
        row["country"]: {"lat": row["lat"], "lon": row["lon"]}
        for _, row in country_coords.iterrows()
    }

    country_edge_rows = []
    for (country_a, country_b), counts in country_edges.items():
        coords_a = coords_lookup.get(country_a, {})
        coords_b = coords_lookup.get(country_b, {})

        country_edge_rows.append({
            "country_a":    country_a,
            "region_a":     REGION_MAP.get(country_a, "Other"),
            "lat_a":        coords_a.get("lat"),
            "lon_a":        coords_a.get("lon"),
            "country_b":    country_b,
            "region_b":     REGION_MAP.get(country_b, "Other"),
            "lat_b":        coords_b.get("lat"),
            "lon_b":        coords_b.get("lon"),
            "paper_count":  counts["paper_count"],
            "pre_chatgpt":  counts["pre"],
            "post_chatgpt": counts["post"],
            # Cross-region flag for your RO2 analysis
            "cross_region": REGION_MAP.get(country_a, "Other") != REGION_MAP.get(country_b, "Other"),
        })

    country_edges_df = pd.DataFrame(country_edge_rows)
    country_edges_df = country_edges_df.sort_values("paper_count", ascending=False).reset_index(drop=True)

    # Save country edges
    country_edge_path = DATA_CLEAN / "scopus_country_edges.csv"
    country_edges_df.to_csv(country_edge_path, index=False, encoding="utf-8")
    log.info(f"   ✅ Saved: {country_edge_path.name}")

    # ── Build COUNTRY nodes file ───────────────────────────────────────────────
    log.info("")
    log.info("📍 Building country nodes file...")

    # Total papers per country (from institutions file)
    country_papers = (
        inst_df
        .assign(paper_count = lambda d: pd.to_numeric(d["paper_count"], errors="coerce").fillna(0))
        .groupby("country")
        .agg(total_papers = ("paper_count", "sum"))
        .reset_index()
    )

    country_nodes_df = country_papers.merge(country_coords, on="country", how="left")
    country_nodes_df["region"] = country_nodes_df["country"].map(REGION_MAP).fillna("Other")
    country_nodes_df = country_nodes_df.sort_values("total_papers", ascending=False).reset_index(drop=True)

    node_path = DATA_CLEAN / "scopus_country_nodes.csv"
    country_nodes_df.to_csv(node_path, index=False, encoding="utf-8")
    log.info(f"   ✅ Saved: {node_path.name}")

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("  COAUTHORSHIP SUMMARY")
    log.info("=" * 60)
    log.info(f"  Institution edges:          {len(inst_edges_df):,}")
    log.info(f"  Country edges (intl only):  {len(country_edges_df):,}")
    log.info(f"  Cross-region edges:         {country_edges_df['cross_region'].sum():,}")
    log.info(f"  Countries in network:       {len(country_nodes_df):,}")
    log.info("")
    log.info("  Top 10 international country collaborations:")
    log.info("  " + "-" * 45)
    for _, row in country_edges_df.head(10).iterrows():
        log.info(f"    {row['country_a']:<20} ↔ {row['country_b']:<20} {int(row['paper_count']):>5,} papers")
    log.info("  " + "-" * 45)
    log.info("")
    log.info("  Cross-region collaboration highlights:")
    cross = country_edges_df[country_edges_df["cross_region"]].head(5)
    for _, row in cross.iterrows():
        log.info(f"    {row['region_a']:<20} ↔ {row['region_b']:<20} via {row['country_a']} — {row['country_b']}")
    log.info("")
    log.info("  Output files in data_clean/:")
    log.info("    scopus_institution_edges.csv  — institution pair network")
    log.info("    scopus_country_edges.csv      — country pair network (for maps)")
    log.info("    scopus_country_nodes.csv      — country nodes with coordinates")
    log.info("=" * 60)
    log.info("")
    log.info("🎉 Coauthorship edges pipeline complete!")
    log.info("")

    return inst_edges_df, country_edges_df, country_nodes_df


# -----------------------------------------------------------------------------
#  ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_coauthorship_pipeline()