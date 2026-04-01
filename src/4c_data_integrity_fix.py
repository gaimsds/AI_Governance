# =============================================================================
#  4c_data_integrity_fix.py
#  Project: Mapping Global AI Governance Narratives
#  Authors: Tambudzai Gundani & Joshua Gray
#  Date:    March 2026
# =============================================================================
#
#  WHAT THIS SCRIPT DOES:
#  ----------------------
#  Fixes data integrity issues identified after running 4_model_finetuning.py
#  and 4b_update_governance_scores.py. Must be run AFTER both of those.
#
#  DEPENDENCIES (must exist before running):
#  ------------------------------------------
#  data_clean/scopus_cleaned.csv          — columns include: scopus_id,
#                                           primary_country, cover_date
#  data_clean/policy_corpus.csv           — columns include: doc_id, text_clean
#  results/scopus_topics_finetuned.csv    — from 4_model_finetuning.py
#  results/governance_papers.csv          — from 4b_update_governance_scores.py
#  results/topics_finetuned.csv           — from 4b_update_governance_scores.py
#  results/policy_topic_assignments.csv   — from 3_modelling.py
#
#  FIXES APPLIED:
#  --------------
#  Fix 1 — primary_country + cover_date merged from scopus_cleaned.csv
#           The modelling script looked for 'country'/'coverDate' (wrong names)
#           Correct column names confirmed: primary_country, cover_date
#
#  Fix 2 — Governance scores propagated into scopus_topics_finetuned.csv
#           4b updated governance_papers.csv but not the full scopus file
#
#  Fix 3 — Policy chunks re-assigned via keyword overlap matching
#           898 chunks were stuck in Topic 0 from Round 1
#           641 were unassigned (-1)
#           This fix matches each document's text against governance topic
#           keywords. Threshold: 2 minimum keyword matches.
#           Non-zero original assignments kept via direct mapping.
#
#  Fix 4 — cross_corpus_alignment_v2.csv rebuilt with 133 finetuned topics
#
#  Fix 5 — policy_document_topics_v2.csv rebuilt with finetuned topic IDs
#
#  OUTPUTS saved to results/:
#  --------------------------
#  scopus_topics_finetuned.csv        — country + cover_date + governance scores
#  governance_papers.csv              — country + cover_date added
#  policy_topic_assignments_v2.csv    — policy chunks with finetuned topic IDs
#  policy_document_topics_v2.csv      — per-document topic summary (finetuned)
#  cross_corpus_alignment_v2.csv      — shared vs academic-only (133 topics)
#  integrity_report.txt               — summary of all fixes applied
#
#  PIPELINE ORDER (for full reproducibility):
#  -------------------------------------------
#  1_data_collection_scopus.py
#  1_data_collection_policyframeworks.py
#  2_data_cleaning_scopus.py
#  2b_institution_geocoding_scopus.py
#  2c_coauthorship_edges_scopus.py
#  2_policy_text_extraction.py
#  3_modelling.py
#  4_model_finetuning.py
#  4b_update_governance_scores.py
#  4c_data_integrity_fix.py          ← THIS SCRIPT
#  4d_sentiment_analysis.py          ← NEXT
# =============================================================================


# -----------------------------------------------------------------------------
#  IMPORTS
# -----------------------------------------------------------------------------

import logging
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
#  PATHS
# -----------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent
DATA_CLEAN  = BASE_DIR / "data_clean"
RESULTS_DIR = BASE_DIR / "results"


# -----------------------------------------------------------------------------
#  LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "integrity_fix.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
#  FIX 1 + FIX 2
#  primary_country + cover_date + governance scores
# =============================================================================

def fix_country_and_scores():
    """
    Fix 1: Merges primary_country and cover_date from scopus_cleaned.csv
           into scopus_topics_finetuned.csv and governance_papers.csv.
           These columns were 'Unknown' because the modelling script used
           wrong column names ('country', 'coverDate') instead of the
           correct ones ('primary_country', 'cover_date').

    Fix 2: Propagates updated governance scores and topic labels from
           topics_finetuned.csv into scopus_topics_finetuned.csv.
           4b_update_governance_scores.py updated governance_papers.csv
           correctly but did not save updated scores to the full file.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  FIX 1 + FIX 2 — COUNTRY + DATE + GOVERNANCE SCORES")
    log.info("=" * 60)

    # Load scopus_cleaned.csv with confirmed column names
    log.info("  Loading scopus_cleaned.csv...")
    cleaned = pd.read_csv(
        DATA_CLEAN / "scopus_cleaned.csv",
        usecols=["scopus_id", "primary_country", "cover_date", "all_countries"],
        dtype=str,
        low_memory=False,
    )
    log.info(f"  Rows loaded: {len(cleaned):,}")
    log.info(f"  Unique countries: {cleaned['primary_country'].nunique()}")
    log.info(f"  Top 8 countries in full corpus:")
    for c, n in cleaned["primary_country"].value_counts().head(8).items():
        log.info(f"    {c:<30} {n:,}")

    # Governance score and label maps
    topics    = pd.read_csv(RESULTS_DIR / "topics_finetuned.csv")
    score_map = dict(zip(topics["topic_id"].astype(str), topics["governance_score"]))
    label_map = dict(zip(topics["topic_id"].astype(str), topics["topic_label"]))

    # ── Update scopus_topics_finetuned.csv ──────────────────────────────────
    log.info("")
    log.info("  Updating scopus_topics_finetuned.csv...")
    ft = pd.read_csv(RESULTS_DIR / "scopus_topics_finetuned.csv", dtype=str)

    # Remove stale country/date columns before merge
    for col in ["country", "cover_date", "coverDate"]:
        if col in ft.columns:
            ft.drop(columns=[col], inplace=True)

    ft = ft.merge(cleaned, on="scopus_id", how="left")

    # Handle potential _x/_y suffixes from merge
    if "cover_date_y" in ft.columns:
        ft["cover_date"] = ft["cover_date_y"].fillna("Unknown")
        ft.drop(columns=[c for c in ["cover_date_x","cover_date_y"] if c in ft.columns],
                inplace=True)
    else:
        ft["cover_date"] = ft.get("cover_date", pd.Series("Unknown", index=ft.index))
        ft["cover_date"] = ft["cover_date"].fillna("Unknown")

    ft["country"] = ft["primary_country"].fillna("Unknown")

    # Clean up merge artefacts
    for col in ["primary_country", "all_countries"]:
        if col in ft.columns:
            ft.drop(columns=[col], inplace=True)

    # Fix 2: Propagate governance scores
    ft["governance_score"]      = ft["topic_id_finetuned"].map(score_map)
    ft["topic_label_finetuned"] = ft["topic_id_finetuned"].map(label_map)
    ft["governance_score"]      = pd.to_numeric(ft["governance_score"], errors="coerce")

    n_country = (ft["country"] != "Unknown").sum()
    n_gov     = (ft["governance_score"] >= 0.5).sum()
    log.info(f"  Papers with real country:    {n_country:,} / {len(ft):,}")
    log.info(f"  Papers with score >= 0.5:    {n_gov:,} (expected: 3,186)")

    ft.to_csv(RESULTS_DIR / "scopus_topics_finetuned.csv", index=False, encoding="utf-8")
    log.info("  ✅ scopus_topics_finetuned.csv saved")

    # ── Update governance_papers.csv ────────────────────────────────────────
    log.info("")
    log.info("  Updating governance_papers.csv...")
    gov = pd.read_csv(RESULTS_DIR / "governance_papers.csv", dtype=str)

    for col in ["country", "cover_date", "coverDate"]:
        if col in gov.columns:
            gov.drop(columns=[col], inplace=True)

    gov = gov.merge(cleaned, on="scopus_id", how="left")

    if "cover_date_y" in gov.columns:
        gov["cover_date"] = gov["cover_date_y"].fillna("Unknown")
        gov.drop(columns=[c for c in ["cover_date_x","cover_date_y"] if c in gov.columns],
                 inplace=True)
    else:
        gov["cover_date"] = gov.get("cover_date", pd.Series("Unknown", index=gov.index))
        gov["cover_date"] = gov["cover_date"].fillna("Unknown")

    gov["country"] = gov["primary_country"].fillna("Unknown")
    for col in ["primary_country", "all_countries"]:
        if col in gov.columns:
            gov.drop(columns=[col], inplace=True)

    gov["governance_score"]      = gov["topic_id_finetuned"].map(score_map)
    gov["topic_label_finetuned"] = gov["topic_id_finetuned"].map(label_map)

    n_gov_country = (gov["country"] != "Unknown").sum()
    log.info(f"  Governance papers with real country: {n_gov_country:,} / {len(gov):,}")
    log.info(f"  Top 8 countries in governance corpus:")
    for c, n in gov["country"].value_counts().head(8).items():
        log.info(f"    {c:<30} {n:,}")

    gov.to_csv(RESULTS_DIR / "governance_papers.csv", index=False, encoding="utf-8")
    log.info("  ✅ governance_papers.csv saved")

    return ft, gov


# =============================================================================
#  FIX 3
#  Re-assign policy chunks stuck in Topic 0 and -1
# =============================================================================

def fix_policy_assignments():
    """
    898 policy chunks were assigned to Topic 0 (catch-all) in Round 1.
    641 chunks were unassigned (-1).

    This function re-assigns them by:
    1. Loading the full document text from policy_corpus.csv
    2. For each Topic 0 or -1 chunk, finding which governance topic's
       keywords overlap most with that document's text
    3. Assigning the chunk to that topic if overlap >= 2 keywords

    Doc IDs confirmed format: USA_National_AI_RD_Strategic_Plan_2023
    (filename without .pdf extension, matching policy_corpus.csv doc_id)

    Non-zero original assignments (T2, T15, T31 etc.) are kept as-is
    via direct mapping — they were correctly assigned in Round 1.

    Limitation: This is keyword-based matching, not embedding-based.
    It is an approximation documented in the methodology section.
    Chunks remaining as -1 after this fix genuinely do not match any
    recognised governance theme in the finetuned topic vocabulary.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  FIX 3 — POLICY CHUNK RE-ASSIGNMENT")
    log.info("=" * 60)

    # Load policy corpus
    corpus   = pd.read_csv(DATA_CLEAN / "policy_corpus.csv", dtype=str)
    text_col = next(
        (c for c in ["text_clean", "text_raw", "text"] if c in corpus.columns),
        None
    )
    log.info(f"  Policy corpus columns: {corpus.columns.tolist()}")
    log.info(f"  Rows: {len(corpus)}  |  Text column: {text_col}")
    log.info(f"  Sample doc_ids: {corpus['doc_id'].head(3).tolist()}")

    if text_col is None:
        log.error("  ❌ No text column in policy_corpus.csv — Fix 3 cannot proceed")
        return None

    # Build doc_id → full text lookup
    doc_texts = {
        str(row["doc_id"]).strip(): str(row.get(text_col, ""))
        for _, row in corpus.iterrows()
    }
    log.info(f"  Documents in lookup: {len(doc_texts)}")

    # Load governance topic keywords
    topics    = pd.read_csv(RESULTS_DIR / "topics_finetuned.csv")
    label_map = dict(zip(topics["topic_id"], topics["topic_label"]))
    score_map = dict(zip(topics["topic_id"], topics["governance_score"]))

    gov_topics = topics[topics["governance_score"] >= 0.5].copy()
    gov_kw = {}
    for _, row in gov_topics.iterrows():
        tid    = int(row["topic_id"])
        words  = str(row.get("top_words", "")).lower()
        kw_set = {w.strip() for w in words.split(",") if len(w.strip()) > 2}
        gov_kw[tid] = kw_set

    log.info(f"  Governance topics for matching: {len(gov_kw)}")

    # Load Round 1 policy assignments
    policy = pd.read_csv(RESULTS_DIR / "policy_topic_assignments.csv", dtype=str)
    policy["topic_id"] = pd.to_numeric(
        policy["topic_id"], errors="coerce"
    ).fillna(-1).astype(int)

    log.info(f"  Total chunks: {len(policy):,}")
    log.info(f"  Topic  0: {(policy['topic_id']==0).sum():,}")
    log.info(f"  Topic -1: {(policy['topic_id']==-1).sum():,}")
    log.info(f"  Other:    {(policy['topic_id']>0).sum():,}")

    # Direct map: non-zero topics from Round 1 are already correct
    direct_map = {
        2: 2,  4: 4,  8: 8,  9: 9,  12: 12, 15: 15,
        17: 17, 21: 21, 24: 24, 25: 25, 31: 31,
        34: 34, 35: 35, 42: 42, -1: -1
    }

    def find_best_governance_topic(doc_id, current_topic_id):
        """Return best matching governance topic ID for this chunk."""
        if current_topic_id not in [0, -1]:
            return direct_map.get(current_topic_id, current_topic_id)

        text = doc_texts.get(str(doc_id), "")
        if not text or len(text) < 100:
            return -1

        doc_words  = set(text.lower().split())
        best_tid   = -1
        best_score = 0
        for tid, kw_set in gov_kw.items():
            overlap = len(doc_words & kw_set)
            if overlap > best_score:
                best_score = overlap
                best_tid   = tid

        # Require minimum 2 keyword matches
        return best_tid if best_score >= 2 else -1

    log.info("  Running keyword-based reassignment...")
    policy = policy.copy()
    policy["topic_id_finetuned"] = policy.apply(
        lambda r: find_best_governance_topic(r["doc_id"], int(r["topic_id"])),
        axis=1
    )

    policy["topic_id_finetuned"]    = pd.to_numeric(
        policy["topic_id_finetuned"], errors="coerce"
    ).fillna(-1).astype(int)
    policy["topic_label_finetuned"] = policy["topic_id_finetuned"].map(label_map)
    policy["governance_score"]      = policy["topic_id_finetuned"].map(score_map)

    # Summary
    reassigned_t0  = (
        (policy["topic_id"] == 0) & (policy["topic_id_finetuned"] != -1)
    ).sum()
    reassigned_neg = (
        (policy["topic_id"] == -1) & (policy["topic_id_finetuned"] != -1)
    ).sum()
    gov_chunks   = (
        pd.to_numeric(policy["governance_score"], errors="coerce") >= 0.5
    ).sum()
    still_minus1 = (policy["topic_id_finetuned"] == -1).sum()

    log.info(f"  Topic 0 reassigned:        {reassigned_t0:,}")
    log.info(f"  Topic -1 reassigned:       {reassigned_neg:,}")
    log.info(f"  Governance chunks (>=0.5): {gov_chunks:,}")
    log.info(f"  Still unassigned (-1):     {still_minus1:,}")
    log.info("")
    log.info("  Assigned chunk distribution:")
    dist = (
        policy[policy["topic_id_finetuned"] != -1]
        .groupby(["topic_id_finetuned", "topic_label_finetuned"])
        .size()
        .reset_index(name="chunks")
        .sort_values("chunks", ascending=False)
    )
    for _, r in dist.iterrows():
        log.info(
            f"    [{int(r['topic_id_finetuned']):3d}]  "
            f"{r['chunks']:4d} chunks  "
            f"{str(r['topic_label_finetuned'])[:50]}"
        )

    policy.to_csv(
        RESULTS_DIR / "policy_topic_assignments_v2.csv",
        index=False, encoding="utf-8"
    )
    log.info("  ✅ policy_topic_assignments_v2.csv saved")
    return policy


# =============================================================================
#  FIX 4
#  Rebuild cross-corpus alignment with 133 finetuned topics
# =============================================================================

def rebuild_cross_corpus_alignment(policy_v2: pd.DataFrame):
    """
    Rebuilds cross_corpus_alignment_v2.csv.
    Original cross_corpus_alignment.csv used Round 1 (46 topics).
    This version uses all 133 finetuned topics.

    Alignment categories:
      shared        — topic in both Scopus governance papers AND policy chunks
      academic_only — topic in Scopus but absent from policy corpus
      policy_only   — topic in policy corpus but absent from Scopus
      neither       — topic has no papers in either (rare)
    """
    log.info("")
    log.info("=" * 60)
    log.info("  FIX 4 — CROSS-CORPUS ALIGNMENT")
    log.info("=" * 60)

    topics     = pd.read_csv(RESULTS_DIR / "topics_finetuned.csv")
    gov_papers = pd.read_csv(RESULTS_DIR / "governance_papers.csv", dtype=str)

    gov_papers["topic_id_finetuned"] = pd.to_numeric(
        gov_papers["topic_id_finetuned"], errors="coerce"
    ).fillna(-1).astype(int)
    policy_v2["topic_id_finetuned"]  = pd.to_numeric(
        policy_v2["topic_id_finetuned"], errors="coerce"
    ).fillna(-1).astype(int)

    scopus_counts = (
        gov_papers[gov_papers["topic_id_finetuned"] != -1]
        ["topic_id_finetuned"].value_counts().reset_index()
    )
    scopus_counts.columns = ["topic_id", "scopus_count"]

    pol_gov = policy_v2[
        pd.to_numeric(policy_v2["governance_score"], errors="coerce") >= 0.5
    ]
    policy_counts = (
        pol_gov[pol_gov["topic_id_finetuned"] != -1]
        ["topic_id_finetuned"].value_counts().reset_index()
    )
    policy_counts.columns = ["topic_id", "policy_count"]

    alignment = topics.merge(scopus_counts, on="topic_id", how="left")
    alignment = alignment.merge(policy_counts, on="topic_id", how="left")
    alignment["scopus_count"] = alignment["scopus_count"].fillna(0).astype(int)
    alignment["policy_count"] = alignment["policy_count"].fillna(0).astype(int)
    alignment["in_scopus"]    = alignment["scopus_count"] > 0
    alignment["in_policy"]    = alignment["policy_count"] > 0
    alignment["alignment"]    = alignment.apply(
        lambda r:
            "shared"        if r["in_scopus"] and r["in_policy"]
            else "academic_only" if r["in_scopus"]
            else "policy_only"   if r["in_policy"]
            else "neither",
        axis=1
    )
    alignment.sort_values("scopus_count", ascending=False, inplace=True)
    alignment.to_csv(
        RESULTS_DIR / "cross_corpus_alignment_v2.csv",
        index=False, encoding="utf-8"
    )

    gov_align = alignment[alignment["governance_score"] >= 0.5]
    shared    = (gov_align["alignment"] == "shared").sum()
    acad_only = (gov_align["alignment"] == "academic_only").sum()
    pol_only  = (gov_align["alignment"] == "policy_only").sum()

    log.info(f"  Shared governance topics:        {shared}")
    log.info(f"  Academic-only governance topics: {acad_only}")
    log.info(f"  Policy-only governance topics:   {pol_only}")
    log.info("")
    log.info("  Shared governance topics:")
    for _, r in gov_align[gov_align["alignment"] == "shared"].iterrows():
        log.info(f"    [{int(r['topic_id']):3d}]  scopus={r['scopus_count']:,}  "
                 f"policy={r['policy_count']}  {r['topic_label'][:50]}")
    log.info("")
    log.info("  Academic-only governance topics:")
    for _, r in gov_align[gov_align["alignment"] == "academic_only"].iterrows():
        log.info(f"    [{int(r['topic_id']):3d}]  scopus={r['scopus_count']:,}  "
                 f"{r['topic_label'][:50]}")

    log.info("  ✅ cross_corpus_alignment_v2.csv saved")
    return alignment


# =============================================================================
#  FIX 5
#  Rebuild policy document topics with finetuned IDs
# =============================================================================

def rebuild_policy_document_topics(policy_v2: pd.DataFrame):
    """
    Rebuilds policy_document_topics_v2.csv.
    Original used Round 1 topic IDs where 898 chunks were Topic 0.
    This version uses finetuned topic IDs from Fix 3.

    For each of the 35 policy documents, shows which governance topics
    appear in its chunks and what proportion of chunks discuss each topic.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  FIX 5 — POLICY DOCUMENT TOPICS")
    log.info("=" * 60)

    topics    = pd.read_csv(RESULTS_DIR / "topics_finetuned.csv")
    label_map = dict(zip(topics["topic_id"], topics["topic_label"]))

    policy_v2["topic_id_finetuned"] = pd.to_numeric(
        policy_v2["topic_id_finetuned"], errors="coerce"
    ).fillna(-1).astype(int)

    assigned   = policy_v2[policy_v2["topic_id_finetuned"] != -1]
    doc_topics = (
        assigned
        .groupby(["doc_id", "country", "region", "doc_type", "topic_id_finetuned"])
        .size()
        .reset_index(name="chunk_count")
    )
    doc_totals = policy_v2.groupby("doc_id").size().reset_index(name="total_chunks")
    doc_topics = doc_topics.merge(doc_totals, on="doc_id")
    doc_topics["proportion"] = (
        doc_topics["chunk_count"] / doc_topics["total_chunks"]
    ).round(4)
    doc_topics["topic_label_finetuned"] = doc_topics["topic_id_finetuned"].map(label_map)
    doc_topics.sort_values(["country", "chunk_count"], ascending=[True, False], inplace=True)

    doc_topics.to_csv(
        RESULTS_DIR / "policy_document_topics_v2.csv",
        index=False, encoding="utf-8"
    )

    log.info(f"  Documents with topic assignments: {doc_topics['doc_id'].nunique()} / 35")
    log.info(f"  Total topic-document rows:        {len(doc_topics)}")
    log.info("")
    log.info("  Per-document breakdown (top 25 by chunk count):")
    for _, r in doc_topics.sort_values("chunk_count", ascending=False).head(25).iterrows():
        log.info(
            f"    {str(r['country']):<22}  [{int(r['topic_id_finetuned']):3d}]  "
            f"{r['chunk_count']:3d} chunks ({r['proportion']:.0%})  "
            f"{str(r['topic_label_finetuned'])[:40]}"
        )

    log.info("  ✅ policy_document_topics_v2.csv saved")


# =============================================================================
#  WRITE INTEGRITY REPORT
# =============================================================================

def write_integrity_report(ft, gov, policy_v2, alignment):
    """Writes results/integrity_report.txt summarising all fixes."""
    log.info("")
    log.info("  Writing integrity_report.txt...")

    n_country_ft  = (ft["country"] != "Unknown").sum() if ft is not None else 0
    n_gov_country = (gov["country"] != "Unknown").sum() if gov is not None else 0
    n_gov_score   = int(
        (pd.to_numeric(ft["governance_score"], errors="coerce") >= 0.5).sum()
    ) if ft is not None else 0

    n_reassigned = n_gov_chunks = n_still_minus1 = 0
    if policy_v2 is not None:
        policy_v2["topic_id_finetuned"] = pd.to_numeric(
            policy_v2["topic_id_finetuned"], errors="coerce"
        ).fillna(-1).astype(int)
        n_reassigned  = (
            (policy_v2["topic_id"].astype(str).isin(["0","-1"])) &
            (policy_v2["topic_id_finetuned"] != -1)
        ).sum()
        n_gov_chunks  = int(
            (pd.to_numeric(policy_v2["governance_score"], errors="coerce") >= 0.5).sum()
        )
        n_still_minus1 = int((policy_v2["topic_id_finetuned"] == -1).sum())

    shared = acad_only = pol_only = 0
    if alignment is not None:
        g         = alignment[alignment["governance_score"] >= 0.5]
        shared    = int((g["alignment"] == "shared").sum())
        acad_only = int((g["alignment"] == "academic_only").sum())
        pol_only  = int((g["alignment"] == "policy_only").sum())

    lines = [
        "DATA INTEGRITY FIX REPORT — 4c_data_integrity_fix.py",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "FIX 1 — COUNTRY + COVER_DATE",
        f"  scopus_topics_finetuned papers with real country: {n_country_ft:,}",
        f"  governance_papers with real country:              {n_gov_country:,}",
        f"  Source: primary_country, cover_date from scopus_cleaned.csv",
        f"  primary_country = first author affiliation country",
        f"  (standard bibliometric convention, documented in methodology)",
        "",
        "FIX 2 — GOVERNANCE SCORES PROPAGATED",
        f"  Papers with governance_score >= 0.5: {n_gov_score:,}",
        f"  Score source: topics_finetuned.csv (updated by 4b)",
        "",
        "FIX 3 — POLICY CHUNK RE-ASSIGNMENT",
        f"  Chunks reassigned from Topic 0/-1: {n_reassigned:,}",
        f"  Governance policy chunks (>=0.5):  {n_gov_chunks:,}",
        f"  Chunks still unassigned (-1):      {n_still_minus1:,}",
        f"  Method: keyword overlap matching (min threshold=2)",
        f"  Limitation: keyword-based approximation, not embedding-based.",
        f"  Remaining -1 chunks reflect genuine vocabulary gap between",
        f"  academic abstracts and policy document language.",
        f"  Documented in methodology as a known limitation.",
        "",
        "FIX 4 — CROSS-CORPUS ALIGNMENT REBUILT",
        f"  Total finetuned topics: 133 (vs 46 in Round 1)",
        f"  Shared governance topics:        {shared}",
        f"  Academic-only governance topics: {acad_only}",
        f"  Policy-only governance topics:   {pol_only}",
        "",
        "FIX 5 — POLICY DOCUMENT TOPICS REBUILT",
        f"  Saved: policy_document_topics_v2.csv",
        "",
        "OUTPUT FILES:",
        "  scopus_topics_finetuned.csv      — country + cover_date + scores FIXED",
        "  governance_papers.csv            — country + cover_date FIXED",
        "  policy_topic_assignments_v2.csv  — finetuned topic IDs",
        "  policy_document_topics_v2.csv    — per-document finetuned topics",
        "  cross_corpus_alignment_v2.csv    — rebuilt with 133 topics",
        "  integrity_report.txt             — this file",
        "",
        "FULL PIPELINE EXECUTION ORDER:",
        "  1_data_collection_scopus.py",
        "  1_data_collection_policyframeworks.py",
        "  2_data_cleaning_scopus.py",
        "  2b_institution_geocoding_scopus.py",
        "  2c_coauthorship_edges_scopus.py",
        "  2_policy_text_extraction.py",
        "  3_modelling.py",
        "  4_model_finetuning.py",
        "  4b_update_governance_scores.py",
        "  4c_data_integrity_fix.py   ← THIS SCRIPT",
        "  4d_sentiment_analysis.py   ← NEXT",
    ]

    with open(RESULTS_DIR / "integrity_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("  ✅ integrity_report.txt saved")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("  DATA INTEGRITY FIX PIPELINE — 4c")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    ft = gov = policy_v2 = alignment = None

    # Fix 1 + 2
    ft, gov = fix_country_and_scores()

    # Fix 3
    policy_v2 = fix_policy_assignments()
    if policy_v2 is None:
        log.error("Fix 3 failed — check data_clean/policy_corpus.csv")
        return

    # Fix 4
    alignment = rebuild_cross_corpus_alignment(policy_v2)

    # Fix 5
    rebuild_policy_document_topics(policy_v2)

    # Report
    write_integrity_report(ft, gov, policy_v2, alignment)

    log.info("")
    log.info("=" * 60)
    log.info("  4c COMPLETE")
    log.info("=" * 60)
    log.info("  scopus_topics_finetuned.csv      ✅")
    log.info("  governance_papers.csv            ✅")
    log.info("  policy_topic_assignments_v2.csv  ✅")
    log.info("  cross_corpus_alignment_v2.csv    ✅")
    log.info("  policy_document_topics_v2.csv    ✅")
    log.info("  integrity_report.txt             ✅")
    log.info("")
    log.info("  Next: commit to Git, then build 4d_sentiment_analysis.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()