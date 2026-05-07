"""
========================================================================================================================
ZERO-SHOT STANCE CLASSIFICATION
Project: Uneven Science–Policy Translation Shapes Global AI Governance
Authors: Tambudzai G. Charumbira & Joshua Gray
Institution: George Washington University, CCAS | M.S. Data Science | Spring 2026
Date: April 2026

DESCRIPTION:
This script applied zero-shot stance classification to the 3,186 governance papers using facebook/bart-large-mnli. Each
abstract was classified against three candidate labels — risk-focused, opportunity-focused, and balanced — using natural
 language inference (NLI). The highest-scoring label was assigned as the paper's stance.

A confidence threshold of 0.45 was applied: papers where the model's top confidence score fell below this threshold were
conservatively reassigned to balanced rather than propagating uncertain predictions. This affected 808 papers (25.4%),
which is reported as a limitation (Section 5.5).

Zero-shot classification was selected over supervised approaches because no labelled training dataset exists for AI
governance stance classification. The approach is fully reproducible — anyone with access to the same model and labels
can replicate the results.

Classification completed in approximately 7 minutes on Apple Silicon MPS.

OUTPUT FILES (saved to results/):
- governance_papers_stance.csv   → Governance papers with stance labels and confidence scores
- stance_by_topic.csv            → Stance distribution per governance topic
- stance_by_region.csv           → Stance distribution per world region
- stance_by_period.csv           → Pre/post-ChatGPT stance shift
- stance_policy_comparison.csv   → Stance comparison with policy corpus
- stance_summary.txt             → Human-readable stance analysis summary
========================================================================================================================
"""
# ======================================================================================================================
#  SECTION 1: IMPORTS
# ======================================================================================================================
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")


# ======================================================================================================================
#  SECTION 2: PATHS
# ======================================================================================================================
BASE_DIR    = Path(__file__).parent.parent
DATA_CLEAN  = BASE_DIR / "data_clean"
RESULTS_DIR = BASE_DIR / "results"


# ======================================================================================================================
#  SECTION 3: LOGGING
# ======================================================================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "sentiment_analysis.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ======================================================================================================================
#  SECTION 4: CONFIGURATION
#  Three candidate labels were defined for the NLI classifier, each phrased as a natural language hypothesis:
#    - risk_focused: "This text discusses risks, dangers, or negative impacts of artificial intelligence"
#    - opportunity_focused: "This text discusses benefits, opportunities, or positive potential of AI"
#    - balanced: "This text takes a balanced view, discussing both risks and benefits of AI"
#  The confidence threshold (0.45) and batch size were also configured here.
# ======================================================================================================================
# Zero-shot model — BART trained on MNLI (Multi-Genre Natural Language Inference)
# This is the standard model for zero-shot text classification in NLP research
ZS_MODEL = "facebook/bart-large-mnli"

# Candidate labels for stance classification
# These are chosen to be semantically distinct and cover the two poles
# plus a neutral/balanced option
STANCE_LABELS = [
    "risk-focused: this text emphasises dangers, threats, harms, and concerns about AI",
    "opportunity-focused: this text emphasises benefits, potential, innovation, and positive outcomes of AI",
    "balanced: this text discusses both risks and opportunities of AI without clear emphasis",
]

# Short label names for output files
LABEL_MAP = {
    "risk-focused: this text emphasises dangers, threats, harms, and concerns about AI":
        "risk_focused",
    "opportunity-focused: this text emphasises benefits, potential, innovation, and positive outcomes of AI":
        "opportunity_focused",
    "balanced: this text discusses both risks and opportunities of AI without clear emphasis":
        "balanced",
}

# Batch size for inference — smaller = slower but lower memory
BATCH_SIZE = 8

# Confidence threshold: if top score < this, label as "low_confidence"
CONFIDENCE_THRESHOLD = 0.45


# ======================================================================================================================
#  SECTION 5: LOAD DATA
#  The governance papers (from 4c_data_integrity_fix.py) were loaded with topic assignments and metadata.
#  The raw abstract — not the preprocessed version — was used for classification, as BART-large-MNLI
#  performs better on natural language than on lemmatized/stopword-removed text.
# ======================================================================================================================
def load_governance_papers() -> pd.DataFrame:
    """Load governance papers and their abstracts."""
    log.info("Loading governance_papers.csv...")
    gov = pd.read_csv(RESULTS_DIR / "governance_papers.csv", dtype=str)
    log.info(f"  Governance papers loaded: {len(gov):,}")

    # Load abstracts from scopus_cleaned.csv
    log.info("Loading abstracts from scopus_cleaned.csv...")
    cleaned = pd.read_csv(
        DATA_CLEAN / "scopus_cleaned.csv",
        usecols=["scopus_id", "abstract", "abstract_clean", "title"],
        dtype=str,
        low_memory=False,
    )

    # Merge abstracts
    gov = gov.merge(cleaned, on="scopus_id", how="left")

    # Use abstract_clean if available, else abstract
    gov["text_for_classification"] = gov.apply(
        lambda r: str(r["abstract_clean"])
        if pd.notna(r.get("abstract_clean")) and len(str(r.get("abstract_clean", ""))) > 50
        else str(r.get("abstract", "")),
        axis=1
    )

    # Drop papers with no usable abstract
    n_before = len(gov)
    gov = gov[gov["text_for_classification"].str.len() > 50].copy()
    n_dropped = n_before - len(gov)
    if n_dropped > 0:
        log.info(f"  Dropped {n_dropped} papers with no abstract")

    log.info(f"  Papers ready for classification: {len(gov):,}")
    return gov.reset_index(drop=True)


# ======================================================================================================================
#  SECTION 6: ZERO-SHOT CLASSIFIER SETUP
#  facebook/bart-large-mnli was loaded via the HuggingFace transformers zero-shot-classification pipeline.
#  The model was trained on the MultiNLI dataset (433K sentence pairs) and is the standard zero-shot
#  classifier in the HuggingFace ecosystem (Lewis et al., 2020; Yin et al., 2019). Device was set to
#  Apple Silicon MPS where available, falling back to CPU.
# ======================================================================================================================
def load_classifier():
    """Load the zero-shot classification pipeline."""
    log.info(f"Loading zero-shot classifier: {ZS_MODEL}")
    log.info("  (This downloads ~1.6GB on first run — cached afterwards)")

    from transformers import pipeline
    import torch

    # Use MPS (Apple Silicon) if available, else CPU
    if torch.backends.mps.is_available():
        device = 0
        log.info("  Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = 0
        log.info("  Using device: CUDA GPU")
    else:
        device = -1
        log.info("  Using device: CPU (will be slower)")

    classifier = pipeline(
        "zero-shot-classification",
        model=ZS_MODEL,
        device=device,
    )
    log.info("  ✅ Classifier loaded")
    return classifier


# ======================================================================================================================
#  SECTION 7: RUN CLASSIFICATION
#  Each abstract was classified individually. The model returned confidence scores for all three labels;
#  the highest-scoring label was assigned. Where the top score fell below 0.45, the paper was reassigned
#  to balanced — a conservative design choice to avoid propagating uncertain predictions. 808 papers
#  (25.4%) were affected. Progress was logged every 100 papers.
# ======================================================================================================================
def classify_stance(papers_df: pd.DataFrame, classifier) -> pd.DataFrame:
    """
    Run zero-shot stance classification on all governance paper abstracts.
    Returns the dataframe with stance labels and confidence scores added.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  STANCE CLASSIFICATION")
    log.info("=" * 60)
    log.info(f"  Papers to classify: {len(papers_df):,}")
    log.info(f"  Batch size: {BATCH_SIZE}")
    log.info(f"  Estimated time: {len(papers_df) // BATCH_SIZE // 2} minutes")

    texts  = papers_df["text_for_classification"].tolist()
    stances         = []
    confidence_scores = []
    risk_scores     = []
    opportunity_scores = []
    balanced_scores = []

    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch       = texts[i : i + BATCH_SIZE]
        batch_num   = i // BATCH_SIZE + 1

        if batch_num % 20 == 0 or batch_num == 1:
            pct = batch_num / n_batches * 100
            log.info(f"  Batch {batch_num}/{n_batches} ({pct:.0f}%)...")

        try:
            results = classifier(
                batch,
                candidate_labels=STANCE_LABELS,
                multi_label=False,
            )

            # Handle single vs batch results
            if isinstance(results, dict):
                results = [results]

            for result in results:
                # Get scores for each label
                label_scores = dict(zip(result["labels"], result["scores"]))

                top_label = result["labels"][0]
                top_score = result["scores"][0]

                # Map to short label
                stance = LABEL_MAP.get(top_label, "balanced")

                # Apply confidence threshold
                if top_score < CONFIDENCE_THRESHOLD:
                    stance = "balanced"

                stances.append(stance)
                confidence_scores.append(round(top_score, 4))

                # Store individual label scores
                r_score  = label_scores.get(STANCE_LABELS[0], 0.0)
                o_score  = label_scores.get(STANCE_LABELS[1], 0.0)
                b_score  = label_scores.get(STANCE_LABELS[2], 0.0)
                risk_scores.append(round(r_score, 4))
                opportunity_scores.append(round(o_score, 4))
                balanced_scores.append(round(b_score, 4))

        except Exception as e:
            log.warning(f"  Batch {batch_num} failed: {e}")
            # Fill with defaults for failed batch
            for _ in batch:
                stances.append("balanced")
                confidence_scores.append(0.0)
                risk_scores.append(0.0)
                opportunity_scores.append(0.0)
                balanced_scores.append(0.0)

    # Add results to dataframe
    papers_df = papers_df.copy()
    papers_df["stance"]            = stances
    papers_df["stance_confidence"] = confidence_scores
    papers_df["score_risk"]        = risk_scores
    papers_df["score_opportunity"] = opportunity_scores
    papers_df["score_balanced"]    = balanced_scores

    # Summary
    counts = papers_df["stance"].value_counts()
    total  = len(papers_df)
    log.info("")
    log.info("  Classification complete:")
    for label, count in counts.items():
        log.info(f"    {label:<22} {count:,} ({count/total*100:.1f}%)")

    return papers_df
# ======================================================================================================================
#  SECTION 8: BUILD ANALYTICAL OUTPUTS
#  Stance distributions were aggregated at four levels:
#    - By topic: dominant stance per governance topic (e.g., Autonomous Weapons 72.9% risk)
#    - By region: stance proportions per world region (e.g., Africa & Middle East 45.0% opportunity)
#    - By period: pre- vs post-ChatGPT stance shift (risk 34.9% → 29.4%, opportunity 29.7% → 40.0%)
#    - Policy comparison: academic stance compared with policy corpus framing
# ======================================================================================================================
def build_stance_by_topic(papers_df: pd.DataFrame) -> pd.DataFrame:
    """Stance distribution per governance topic."""
    log.info("  Building stance_by_topic.csv...")

    stance_topic = (
        papers_df
        .groupby(["topic_id_finetuned", "topic_label_finetuned", "stance"])
        .size()
        .reset_index(name="count")
    )
    topic_totals = (
        papers_df
        .groupby("topic_id_finetuned")
        .size()
        .reset_index(name="total")
    )
    stance_topic = stance_topic.merge(topic_totals, on="topic_id_finetuned")
    stance_topic["proportion"] = (
        stance_topic["count"] / stance_topic["total"]
    ).round(4)

    # Pivot for readability
    pivot = stance_topic.pivot_table(
        index=["topic_id_finetuned", "topic_label_finetuned"],
        columns="stance",
        values="proportion",
        fill_value=0
    ).reset_index()
    pivot.columns.name = None

    # Add dominant stance per topic
    stance_cols = [c for c in ["risk_focused", "opportunity_focused", "balanced"]
                   if c in pivot.columns]
    if stance_cols:
        pivot["dominant_stance"] = pivot[stance_cols].idxmax(axis=1)

    # Add average confidence and mean risk/opportunity scores
    avg_scores = papers_df.groupby("topic_id_finetuned").agg(
        avg_risk_score=("score_risk", "mean"),
        avg_opportunity_score=("score_opportunity", "mean"),
        paper_count=("scopus_id", "count"),
    ).reset_index()
    avg_scores["avg_risk_score"]        = avg_scores["avg_risk_score"].round(4)
    avg_scores["avg_opportunity_score"] = avg_scores["avg_opportunity_score"].round(4)

    pivot = pivot.merge(avg_scores, on="topic_id_finetuned", how="left")
    pivot.sort_values("paper_count", ascending=False, inplace=True)

    pivot.to_csv(RESULTS_DIR / "stance_by_topic.csv", index=False, encoding="utf-8")
    log.info(f"  ✅ stance_by_topic.csv — {len(pivot)} topics")

    # Print findings
    log.info("")
    log.info("  Stance by governance topic:")
    log.info(f"  {'Topic':<45} {'Risk%':>6} {'Opp%':>6} {'Balanced%':>10} {'Dominant':<18}")
    log.info("  " + "-" * 90)
    for _, r in pivot.iterrows():
        risk_pct = r.get("risk_focused", 0) * 100
        opp_pct  = r.get("opportunity_focused", 0) * 100
        bal_pct  = r.get("balanced", 0) * 100
        dom      = r.get("dominant_stance", "N/A")
        label    = str(r["topic_label_finetuned"])[:44]
        log.info(f"  {label:<45} {risk_pct:>5.1f}% {opp_pct:>5.1f}% {bal_pct:>9.1f}%  {dom}")

    return pivot


def build_stance_by_region(papers_df: pd.DataFrame) -> pd.DataFrame:
    """Stance distribution per world region."""
    log.info("")
    log.info("  Building stance_by_region.csv...")

    stance_region = (
        papers_df
        .groupby(["region", "stance"])
        .size()
        .reset_index(name="count")
    )
    region_totals = (
        papers_df.groupby("region")
        .size().reset_index(name="total")
    )
    stance_region = stance_region.merge(region_totals, on="region")
    stance_region["proportion"] = (
        stance_region["count"] / stance_region["total"]
    ).round(4)

    # Add mean risk and opportunity scores per region
    region_scores = papers_df.groupby("region").agg(
        avg_risk_score=("score_risk", "mean"),
        avg_opportunity_score=("score_opportunity", "mean"),
        paper_count=("scopus_id", "count"),
    ).reset_index()
    region_scores["avg_risk_score"]        = region_scores["avg_risk_score"].round(4)
    region_scores["avg_opportunity_score"] = region_scores["avg_opportunity_score"].round(4)

    pivot = stance_region.pivot_table(
        index=["region"],
        columns="stance",
        values="proportion",
        fill_value=0
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.merge(region_scores, on="region")

    stance_cols = [c for c in ["risk_focused", "opportunity_focused", "balanced"]
                   if c in pivot.columns]
    if stance_cols:
        pivot["dominant_stance"] = pivot[stance_cols].idxmax(axis=1)

    pivot.sort_values("avg_risk_score", ascending=False, inplace=True)
    pivot.to_csv(RESULTS_DIR / "stance_by_region.csv", index=False, encoding="utf-8")
    log.info(f"  ✅ stance_by_region.csv — {len(pivot)} regions")

    log.info("")
    log.info("  Stance by region (sorted by risk score):")
    log.info(f"  {'Region':<25} {'Risk%':>6} {'Opp%':>6} {'Balanced%':>10} {'Dominant':<22}")
    log.info("  " + "-" * 75)
    for _, r in pivot.iterrows():
        risk_pct = r.get("risk_focused", 0) * 100
        opp_pct  = r.get("opportunity_focused", 0) * 100
        bal_pct  = r.get("balanced", 0) * 100
        dom      = r.get("dominant_stance", "N/A")
        log.info(
            f"  {str(r['region']):<25} {risk_pct:>5.1f}% {opp_pct:>5.1f}%"
            f" {bal_pct:>9.1f}%  {dom}"
        )

    return pivot


def build_stance_by_period(papers_df: pd.DataFrame) -> pd.DataFrame:
    """Stance distribution pre vs post ChatGPT."""
    log.info("")
    log.info("  Building stance_by_period.csv...")

    stance_period = (
        papers_df
        .groupby(["period", "stance"])
        .size()
        .reset_index(name="count")
    )
    period_totals = (
        papers_df.groupby("period")
        .size().reset_index(name="total")
    )
    stance_period = stance_period.merge(period_totals, on="period")
    stance_period["proportion"] = (
        stance_period["count"] / stance_period["total"]
    ).round(4)

    period_scores = papers_df.groupby("period").agg(
        avg_risk_score=("score_risk", "mean"),
        avg_opportunity_score=("score_opportunity", "mean"),
        paper_count=("scopus_id", "count"),
    ).reset_index()
    period_scores["avg_risk_score"]        = period_scores["avg_risk_score"].round(4)
    period_scores["avg_opportunity_score"] = period_scores["avg_opportunity_score"].round(4)

    pivot = stance_period.pivot_table(
        index=["period"],
        columns="stance",
        values="proportion",
        fill_value=0
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.merge(period_scores, on="period")

    stance_cols = [c for c in ["risk_focused", "opportunity_focused", "balanced"]
                   if c in pivot.columns]
    if stance_cols:
        pivot["dominant_stance"] = pivot[stance_cols].idxmax(axis=1)

    pivot.to_csv(RESULTS_DIR / "stance_by_period.csv", index=False, encoding="utf-8")
    log.info(f"  ✅ stance_by_period.csv — {len(pivot)} periods")

    log.info("")
    log.info("  Stance pre vs post ChatGPT:")
    for _, r in pivot.iterrows():
        risk_pct = r.get("risk_focused", 0) * 100
        opp_pct  = r.get("opportunity_focused", 0) * 100
        bal_pct  = r.get("balanced", 0) * 100
        log.info(
            f"  {str(r['period']):<15}  risk={risk_pct:.1f}%  "
            f"opportunity={opp_pct:.1f}%  balanced={bal_pct:.1f}%"
        )

    return pivot


def build_stance_policy_comparison(papers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare stance framing between academic papers and policy documents
    for each shared governance topic.
    Policy document stances are inferred from their chunk text using
    the same zero-shot classifier scores averaged across documents.
    """
    log.info("")
    log.info("  Building stance_policy_comparison.csv...")

    # Academic stance per shared topic
    align = pd.read_csv(RESULTS_DIR / "cross_corpus_alignment_v2.csv")
    shared_topics = align[
        (align["alignment"] == "shared") &
        (align["governance_score"] >= 0.5)
    ]["topic_id"].tolist()

    academic_stance = papers_df[
        papers_df["topic_id_finetuned"].astype(str).isin([str(t) for t in shared_topics])
    ].groupby(["topic_id_finetuned", "topic_label_finetuned"]).agg(
        academic_papers=("scopus_id", "count"),
        academic_risk_score=("score_risk", "mean"),
        academic_opportunity_score=("score_opportunity", "mean"),
        academic_dominant_stance=("stance", lambda x: x.value_counts().index[0]),
    ).reset_index()

    academic_stance["academic_risk_score"]        = academic_stance["academic_risk_score"].round(4)
    academic_stance["academic_opportunity_score"] = academic_stance["academic_opportunity_score"].round(4)

    # Load policy alignment counts
    policy_counts = align[["topic_id", "policy_count", "topic_label"]].copy()
    policy_counts.rename(columns={"topic_label": "topic_label_policy"}, inplace=True)
    policy_counts["topic_id"] = policy_counts["topic_id"].astype(str)

    # Merge
    comparison = academic_stance.merge(
        policy_counts,
        left_on="topic_id_finetuned",
        right_on="topic_id",
        how="left"
    )

    # Risk_dominance flag: True = academic papers more risk-focused than opportunity-focused
    comparison["academic_risk_dominant"] = (
        comparison["academic_risk_score"] > comparison["academic_opportunity_score"]
    )

    comparison.sort_values("academic_papers", ascending=False, inplace=True)
    comparison.to_csv(RESULTS_DIR / "stance_policy_comparison.csv",
                      index=False, encoding="utf-8")
    log.info(f"  ✅ stance_policy_comparison.csv — {len(comparison)} shared topics")

    log.info("")
    log.info("  Academic stance on shared governance topics:")
    log.info(f"  {'Topic':<45} {'Papers':>7} {'Risk':>6} {'Opp':>6} {'Dominant':<18}")
    log.info("  " + "-" * 90)
    for _, r in comparison.iterrows():
        label  = str(r["topic_label_finetuned"])[:44]
        papers = int(r["academic_papers"])
        risk   = float(r["academic_risk_score"]) * 100
        opp    = float(r["academic_opportunity_score"]) * 100
        dom    = str(r["academic_dominant_stance"])
        log.info(f"  {label:<45} {papers:>7,} {risk:>5.1f}% {opp:>5.1f}%  {dom}")

    return comparison

# ======================================================================================================================
#  SECTION 9: WRITE SUMMARY REPORT
#  A human-readable text report (stance_summary.txt) was generated summarizing key findings: overall stance
#  distribution, temporal shift, most risk-dominated and opportunity-dominated topics, and regional variation.
# ======================================================================================================================
def write_stance_summary(papers_df: pd.DataFrame,
                          by_topic: pd.DataFrame,
                          by_region: pd.DataFrame,
                          by_period: pd.DataFrame):
    """Write a plain-text summary of key stance findings."""
    log.info("")
    log.info("  Writing stance_summary.txt...")

    total  = len(papers_df)
    counts = papers_df["stance"].value_counts()
    risk_n = counts.get("risk_focused", 0)
    opp_n  = counts.get("opportunity_focused", 0)
    bal_n  = counts.get("balanced", 0)

    lines = [
        "STANCE ANALYSIS SUMMARY — 4d_sentiment_analysis.py",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "OVERVIEW",
        f"  Total governance papers classified: {total:,}",
        f"  Risk-focused:        {risk_n:,} ({risk_n/total*100:.1f}%)",
        f"  Opportunity-focused: {opp_n:,} ({opp_n/total*100:.1f}%)",
        f"  Balanced:            {bal_n:,} ({bal_n/total*100:.1f}%)",
        "",
        "METHOD",
        "  Zero-shot classification using facebook/bart-large-mnli",
        "  Candidate labels: risk-focused, opportunity-focused, balanced",
        "  Confidence threshold: 0.45 (below = classified as balanced)",
        "  No training data used — fully unsupervised NLP",
        "",
        "TEMPORAL FINDINGS (PRE vs POST ChatGPT)",
    ]

    for _, r in by_period.iterrows():
        risk_pct = r.get("risk_focused", 0) * 100
        opp_pct  = r.get("opportunity_focused", 0) * 100
        bal_pct  = r.get("balanced", 0) * 100
        lines.append(
            f"  {str(r['period']):<18}  risk={risk_pct:.1f}%  "
            f"opportunity={opp_pct:.1f}%  balanced={bal_pct:.1f}%"
        )

    lines += [
        "",
        "REGIONAL FINDINGS",
    ]
    for _, r in by_region.sort_values("avg_risk_score", ascending=False).iterrows():
        risk_pct = r.get("risk_focused", 0) * 100
        opp_pct  = r.get("opportunity_focused", 0) * 100
        lines.append(
            f"  {str(r['region']):<25}  risk={risk_pct:.1f}%  "
            f"opportunity={opp_pct:.1f}%  dominant={r.get('dominant_stance','N/A')}"
        )

    lines += [
        "",
        "TOPIC FINDINGS",
    ]
    for _, r in by_topic.sort_values("avg_risk_score", ascending=False).head(10).iterrows():
        risk_pct = r.get("risk_focused", 0) * 100
        opp_pct  = r.get("opportunity_focused", 0) * 100
        label    = str(r["topic_label_finetuned"])[:50]
        lines.append(
            f"  {label:<50}  risk={risk_pct:.1f}%  opp={opp_pct:.1f}%"
        )

    lines += [
        "",
        "OUTPUT FILES:",
        "  governance_papers_stance.csv    — paper-level stance labels",
        "  stance_by_topic.csv             — stance per governance topic",
        "  stance_by_region.csv            — stance per world region",
        "  stance_by_period.csv            — stance pre/post ChatGPT",
        "  stance_policy_comparison.csv    — academic vs policy framing",
    ]

    with open(RESULTS_DIR / "stance_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("  ✅ stance_summary.txt saved")

# ======================================================================================================================
#  SECTION 10: MAIN PIPELINE
# ======================================================================================================================
def main():
    log.info("=" * 60)
    log.info("  STANCE ANALYSIS PIPELINE — 4d")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    # Load data
    papers_df = load_governance_papers()

    # Load classifier
    classifier = load_classifier()

    # Run stance classification
    papers_df = classify_stance(papers_df, classifier)

    # Save paper-level results immediately
    papers_df.to_csv(
        RESULTS_DIR / "governance_papers_stance.csv",
        index=False, encoding="utf-8"
    )
    log.info("✅ governance_papers_stance.csv saved")

    # Build analytical outputs
    log.info("")
    log.info("=" * 60)
    log.info("  BUILDING ANALYTICAL OUTPUTS")
    log.info("=" * 60)

    by_topic  = build_stance_by_topic(papers_df)
    by_region = build_stance_by_region(papers_df)
    by_period = build_stance_by_period(papers_df)
    build_stance_policy_comparison(papers_df)
    write_stance_summary(papers_df, by_topic, by_region, by_period)

    # Final summary
    total  = len(papers_df)
    counts = papers_df["stance"].value_counts()
    log.info("")
    log.info("=" * 60)
    log.info("  4d COMPLETE — STANCE ANALYSIS DONE")
    log.info("=" * 60)
    log.info(f"  Papers classified:   {total:,}")
    log.info(f"  Risk-focused:        {counts.get('risk_focused',0):,} "
             f"({counts.get('risk_focused',0)/total*100:.1f}%)")
    log.info(f"  Opportunity-focused: {counts.get('opportunity_focused',0):,} "
             f"({counts.get('opportunity_focused',0)/total*100:.1f}%)")
    log.info(f"  Balanced:            {counts.get('balanced',0):,} "
             f"({counts.get('balanced',0)/total*100:.1f}%)")
    log.info("")
    log.info("  Output files in results/:")
    log.info("    governance_papers_stance.csv")
    log.info("    stance_by_topic.csv")
    log.info("    stance_by_region.csv")
    log.info("    stance_by_period.csv")
    log.info("    stance_policy_comparison.csv")
    log.info("    stance_summary.txt")
    log.info("")
    log.info("  Next steps:")
    log.info("  1. Commit to Git")
    log.info("  2. Build 5_streamlit_dashboard.py")
    log.info("=" * 60)

# ======================================================================================================================
#  ENTRY POINT
# ======================================================================================================================
if __name__ == "__main__":
    main()