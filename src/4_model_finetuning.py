"""
========================================================================================================================
BERTOPIC FINETUNING — TOPIC 0 DECOMPOSITION AND LABELLING
Project: Uneven Science–Policy Translation Shapes Global AI Governance
Authors: Tambudzai G. Charumbira & Joshua Gray
Institution: George Washington University, CCAS | M.S. Data Science | Spring 2026
Date: March 2026

DESCRIPTION:
This script addressed the Topic 0 catch-all problem from Stage 1. The 21,900 papers (53% of the corpus) that
fell into the generic Topic 0 were extracted and processed through a second BERTopic pass with a lower
min_cluster_size (15 vs 50), yielding 88 meaningful sub-topics. These were merged back into the main topic
inventory, producing a final total of 133 topics (45 original + 88 sub-topics).

Each topic was then assigned a human-readable label and a governance relevance score (0.0–1.0) based on
keyword inspection and alignment with OECD AI Principles, EU AI Act risk categories, and UNESCO Ethics
Recommendation. Topics scoring ≥0.5 were classified as governance-relevant.

Note: Sub-topic governance scores were finalized in 4b_update_governance_scores.py after manual review.

OUTPUT FILES (saved to results/):
- topics_finetuned.csv             → All 133 topics with labels and governance scores
- scopus_topics_finetuned.csv      → All papers with updated topic IDs
- topic0_subtopics.csv             → 88 sub-topics found within Topic 0
- governance_topics.csv            → Governance-relevant topics only (score ≥ 0.5)
- governance_papers.csv            → Papers assigned to governance topics
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
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

# ======================================================================================================================
#  SECTION 2: PATHS
# ======================================================================================================================
BASE_DIR    = Path(__file__).parent.parent
DATA_CLEAN  = BASE_DIR / "data_clean"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================================================================
#  SECTION 3: LOGGING
# ======================================================================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "finetuning.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ======================================================================================================================
#  SECTION 4: GOVERNANCE RELEVANCE SCORES AND TOPIC LABELS
#  Each of the 46 original topics was assigned a governance relevance score (0.0–1.0) based on expert inspection of topic
#  keywords and alignment with three established governance frameworks: OECD AI Principles (2019), EU AI Act risk
#  categories (European Parliament, 2024), and UNESCO Ethics Recommendation (2021). Scoring criteria:
#    1.0 = core governance theme (e.g., Algorithmic Fairness, EU Regulation)
#    0.7–0.9 = significant governance relevance (e.g., Cybersecurity, Autonomous Weapons)
#    0.5 = moderate relevance (e.g., Accessibility, Islamic/Religious Ethics)
#    0.1–0.3 = peripheral or application-domain topics
#    0.0 = pure AI application with no governance dimension
#  Topic 0 was excluded from scoring as it was replaced by sub-topics in the decomposition stage.
# ======================================================================================================================
TOPIC_GOVERNANCE_SCORES = {
    0:  None,   # Topic 0 — will be replaced by sub-topics
    1:  0.0,    # Drug/molecular — AI application (bioinformatics)
    2:  0.9,    # Cybersecurity/IoT — AI security governance
    3:  0.1,    # Tourism — AI application
    4:  0.3,    # Sentiment/emotion analysis — some governance (social media)
    5:  0.1,    # Speech/hearing — AI application (accessibility)
    6:  0.0,    # Sport — AI application
    7:  0.2,    # ADHD/autism diagnosis — AI in healthcare, some ethics
    8:  0.7,    # Journalism/news/media — AI and information integrity
    9:  0.3,    # Library/academic — AI in education
    10: 0.1,    # Retinal/ophthalmology — AI application
    11: 0.2,    # Housing/real estate — AI in markets
    12: 0.3,    # Image/video/face recognition — privacy governance implications
    13: 0.1,    # Cultural heritage — AI application
    14: 0.1,    # Music — AI application (creative)
    15: 0.8,    # Crime/police/predictive policing — core governance theme
    16: 0.2,    # Text/NLP/Arabic — AI application (language)
    17: 0.5,    # Religious/Islamic/ethics — AI ethics from non-Western lens
    18: 0.3,    # Entrepreneurship/innovation — AI economy
    19: 0.2,    # Finance/investment/trading — AI in markets
    20: 0.0,    # EEG/brain signals — AI application (neuroscience)
    21: 0.8,    # Trust/trustworthiness — core AI governance theme
    22: 0.4,    # Poverty/migration — AI for development (Global South)
    23: 0.0,    # Bacterial/genomics — AI application (biology)
    24: 0.8,    # Fake news/disinformation — AI and information governance
    25: 0.0,    # Cell/microscopy — AI application (biology)
    26: 0.2,    # Quantum computing — emerging tech
    27: 0.0,    # GHz/wireless/band — AI application (engineering)
    28: 0.0,    # Dental/forensic — AI application
    29: 0.1,    # Sleep/anxiety — AI in health
    30: 0.2,    # Machine translation — AI application (NLP)
    31: 1.0,    # Fairness/bias/algorithmic discrimination — core governance
    32: 0.3,    # Blockchain — AI and distributed systems
    33: 0.3,    # Chatbots/conversational AI — AI interaction governance
    34: 0.9,    # Military/autonomous weapons — lethal AI governance
    35: 0.9,    # Gender bias/stereotypes — AI equity and rights
    36: 0.2,    # Personality/psychological — AI behavioural profiling
    37: 0.4,    # Causal inference — AI methodology (some policy use)
    38: 0.3,    # Malaria/disease — AI for public health (Global South)
    39: 0.0,    # Drug/pharmaceutical — AI application
    40: 0.0,    # Dental education — AI application
    41: 0.4,    # Patent/technology — AI innovation governance
    42: 0.5,    # Disability/accessibility — AI and inclusion
    43: 0.3,    # IoT/edge computing — AI infrastructure
    44: 0.0,    # Optical/sensor — AI application (engineering)
    45: 0.1,    # Tumour/MRI — AI in radiology
}

# Human-readable labels for original topics
TOPIC_LABELS = {
    0:  "General AI/ML (to be decomposed)",
    1:  "Drug Discovery & Molecular AI",
    2:  "AI Cybersecurity & IoT Threats",
    3:  "AI in Tourism & Hospitality",
    4:  "Sentiment & Emotion Analysis",
    5:  "Speech & Language Recognition",
    6:  "AI in Sports & Performance",
    7:  "AI in Neurodevelopmental Diagnosis",
    8:  "AI, Journalism & Media",
    9:  "AI in Academic Libraries",
    10: "AI in Ophthalmology",
    11: "AI in Real Estate Valuation",
    12: "Computer Vision & Face Recognition",
    13: "AI in Cultural Heritage",
    14: "AI in Music & Creative Arts",
    15: "Predictive Policing & Criminal Justice AI",
    16: "NLP & Text Classification",
    17: "AI Ethics from Islamic/Religious Perspective",
    18: "AI in Entrepreneurship & Innovation",
    19: "AI in Finance & Investment",
    20: "AI in Neuroimaging & EEG",
    21: "Trust & Trustworthiness in AI",
    22: "AI for Poverty & Development (Global South)",
    23: "AI in Genomics & Microbiology",
    24: "AI, Fake News & Disinformation",
    25: "AI in Cell Biology & Microscopy",
    26: "Quantum Computing & AI",
    27: "AI in Wireless Communications",
    28: "AI in Forensic & Dental Analysis",
    29: "AI in Sleep & Mental Health",
    30: "Machine Translation & Multilingual AI",
    31: "Algorithmic Fairness & Bias",
    32: "AI & Blockchain Technology",
    33: "Conversational AI & Chatbots",
    34: "Autonomous Weapons & Military AI",
    35: "Gender Bias & AI Representation",
    36: "AI in Personality & Psychological Profiling",
    37: "Causal Inference with AI",
    38: "AI in Epidemiology & Public Health",
    39: "AI in Pharmaceutical Formulation",
    40: "AI in Dental Education",
    41: "AI Patent & Technology Innovation",
    42: "AI Accessibility & Disability Inclusion",
    43: "Edge Computing & IoT Networks",
    44: "AI in Optical & Photonic Devices",
    45: "AI in Brain Tumour Imaging",
}

TOPIC0_SUBLABELS = {
    "educational_ai":       "Educational AI & Learning Systems",
    "nlp_llm":              "NLP, Large Language Models & Text Generation",
    "healthcare_clinical":  "Clinical AI & Healthcare Prediction",
    "ai_governance_ethics": "AI Governance, Ethics & Policy",
    "computer_vision":      "Computer Vision & Image Recognition",
    "recommender":          "Recommender Systems & Personalisation",
    "ai_economics":         "AI in Business & Economic Forecasting",
    "explainability":       "Explainability & Interpretable AI",
    "general_ml":           "General Machine Learning Methods",
}

# Governance scores for Topic 0 sub-topics (assigned after labelling)
TOPIC0_GOV_SCORES = {
    "educational_ai":       0.2,
    "nlp_llm":              0.4,
    "healthcare_clinical":  0.3,
    "ai_governance_ethics": 1.0,
    "computer_vision":      0.2,
    "recommender":          0.3,
    "ai_economics":         0.2,
    "explainability":       0.7,
    "general_ml":           0.1,
}

# ======================================================================================================================
#  SECTION 5: DATA LOADING
#  The cleaned Scopus abstracts and Stage 1 topic assignments were loaded and merged on paper ID. Topic 0
#  papers were identified for decomposition.
# ======================================================================================================================
def load_data():
    """Load Scopus abstracts and existing topic assignments."""
    log.info("Loading data...")

    scopus_path = DATA_CLEAN / "scopus_abstracts_nlp.csv"
    if not scopus_path.exists():
        scopus_path = DATA_CLEAN / "scopus_cleaned.csv"
    scopus_df = pd.read_csv(scopus_path, dtype=str, low_memory=False)
    text_col  = "abstract_clean" if "abstract_clean" in scopus_df.columns else "abstract"
    scopus_df["text"] = scopus_df[text_col].astype(str)

    for col in ["region", "period", "country", "coverDate"]:
        if col not in scopus_df.columns:
            scopus_df[col] = "Unknown"

    assignments_path = RESULTS_DIR / "scopus_topic_assignments.csv"
    assignments_df   = pd.read_csv(assignments_path, dtype=str, low_memory=False)
    assignments_df["topic_id"] = pd.to_numeric(
        assignments_df["topic_id"], errors="coerce"
    ).fillna(-1).astype(int)

    id_col = "eid" if "eid" in scopus_df.columns else scopus_df.columns[0]
    merged = scopus_df.merge(
        assignments_df[[id_col, "topic_id"]],
        on=id_col, how="left"
    )
    merged["topic_id"] = pd.to_numeric(
        merged["topic_id"], errors="coerce"
    ).fillna(-1).astype(int)

    log.info(f"   Loaded {len(merged):,} papers with topic assignments")
    log.info(f"   Topic 0 papers: {(merged['topic_id'] == 0).sum():,}")
    return merged

# ======================================================================================================================
#  SECTION 6: TOPIC 0 DECOMPOSITION
#  A second BERTopic pass was applied to the 21,900 Topic 0 papers using the same embedding model (all-MiniLM-L6-v2) but
#  with a lower min_cluster_size (15) to allow smaller, more specific clusters. This yielded 88 sub-topics. Sub-topic
#  IDs were offset by 100 (i.e., 100, 101, 102...) to avoid collision with the original topic IDs (1–45). The
#  decomposition model was saved separately.
# ======================================================================================================================
def decompose_topic_0(scopus_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Run a second BERTopic pass on Topic 0 papers only.
    Returns a DataFrame with sub-topic assignments and the
    ID offset used so sub-topic IDs don't clash with existing topics.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  STAGE 1 — TOPIC 0 DECOMPOSITION")
    log.info("=" * 60)

    t0_df    = scopus_df[scopus_df["topic_id"] == 0].copy()
    t0_texts = t0_df["text"].tolist()
    log.info(f"   Topic 0 papers to decompose: {len(t0_texts):,}")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    umap_model = UMAP(
        n_neighbors=15,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=3,
        ngram_range=(1, 2),
    )
    sub_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics="auto",
        top_n_words=10,
        verbose=False,
    )

    log.info("   Fitting decomposition model on Topic 0 papers...")
    sub_topics, _ = sub_model.fit_transform(t0_texts)
    sub_info      = sub_model.get_topic_info()
    n_subtopics   = len(sub_info[sub_info["Topic"] != -1])
    n_unassigned  = sum(1 for t in sub_topics if t == -1)
    log.info(f"   Sub-topics found:    {n_subtopics}")
    log.info(f"   Unassigned (noise):  {n_unassigned:,}")
    ID_OFFSET = 100
    t0_df = t0_df.copy()
    t0_df["sub_topic_id"] = [
        (t + ID_OFFSET) if t != -1 else -1
        for t in sub_topics
    ]

    subtopic_rows = []
    for _, row in sub_info.iterrows():
        st_id = row["Topic"]
        if st_id == -1:
            continue
        words = sub_model.get_topic(st_id)
        top_words = ", ".join([w for w, _ in words[:10]])
        subtopic_rows.append({
            "sub_topic_id":    st_id + ID_OFFSET,
            "original_t0_id":  st_id,
            "topic_size":      row["Count"],
            "top_words":       top_words,
        })

    subtopics_df = pd.DataFrame(subtopic_rows)

    log.info("")
    log.info("  Sub-topics found within Topic 0:")
    log.info("  " + "-" * 50)
    for _, row in subtopics_df.iterrows():
        log.info(f"  [{row['sub_topic_id']}] ({row['topic_size']:,} papers)")
        log.info(f"      {row['top_words']}")

    sub_model.save(
        str(MODELS_DIR / "topic0_decomposition_model"),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model="all-MiniLM-L6-v2",
    )
    return t0_df, subtopics_df, ID_OFFSET

# ======================================================================================================================
#  SECTION 7: MERGE SUB-TOPICS INTO MAIN CORPUS
#  Topic 0 assignments in the main dataset were replaced with the sub-topic IDs from the decomposition. Papers that
#  remained as noise (-1) in the second pass retained their unassigned status.
# ======================================================================================================================
def merge_subtopics(scopus_df: pd.DataFrame,
                    t0_df: pd.DataFrame,
                    subtopics_df: pd.DataFrame,
                    id_offset: int) -> pd.DataFrame:
    """
    Replace Topic 0 assignments with sub-topic assignments.
    Papers that were Topic 0 but remained noise (-1) in decomposition
    keep their -1 assignment.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  STAGE 2 — MERGING SUB-TOPICS INTO MAIN CORPUS")
    log.info("=" * 60)
    id_col  = "eid" if "eid" in scopus_df.columns else scopus_df.columns[0]
    updated = scopus_df.copy()
    sub_map = dict(zip(t0_df[id_col], t0_df["sub_topic_id"]))

    def update_topic(row):
        if row["topic_id"] == 0:
            return sub_map.get(row[id_col], -1)
        return row["topic_id"]

    updated["topic_id_finetuned"] = updated.apply(update_topic, axis=1)
    n_reassigned = (updated["topic_id_finetuned"] != updated["topic_id"]).sum()
    log.info(f"   Papers reassigned from Topic 0 → sub-topics: {n_reassigned:,}")
    log.info(f"   Unique topic IDs now: {updated['topic_id_finetuned'].nunique()}")

    return updated

# ======================================================================================================================
#  SECTION 8: LABEL AND SCORE ALL TOPICS
#  The final enriched topic list was built by combining 45 original topics (with pre-assigned labels and governance scores)
#  and 88 sub-topics (with default labels derived from keywords). Sub-topic labels and governance scores were finalized
#  manually after keyword review and updated via 4b_update_governance_scores.py.
# ======================================================================================================================
def label_and_score_topics(merged_df: pd.DataFrame,
                            subtopics_df: pd.DataFrame,
                            id_offset: int) -> pd.DataFrame:
    """
    Build the final enriched topic list with human-readable labels
    and governance relevance scores.
    Prints sub-topic keywords so you can manually assign labels if needed.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  STAGE 3 — LABELLING AND SCORING TOPICS")
    log.info("=" * 60)
    rows = []

    for topic_id, label in TOPIC_LABELS.items():
        if topic_id == 0:
            continue
        gov_score = TOPIC_GOVERNANCE_SCORES.get(topic_id, 0.0)
        count     = (merged_df["topic_id_finetuned"] == topic_id).sum()
        rows.append({
            "topic_id":           topic_id,
            "topic_label":        label,
            "governance_score":   gov_score,
            "paper_count":        count,
            "source":             "original",
            "top_words":          "",
        })

    for _, row in subtopics_df.iterrows():
        st_id    = row["sub_topic_id"]
        count    = (merged_df["topic_id_finetuned"] == st_id).sum()
        label    = f"Topic0_Sub_{st_id}: {row['top_words'][:60]}"
        gov_score = 0.3
        rows.append({
            "topic_id":           st_id,
            "topic_label":        label,
            "governance_score":   gov_score,
            "paper_count":        count,
            "source":             "topic0_subtopic",
            "top_words":          row["top_words"],
        })

    topics_finetuned_df = pd.DataFrame(rows)
    topics_finetuned_df.sort_values("paper_count", ascending=False, inplace=True)
    topics_finetuned_df.reset_index(drop=True, inplace=True)
    gov_df = topics_finetuned_df[topics_finetuned_df["governance_score"] >= 0.5]

    log.info(f"   Total topics after decomposition: {len(topics_finetuned_df)}")
    log.info(f"   Governance-relevant topics (score ≥ 0.5): {len(gov_df)}")
    log.info("")
    log.info("  Governance topics identified:")
    log.info("  " + "-" * 50)
    for _, r in gov_df.sort_values("governance_score", ascending=False).iterrows():
        log.info(f"  [{r['topic_id']:3d}] score={r['governance_score']}  "
                 f"n={r['paper_count']:,}  {r['topic_label'][:55]}")
    return topics_finetuned_df

# ======================================================================================================================
#  SECTION 9: SAVE OUTPUTS
#  All finetuned outputs were saved including: the complete 133-topic inventory, sub-topic details, governance-relevant
#  subset, updated paper assignments, and regional/temporal governance distributions.
# ======================================================================================================================
def save_outputs(merged_df: pd.DataFrame,
                 topics_finetuned_df: pd.DataFrame,
                 subtopics_df: pd.DataFrame):
    """Save all finetuned outputs to results/."""
    log.info("")
    log.info("=" * 60)
    log.info("  STAGE 4 — SAVING OUTPUTS")
    log.info("=" * 60)

    # 1. Full finetuned topic list
    out = RESULTS_DIR / "topics_finetuned.csv"
    topics_finetuned_df.to_csv(out, index=False, encoding="utf-8")
    log.info(f"    topics_finetuned.csv — {len(topics_finetuned_df)} topics")

    # 2. Sub-topic detail
    out = RESULTS_DIR / "topic0_subtopics.csv"
    subtopics_df.to_csv(out, index=False, encoding="utf-8")
    log.info(f"    topic0_subtopics.csv — {len(subtopics_df)} sub-topics")

    # 3. Governance topics only
    gov_df = topics_finetuned_df[topics_finetuned_df["governance_score"] >= 0.5]
    out    = RESULTS_DIR / "governance_topics.csv"
    gov_df.to_csv(out, index=False, encoding="utf-8")
    log.info(f"    governance_topics.csv — {len(gov_df)} topics")

    # 4. Full Scopus assignments with finetuned topic IDs
    id_col     = "eid" if "eid" in merged_df.columns else merged_df.columns[0]
    label_map  = dict(zip(topics_finetuned_df["topic_id"],
                           topics_finetuned_df["topic_label"]))
    score_map  = dict(zip(topics_finetuned_df["topic_id"],
                           topics_finetuned_df["governance_score"]))

    merged_df["topic_label_finetuned"]    = merged_df["topic_id_finetuned"].map(label_map)
    merged_df["governance_score"]         = merged_df["topic_id_finetuned"].map(score_map)

    cols = [id_col, "topic_id_finetuned", "topic_label_finetuned",
            "governance_score", "region", "period", "country"]
    cols = [c for c in cols if c in merged_df.columns]
    out  = RESULTS_DIR / "scopus_topics_finetuned.csv"
    merged_df[cols].to_csv(out, index=False, encoding="utf-8")
    log.info(f"    scopus_topics_finetuned.csv — {len(merged_df):,} papers")

    # 5. Governance papers only
    gov_papers = merged_df[merged_df["governance_score"] >= 0.5]
    out        = RESULTS_DIR / "governance_papers.csv"
    gov_papers[cols].to_csv(out, index=False, encoding="utf-8")
    log.info(f"    governance_papers.csv — {len(gov_papers):,} papers")

    # 6. Updated region × governance topic distribution
    region_gov = (
        gov_papers[gov_papers["topic_id_finetuned"] != -1]
        .groupby(["region", "topic_id_finetuned", "topic_label_finetuned"])
        .size()
        .reset_index(name="count")
    )
    region_totals = merged_df.groupby("region").size().reset_index(name="total")
    region_gov    = region_gov.merge(region_totals, on="region")
    region_gov["proportion"] = (region_gov["count"] / region_gov["total"]).round(4)
    out = RESULTS_DIR / "region_governance_distribution.csv"
    region_gov.to_csv(out, index=False, encoding="utf-8")
    log.info(f"    region_governance_distribution.csv")

    # 7. Updated period × governance topic distribution
    period_gov = (
        gov_papers[gov_papers["topic_id_finetuned"] != -1]
        .groupby(["period", "topic_id_finetuned", "topic_label_finetuned"])
        .size()
        .reset_index(name="count")
    )
    period_totals = merged_df.groupby("period").size().reset_index(name="total")
    period_gov    = period_gov.merge(period_totals, on="period")
    period_gov["proportion"] = (period_gov["count"] / period_gov["total"]).round(4)
    out = RESULTS_DIR / "period_governance_distribution.csv"
    period_gov.to_csv(out, index=False, encoding="utf-8")
    log.info(f"    period_governance_distribution.csv")

    log.info("")
    log.info("    ACTION REQUIRED — Review Sub-Topic Labels:")
    log.info("  Open results/topic0_subtopics.csv")
    log.info("  Read the top_words for each sub-topic")
    log.info("  Open results/topics_finetuned.csv")
    log.info("  Update the topic_label and governance_score columns")
    log.info("  for all rows where source = 'topic0_subtopic'")
    log.info("  Then re-run the save step or update manually in Excel")

    return merged_df

# ======================================================================================================================
#  SECTION 10: SUMMARY
# ======================================================================================================================
def print_summary(merged_df: pd.DataFrame,
                  topics_finetuned_df: pd.DataFrame):

    gov_papers = merged_df[merged_df["governance_score"] >= 0.5]
    n_assigned = (merged_df["topic_id_finetuned"] != -1).sum()
    n_total    = len(merged_df)
    n_gov      = len(gov_papers)

    log.info("")
    log.info("=" * 60)
    log.info("  FINETUNING SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total topics after decomposition: {len(topics_finetuned_df)}")
    log.info(f"  Papers assigned (non-outlier):    {n_assigned:,} / {n_total:,} "
             f"({n_assigned/n_total:.1%})")
    log.info(f"  Governance-relevant papers:       {n_gov:,} / {n_total:,} "
             f"({n_gov/n_total:.1%})")
    log.info("")
    log.info("  Topic 0 decomposition result:")
    sub_rows = topics_finetuned_df[topics_finetuned_df["source"] == "topic0_subtopic"]
    for _, r in sub_rows.sort_values("paper_count", ascending=False).iterrows():
        log.info(f"    Sub-topic {r['topic_id']:3d}: {r['paper_count']:,} papers")
        log.info(f"      {r['top_words'][:70]}")
    log.info("")
    log.info("  Next steps:")
    log.info("  1. Review topic0_subtopics.csv and update labels manually")
    log.info("  2. Run 5_streamlit_dashboard.py for interactive visualisation")
    log.info("=" * 60)
    log.info(" Finetuning complete!")

# ======================================================================================================================
#  SECTION 11: MAIN PIPELINE
# ======================================================================================================================
def main():

    log.info("=" * 60)
    log.info("  BERTOPIC FINETUNING PIPELINE")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    scopus_df = load_data()
    t0_df, subtopics_df, id_offset = decompose_topic_0(scopus_df)
    merged_df = merge_subtopics(scopus_df, t0_df, subtopics_df, id_offset)
    topics_finetuned_df = label_and_score_topics(merged_df, subtopics_df, id_offset)
    merged_df = save_outputs(merged_df, topics_finetuned_df, subtopics_df)
    print_summary(merged_df, topics_finetuned_df)

# ======================================================================================================================
#  SECTION 11: MAIN PIPELINE
# ======================================================================================================================
if __name__ == "__main__":
    main()