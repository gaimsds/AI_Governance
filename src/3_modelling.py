# =============================================================================
#  BERTOPIC MODELLING SCRIPT
#  Project: Mapping Global AI Governance Narratives
#  Authors: Tambudzai Gundani & Joshua Gray
#  Date:    March 2026
# =============================================================================
#
#  WHAT THIS SCRIPT DOES:
#  ----------------------
#  Runs BERTopic topic modelling on both the Scopus academic corpus and the
#  national AI policy frameworks corpus, then produces comparative outputs
#  that directly answer the three research objectives.
#
#  WORKFLOW:
#  ---------
#  Stage 1 — Data loading + policy document chunking
#  Stage 2 — Model comparison: M1 (Base), M2 (Quality), M3 (Multilingual)
#             Each model is trained on Scopus, then applied to policy corpus
#  Stage 3 — Metric evaluation: coherence, diversity, outlier rate, topic count
#  Stage 4 — Automatic best model selection
#  Stage 5 — Hyperparameter tuning on best model
#  Stage 6 — Final model: research outputs for RO1, RO2, RO3
#  Stage 7 — Save all outputs to results/
#
#  RESEARCH OBJECTIVES ADDRESSED:
#  --------------------------------
#  RO1/RQ1 → Dominant AI risk/governance themes (topic keywords + labels)
#  RO2/RQ2 → Spatial/institutional distribution (topic distributions by region)
#  RO3/RQ3 → Alignment/divergence between academic and policy discourse
#
#  OUTPUT FILES (saved to results/):
#  -----------------------------------
#  model_comparison.csv              — M1/M2/M3 metrics side by side
#  hyperparameter_results.csv        — tuning grid results
#  topics_overview.csv               — all topics: keywords, size, label
#  scopus_topic_assignments.csv      — each paper + its topic + metadata
#  policy_topic_assignments.csv      — each policy chunk + its topic
#  region_topic_distribution.csv     — topic proportions by world region (Scopus)
#  period_topic_distribution.csv     — topic proportions pre/post ChatGPT (Scopus)
#  policy_document_topics.csv        — topic distribution per policy document
#  cross_corpus_alignment.csv        — shared vs divergent topics between corpora
#
#  INSTALL DEPENDENCIES:
#  ---------------------
#  pip install bertopic sentence-transformers umap-learn hdbscan gensim
# =============================================================================


# -----------------------------------------------------------------------------
#  SECTION 1: IMPORTS
# -----------------------------------------------------------------------------

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# BERTopic stack
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Coherence scoring
try:
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora.dictionary import Dictionary
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    print("⚠️  gensim not installed — coherence scoring will be skipped")
    print("   Run: pip install gensim")

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
#  SECTION 2: PATHS
# -----------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent
DATA_CLEAN  = BASE_DIR / "data_clean"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
#  SECTION 3: LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "modelling.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#  SECTION 4: CONFIGURATION
# -----------------------------------------------------------------------------

# Policy document chunking
CHUNK_SIZE       = 500   # words per chunk
CHUNK_OVERLAP    = 50    # word overlap between chunks

# BERTopic base settings (overridden in tuning)
MIN_TOPIC_SIZE   = 30    # minimum papers per topic
NR_TOPICS        = "auto"

# Hyperparameter tuning grid
TUNING_GRID = [
    {"min_cluster_size": 20, "n_neighbors": 10, "n_components": 5},
    {"min_cluster_size": 30, "n_neighbors": 15, "n_components": 5},
    {"min_cluster_size": 50, "n_neighbors": 15, "n_components": 10},
    {"min_cluster_size": 30, "n_neighbors": 10, "n_components": 10},
    {"min_cluster_size": 50, "n_neighbors": 20, "n_components": 5},
]

# The three model configurations to compare
MODEL_CONFIGS = {
    "M1_Base": {
        "embedding_model": "all-MiniLM-L6-v2",
        "description":     "Fast English baseline",
    },
    "M2_Quality": {
        "embedding_model": "all-mpnet-base-v2",
        "description":     "Higher quality English embeddings",
    },
    "M3_Multilingual": {
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "description":     "Multilingual — handles all 35 policy docs",
    },
}


# -----------------------------------------------------------------------------
#  SECTION 5: DATA LOADING
# -----------------------------------------------------------------------------

def load_scopus(max_docs: int = None) -> pd.DataFrame:
    """Load cleaned Scopus abstracts."""
    path = DATA_CLEAN / "scopus_abstracts_nlp.csv"
    if not path.exists():
        path = DATA_CLEAN / "scopus_cleaned.csv"
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # Use abstract_clean if available, otherwise fall back to abstract
    text_col = "abstract_clean" if "abstract_clean" in df.columns else "abstract"
    df = df[df[text_col].notna() & (df[text_col].str.len() > 50)].copy()
    df["text"] = df[text_col].astype(str)

    # Ensure required metadata columns exist
    for col in ["region", "period", "country", "coverDate"]:
        if col not in df.columns:
            df[col] = "Unknown"

    if max_docs:
        df = df.sample(min(max_docs, len(df)), random_state=42)

    log.info(f"   Scopus corpus:  {len(df):,} documents loaded")
    return df.reset_index(drop=True)


def load_policy(english_only: bool = False) -> pd.DataFrame:
    """Load extracted policy corpus."""
    filename = "policy_corpus_english_only.csv" if english_only else "policy_corpus.csv"
    path = DATA_CLEAN / filename
    if not path.exists():
        log.error(f"❌ {filename} not found. Run 2_policy_text_extraction.py first.")
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = df[df["text_clean"].notna() & (df["text_clean"].str.len() > 100)].copy()
    log.info(f"   Policy corpus:  {len(df):,} documents loaded")
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
#  SECTION 6: POLICY DOCUMENT CHUNKING
# -----------------------------------------------------------------------------

def chunk_policy_docs(policy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Split long policy documents into ~500-word chunks.
    Each chunk inherits the metadata of its parent document.
    This allows BERTopic to assign fine-grained topics to different
    sections of long documents like the EU AI Act.
    """
    log.info("   Chunking policy documents...")
    chunks = []

    for _, row in policy_df.iterrows():
        text = str(row.get("text_clean", ""))
        words = text.split()

        if len(words) <= CHUNK_SIZE:
            # Short document — use as-is
            chunk_row = row.to_dict()
            chunk_row["chunk_id"]   = f"{row['doc_id']}_chunk_0"
            chunk_row["chunk_num"]  = 0
            chunk_row["text"]       = text
            chunks.append(chunk_row)
        else:
            # Chunk with overlap
            start = 0
            chunk_num = 0
            while start < len(words):
                end = min(start + CHUNK_SIZE, len(words))
                chunk_text = " ".join(words[start:end])
                if len(chunk_text.split()) >= 50:  # skip tiny trailing chunks
                    chunk_row = row.to_dict()
                    chunk_row["chunk_id"]  = f"{row['doc_id']}_chunk_{chunk_num}"
                    chunk_row["chunk_num"] = chunk_num
                    chunk_row["text"]      = chunk_text
                    chunks.append(chunk_row)
                start += CHUNK_SIZE - CHUNK_OVERLAP
                chunk_num += 1

    chunks_df = pd.DataFrame(chunks).reset_index(drop=True)
    log.info(f"   Policy docs chunked: {len(policy_df)} docs → {len(chunks_df)} chunks")
    return chunks_df


# -----------------------------------------------------------------------------
#  SECTION 7: METRICS
# -----------------------------------------------------------------------------

def compute_coherence(topic_words: list[list[str]],
                      tokenized_docs: list[list[str]]) -> float:
    """Compute topic coherence (c_v) using gensim."""
    if not HAS_GENSIM:
        return -1.0
    try:
        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokenized_docs,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_v",
        )
        return round(cm.get_coherence(), 4)
    except Exception as e:
        log.warning(f"   Coherence computation failed: {e}")
        return -1.0


def compute_diversity(topic_words: list[list[str]]) -> float:
    """
    Topic diversity = proportion of unique words across all topic top-words.
    Range 0–1; higher is better (topics are more distinct).
    """
    if not topic_words:
        return 0.0
    all_words  = [w for words in topic_words for w in words]
    unique     = len(set(all_words))
    total      = len(all_words)
    return round(unique / total, 4) if total > 0 else 0.0


def evaluate_model(model: BERTopic, docs: list[str],
                   topics: list[int]) -> dict:
    """Compute all evaluation metrics for a fitted BERTopic model."""
    topic_info = model.get_topic_info()

    # Topic words for coherence + diversity
    topic_words = []
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue
        words = [w for w, _ in model.get_topic(topic_id)]
        if words:
            topic_words.append(words[:10])

    # Tokenized docs for coherence
    tokenized = [doc.split() for doc in docs]

    # Outlier rate = % docs assigned to topic -1
    n_outliers  = sum(1 for t in topics if t == -1)
    outlier_rate = round(n_outliers / len(topics), 4) if topics else 1.0

    n_topics    = len(topic_info[topic_info["Topic"] != -1])
    coherence   = compute_coherence(topic_words, tokenized)
    diversity   = compute_diversity(topic_words)

    return {
        "n_topics":     n_topics,
        "outlier_rate": outlier_rate,
        "coherence_cv": coherence,
        "diversity":    diversity,
    }


# -----------------------------------------------------------------------------
#  SECTION 8: BUILD BERTOPIC MODEL
# -----------------------------------------------------------------------------

def build_model(embedding_model_name: str,
                min_cluster_size: int = 30,
                n_neighbors: int = 15,
                n_components: int = 5) -> BERTopic:
    """Build a BERTopic model with specified configuration."""

    embedding_model = SentenceTransformer(embedding_model_name)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=5,
        ngram_range=(1, 2),  # unigrams and bigrams
    )

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=NR_TOPICS,
        top_n_words=10,
        verbose=False,
    )

    return model


# -----------------------------------------------------------------------------
#  SECTION 9: STAGE 2 — MODEL COMPARISON
# -----------------------------------------------------------------------------

def run_model_comparison(scopus_texts: list[str],
                         policy_texts: list[str]) -> tuple[str, dict]:
    """
    Train M1, M2, M3 on Scopus texts, transform policy texts,
    evaluate metrics, return best model name and all results.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  STAGE 2 — MODEL COMPARISON")
    log.info("=" * 60)

    comparison_results = {}

    for model_name, config in MODEL_CONFIGS.items():
        log.info("")
        log.info(f"  Training {model_name}: {config['description']}")
        log.info(f"  Embedding: {config['embedding_model']}")

        try:
            model = build_model(config["embedding_model"])

            # Fit on Scopus
            log.info(f"  Fitting on {len(scopus_texts):,} Scopus abstracts...")
            topics, _ = model.fit_transform(scopus_texts)

            # Evaluate on Scopus
            metrics = evaluate_model(model, scopus_texts, topics)

            # Transform policy corpus
            log.info(f"  Transforming {len(policy_texts)} policy chunks...")
            policy_topics, _ = model.transform(policy_texts)
            policy_outlier_rate = round(
                sum(1 for t in policy_topics if t == -1) / len(policy_topics), 4
            )

            metrics["policy_outlier_rate"] = policy_outlier_rate
            metrics["embedding_model"]     = config["embedding_model"]
            metrics["description"]         = config["description"]

            comparison_results[model_name] = {
                "metrics":        metrics,
                "topics":         topics,
                "policy_topics":  policy_topics,
            }

            log.info(f"  ✅ {model_name} results:")
            log.info(f"     Topics found:          {metrics['n_topics']}")
            log.info(f"     Coherence (c_v):        {metrics['coherence_cv']}")
            log.info(f"     Diversity:              {metrics['diversity']}")
            log.info(f"     Outlier rate (Scopus):  {metrics['outlier_rate']:.1%}")
            log.info(f"     Outlier rate (Policy):  {policy_outlier_rate:.1%}")

            # Save model temporarily
            model.save(
                str(MODELS_DIR / f"{model_name}_model"),
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=config["embedding_model"],
            )

        except Exception as e:
            log.error(f"  ❌ {model_name} failed: {e}")
            continue

    # Select best model: highest coherence + diversity, lowest outlier rate
    # Score = coherence * diversity * (1 - outlier_rate)
    best_model_name = None
    best_score      = -1.0

    for name, result in comparison_results.items():
        m = result["metrics"]
        score = (
            max(m["coherence_cv"], 0) *
            m["diversity"] *
            (1 - m["outlier_rate"])
        )
        if score > best_score:
            best_score      = score
            best_model_name = name

    log.info("")
    log.info(f"  🏆 Best model: {best_model_name} (score={best_score:.4f})")

    return best_model_name, comparison_results


# -----------------------------------------------------------------------------
#  SECTION 10: STAGE 3 — HYPERPARAMETER TUNING
# -----------------------------------------------------------------------------

def run_hyperparameter_tuning(scopus_texts: list[str],
                               best_model_name: str) -> dict:
    """
    Grid search over UMAP/HDBSCAN parameters using the best embedding model.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  STAGE 3 — HYPERPARAMETER TUNING")
    log.info("=" * 60)

    embedding_model_name = MODEL_CONFIGS[best_model_name]["embedding_model"]
    tuning_results       = []
    best_params          = None
    best_score           = -1.0

    log.info(f"  Embedding model: {embedding_model_name}")
    log.info(f"  Grid size: {len(TUNING_GRID)} configurations")

    for i, params in enumerate(TUNING_GRID, 1):
        log.info(f"  [{i}/{len(TUNING_GRID)}] Testing: {params}")
        try:
            model = build_model(
                embedding_model_name,
                min_cluster_size=params["min_cluster_size"],
                n_neighbors=params["n_neighbors"],
                n_components=params["n_components"],
            )
            topics, _ = model.fit_transform(scopus_texts)
            metrics   = evaluate_model(model, scopus_texts, topics)

            score = (
                max(metrics["coherence_cv"], 0) *
                metrics["diversity"] *
                (1 - metrics["outlier_rate"])
            )

            result = {**params, **metrics, "score": round(score, 4)}
            tuning_results.append(result)
            log.info(f"     Topics={metrics['n_topics']}  "
                     f"Coherence={metrics['coherence_cv']}  "
                     f"Diversity={metrics['diversity']}  "
                     f"Outliers={metrics['outlier_rate']:.1%}  "
                     f"Score={score:.4f}")

            if score > best_score:
                best_score  = score
                best_params = params

        except Exception as e:
            log.warning(f"     Failed: {e}")
            continue

    log.info("")
    log.info(f"  🏆 Best hyperparameters: {best_params} (score={best_score:.4f})")

    return best_params, tuning_results


# -----------------------------------------------------------------------------
#  SECTION 11: STAGE 4 — FINAL MODEL + RESEARCH OUTPUTS
# -----------------------------------------------------------------------------

def run_final_model(scopus_df: pd.DataFrame,
                    policy_chunks_df: pd.DataFrame,
                    policy_df: pd.DataFrame,
                    best_model_name: str,
                    best_params: dict):
    """
    Train the final model with best configuration and generate all
    research outputs for RO1, RO2, RO3.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  STAGE 4 — FINAL MODEL + RESEARCH OUTPUTS")
    log.info("=" * 60)

    embedding_model_name = MODEL_CONFIGS[best_model_name]["embedding_model"]
    scopus_texts         = scopus_df["text"].tolist()
    policy_texts         = policy_chunks_df["text"].tolist()

    log.info(f"  Building final model ({best_model_name}, params={best_params})")

    final_model = build_model(
        embedding_model_name,
        min_cluster_size=best_params["min_cluster_size"],
        n_neighbors=best_params["n_neighbors"],
        n_components=best_params["n_components"],
    )

    # Fit on Scopus
    log.info(f"  Fitting on {len(scopus_texts):,} Scopus abstracts...")
    scopus_topics, scopus_probs = final_model.fit_transform(scopus_texts)

    # Transform policy corpus
    log.info(f"  Transforming {len(policy_texts)} policy chunks...")
    policy_topics, policy_probs = final_model.transform(policy_texts)

    # ── OUTPUT 1: Topics overview ─────────────────────────────────────────────
    log.info("  Building topics overview...")
    topic_info = final_model.get_topic_info()

    topics_overview = []
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue
        words = final_model.get_topic(topic_id)
        top_words = ", ".join([w for w, _ in words[:10]])
        topics_overview.append({
            "topic_id":    topic_id,
            "topic_size":  row["Count"],
            "top_words":   top_words,
            "label":       row.get("Name", f"Topic_{topic_id}"),
        })

    topics_overview_df = pd.DataFrame(topics_overview)
    topics_overview_df.to_csv(RESULTS_DIR / "topics_overview.csv",
                               index=False, encoding="utf-8")
    log.info(f"  ✅ topics_overview.csv — {len(topics_overview_df)} topics")

    # ── OUTPUT 2: Scopus topic assignments ────────────────────────────────────
    log.info("  Building Scopus topic assignments...")
    scopus_assigned = scopus_df.copy()
    scopus_assigned["topic_id"] = scopus_topics

    # Merge topic labels
    topic_label_map = {
        row["topic_id"]: row["top_words"]
        for _, row in topics_overview_df.iterrows()
    }
    scopus_assigned["topic_words"] = scopus_assigned["topic_id"].map(topic_label_map)

    cols_to_keep = ["eid" if "eid" in scopus_assigned.columns else scopus_assigned.columns[0],
                    "topic_id", "topic_words", "region", "period",
                    "country", "coverDate"]
    cols_to_keep = [c for c in cols_to_keep if c in scopus_assigned.columns]
    scopus_assigned[cols_to_keep].to_csv(
        RESULTS_DIR / "scopus_topic_assignments.csv", index=False, encoding="utf-8"
    )
    log.info(f"  ✅ scopus_topic_assignments.csv")

    # ── OUTPUT 3: Policy topic assignments ───────────────────────────────────
    log.info("  Building policy topic assignments...")
    policy_chunks_assigned = policy_chunks_df.copy()
    policy_chunks_assigned["topic_id"]    = policy_topics
    policy_chunks_assigned["topic_words"] = policy_chunks_assigned["topic_id"].map(topic_label_map)
    policy_chunks_assigned[
        ["chunk_id", "doc_id", "country", "region", "doc_name",
         "doc_type", "year", "topic_id", "topic_words"]
    ].to_csv(RESULTS_DIR / "policy_topic_assignments.csv",
              index=False, encoding="utf-8")
    log.info(f"  ✅ policy_topic_assignments.csv")

    # ── OUTPUT 4: RO2 — Topic distribution by region (Scopus) ────────────────
    log.info("  Building region × topic distribution (RO2)...")
    region_topic = (
        scopus_assigned[scopus_assigned["topic_id"] != -1]
        .groupby(["region", "topic_id"])
        .size()
        .reset_index(name="count")
    )
    region_totals = scopus_assigned.groupby("region").size().reset_index(name="total")
    region_topic  = region_topic.merge(region_totals, on="region")
    region_topic["proportion"] = (region_topic["count"] / region_topic["total"]).round(4)
    region_topic["topic_words"] = region_topic["topic_id"].map(topic_label_map)
    region_topic.to_csv(RESULTS_DIR / "region_topic_distribution.csv",
                        index=False, encoding="utf-8")
    log.info(f"  ✅ region_topic_distribution.csv")

    # ── OUTPUT 5: RO1 — Topic distribution pre/post ChatGPT ──────────────────
    log.info("  Building pre/post ChatGPT topic distribution (RO1)...")
    period_topic = (
        scopus_assigned[scopus_assigned["topic_id"] != -1]
        .groupby(["period", "topic_id"])
        .size()
        .reset_index(name="count")
    )
    period_totals = scopus_assigned.groupby("period").size().reset_index(name="total")
    period_topic  = period_topic.merge(period_totals, on="period")
    period_topic["proportion"] = (period_topic["count"] / period_topic["total"]).round(4)
    period_topic["topic_words"] = period_topic["topic_id"].map(topic_label_map)
    period_topic.to_csv(RESULTS_DIR / "period_topic_distribution.csv",
                        index=False, encoding="utf-8")
    log.info(f"  ✅ period_topic_distribution.csv")

    # ── OUTPUT 6: RO3 — Topic distribution per policy document ───────────────
    log.info("  Building policy document topic distribution (RO3)...")
    policy_doc_topics = (
        policy_chunks_assigned[policy_chunks_assigned["topic_id"] != -1]
        .groupby(["doc_id", "country", "region", "doc_type", "topic_id"])
        .size()
        .reset_index(name="chunk_count")
    )
    policy_doc_totals = (
        policy_chunks_assigned.groupby("doc_id")
        .size()
        .reset_index(name="total_chunks")
    )
    policy_doc_topics = policy_doc_topics.merge(policy_doc_totals, on="doc_id")
    policy_doc_topics["proportion"] = (
        policy_doc_topics["chunk_count"] / policy_doc_topics["total_chunks"]
    ).round(4)
    policy_doc_topics["topic_words"] = policy_doc_topics["topic_id"].map(topic_label_map)
    policy_doc_topics.to_csv(RESULTS_DIR / "policy_document_topics.csv",
                              index=False, encoding="utf-8")
    log.info(f"  ✅ policy_document_topics.csv")

    # ── OUTPUT 7: RO3 — Cross-corpus topic alignment ──────────────────────────
    log.info("  Building cross-corpus topic alignment table (RO3)...")
    scopus_topic_counts = (
        scopus_assigned[scopus_assigned["topic_id"] != -1]
        ["topic_id"].value_counts().reset_index()
    )
    scopus_topic_counts.columns = ["topic_id", "scopus_count"]

    policy_topic_counts = (
        policy_chunks_assigned[policy_chunks_assigned["topic_id"] != -1]
        ["topic_id"].value_counts().reset_index()
    )
    policy_topic_counts.columns = ["topic_id", "policy_count"]

    alignment = topics_overview_df.merge(scopus_topic_counts, on="topic_id", how="left")
    alignment = alignment.merge(policy_topic_counts, on="topic_id", how="left")
    alignment["scopus_count"]  = alignment["scopus_count"].fillna(0).astype(int)
    alignment["policy_count"]  = alignment["policy_count"].fillna(0).astype(int)
    alignment["in_scopus"]     = alignment["scopus_count"] > 0
    alignment["in_policy"]     = alignment["policy_count"] > 0
    alignment["alignment"]     = alignment.apply(
        lambda r: "shared" if r["in_scopus"] and r["in_policy"]
        else ("academic_only" if r["in_scopus"] else "policy_only"),
        axis=1
    )
    alignment.sort_values("scopus_count", ascending=False, inplace=True)
    alignment.to_csv(RESULTS_DIR / "cross_corpus_alignment.csv",
                     index=False, encoding="utf-8")
    log.info(f"  ✅ cross_corpus_alignment.csv")

    # ── Save final model ──────────────────────────────────────────────────────
    log.info("  Saving final model...")
    final_model.save(
        str(MODELS_DIR / "final_bertopic_model"),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=embedding_model_name,
    )
    log.info("  ✅ Final model saved to models/final_bertopic_model/")

    # ── Final summary ──────────────────────────────────────────────────────────
    n_topics      = len(topics_overview_df)
    n_outliers_s  = sum(1 for t in scopus_topics if t == -1)
    n_outliers_p  = sum(1 for t in policy_topics if t == -1)
    shared_topics = (alignment["alignment"] == "shared").sum()

    log.info("")
    log.info("  FINAL MODEL SUMMARY")
    log.info(f"  Topics discovered:          {n_topics}")
    log.info(f"  Scopus papers assigned:     {len(scopus_topics) - n_outliers_s:,} / {len(scopus_topics):,}")
    log.info(f"  Policy chunks assigned:     {len(policy_topics) - n_outliers_p} / {len(policy_topics)}")
    log.info(f"  Shared topics (RO3):        {shared_topics} / {n_topics}")
    log.info(f"  Academic-only topics:       {(alignment['alignment'] == 'academic_only').sum()}")
    log.info(f"  Policy-only topics:         {(alignment['alignment'] == 'policy_only').sum()}")

    return final_model, topics_overview_df, alignment


# -----------------------------------------------------------------------------
#  SECTION 12: SAVE COMPARISON + TUNING RESULTS
# -----------------------------------------------------------------------------

def save_comparison_results(comparison_results: dict,
                             tuning_results: list,
                             best_model_name: str,
                             best_params: dict):
    """Save model comparison and tuning results to CSV."""

    # Model comparison
    rows = []
    for model_name, result in comparison_results.items():
        row = {"model": model_name, **result["metrics"]}
        row["best"] = model_name == best_model_name
        rows.append(row)
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv",
                          index=False, encoding="utf-8")
    log.info(f"\n✅ model_comparison.csv saved")

    # Hyperparameter tuning
    if tuning_results:
        tuning_df = pd.DataFrame(tuning_results)
        tuning_df["best"] = tuning_df.apply(
            lambda r: all(r[k] == best_params[k] for k in best_params), axis=1
        )
        tuning_df.sort_values("score", ascending=False, inplace=True)
        tuning_df.to_csv(RESULTS_DIR / "hyperparameter_results.csv",
                          index=False, encoding="utf-8")
        log.info(f"✅ hyperparameter_results.csv saved")


# -----------------------------------------------------------------------------
#  SECTION 13: MAIN
# -----------------------------------------------------------------------------

def main():

    log.info("=" * 60)
    log.info("  BERTOPIC MODELLING PIPELINE")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    log.info("")
    log.info("STAGE 1 — LOADING DATA")
    log.info("-" * 40)

    scopus_df  = load_scopus()
    policy_df  = load_policy(english_only=False)

    # Chunk policy documents
    policy_chunks_df = chunk_policy_docs(policy_df)

    scopus_texts = scopus_df["text"].tolist()
    policy_texts = policy_chunks_df["text"].tolist()

    log.info(f"   Ready: {len(scopus_texts):,} Scopus texts + {len(policy_texts)} policy chunks")

    # ── Stage 2: Model comparison ─────────────────────────────────────────────
    best_model_name, comparison_results = run_model_comparison(
        scopus_texts, policy_texts
    )

    # ── Stage 3: Hyperparameter tuning ────────────────────────────────────────
    best_params, tuning_results = run_hyperparameter_tuning(
        scopus_texts, best_model_name
    )

    # ── Save comparison + tuning results ──────────────────────────────────────
    save_comparison_results(
        comparison_results, tuning_results, best_model_name, best_params
    )

    # ── Stage 4: Final model + research outputs ────────────────────────────────
    final_model, topics_df, alignment_df = run_final_model(
        scopus_df, policy_chunks_df, policy_df,
        best_model_name, best_params
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"  Output files saved to: {RESULTS_DIR}")
    log.info("")
    log.info("  Files produced:")
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        log.info(f"    {f.name}")
    log.info("")
    log.info("  Next step: run 4_model_finetuning.py for topic labelling")
    log.info("  Then:      run 5_streamlit_dashboard.py to visualise results")
    log.info("=" * 60)
    log.info("🎉 BERTopic modelling complete!")


# -----------------------------------------------------------------------------
#  ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()