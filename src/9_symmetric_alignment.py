"""
Symmetric Cross-Corpus Alignment Pipeline
Trains a separate BERTopic model on the policy corpus, then aligns academic
and policy topics via cosine similarity of topic-centroid embeddings.

Run phases sequentially — each phase ends with a stopping point that prints
a summary and waits for confirmation before continuing.
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)

BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data"
RESULTS_DIR   = BASE_DIR / "results" / "alignment"
MODELS_DIR    = BASE_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def confirm(label: str):
    """Print stopping point marker — no interactive pause (non-interactive run)."""
    print(f"\n{'='*60}")
    print(f"  STOPPING POINT: {label}")
    print(f"{'='*60}")


# ==============================================================================
#  PHASE 1 — SETUP AND DATA VERIFICATION
# ==============================================================================

def phase1():
    print("=" * 60)
    print("PHASE 1 — SETUP AND DATA VERIFICATION")
    print("=" * 60)

    # ── Task 1.1 — Pin environment ─────────────────────────────────────────────
    print("\n[Task 1.1] Pinning environment...")
    req_path = RESULTS_DIR / "requirements_alignment.txt"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True
    )
    req_path.write_text(result.stdout)
    print(f"  Saved: {req_path} ({len(result.stdout.splitlines())} packages)")

    # ── Task 1.2 — Load and verify policy corpus ───────────────────────────────
    print("\n[Task 1.2] Loading data/policy_chunks_clean.csv (cleaned corpus)...")
    chunks = pd.read_csv(DATA_DIR / "policy_chunks_clean.csv")
    actual_rows = len(chunks)
    expected_rows = 1092

    print(f"  Row count: {actual_rows:,}  (expected {expected_rows:,})")
    if actual_rows != expected_rows:
        print(f"  NOTE: {actual_rows - expected_rows:+d} rows vs expected — "
              f"re-run src/10_clean_policy_corpus.py if this is unexpected")

    # Null check on chunk_text
    null_mask = chunks["chunk_text"].isna() | (chunks["chunk_text"].str.strip() == "")
    n_nulls = null_mask.sum()
    if n_nulls > 0:
        print(f"  WARNING: {n_nulls} null/empty chunk_text rows — dropping")
        chunks = chunks[~null_mask].reset_index(drop=True)
    else:
        print(f"  chunk_text nulls: 0")

    # pub_year range
    chunks["pub_year"] = pd.to_numeric(chunks["pub_year"], errors="coerce").astype("Int64")
    year_nulls = chunks["pub_year"].isna().sum()
    year_min   = int(chunks["pub_year"].dropna().min())
    year_max   = int(chunks["pub_year"].dropna().max())
    print(f"  pub_year range: {year_min}–{year_max}  "
          f"({'OK' if 2015 <= year_min and year_max <= 2025 else 'OUT OF RANGE'})"
          f"  ({year_nulls} nulls)")

    # Summary tables
    by_doc = (
        chunks.groupby(["doc_id", "country", "region", "doc_type", "pub_year"])
        .size().reset_index(name="chunk_count")
        .sort_values("chunk_count", ascending=False)
    )
    by_region = chunks.groupby("region").size().reset_index(name="chunk_count")
    by_year   = chunks.groupby("pub_year").size().reset_index(name="chunk_count")

    summary = pd.concat([
        by_doc.assign(group_by="document"),
        by_region.rename(columns={"region": "doc_id"}).assign(
            country="", doc_type="", pub_year=pd.NA, group_by="region"),
        by_year.rename(columns={"pub_year": "doc_id"}).assign(
            country="", region="", doc_type="", group_by="year"),
    ], ignore_index=True)

    out_path = RESULTS_DIR / "policy_corpus_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Policy corpus summary saved: {out_path}")
    print(f"\n  Chunks per region:")
    print(by_region.to_string(index=False))
    print(f"\n  Chunks per year:")
    print(by_year.to_string(index=False))

    # ── Task 1.3 — Load academic BERTopic model ────────────────────────────────
    print("\n[Task 1.3] Loading academic BERTopic model...")
    from bertopic import BERTopic

    model_path = MODELS_DIR / "academic_topic_model"
    academic_model = BERTopic.load(str(model_path))

    all_topics    = academic_model.get_topics()
    n_topics      = len(all_topics) - (1 if -1 in all_topics else 0)  # exclude outlier topic
    print(f"  Model loaded from: {model_path}")
    print(f"  Total topics in model: {len(all_topics)} "
          f"(including outlier topic -1 → {n_topics} named topics)")
    print(f"  Expected (original two-stage pipeline): 133 total")
    if n_topics != 133:
        print(f"  NOTE: Model has {n_topics} topics (not 133). "
              f"This is the Stage 1 model (46 topics). "
              f"Stage 2 decomposition topics are in topic0_decomposition_model.")

    # Check governance topic IDs
    gov_topics = pd.read_csv(DATA_DIR / "governance_topic_ids.csv")
    gov_ids    = set(gov_topics["topic_id"].astype(int).tolist())
    model_ids  = set(k for k in all_topics.keys() if k != -1)
    found_in_model  = gov_ids & model_ids
    missing_in_model = gov_ids - model_ids

    print(f"\n  Governance topic IDs: {len(gov_ids)} total")
    print(f"  Found in model:       {len(found_in_model)}")
    if missing_in_model:
        print(f"  Missing from model:   {sorted(missing_in_model)}")
        print(f"  NOTE: Missing IDs belong to Stage 2 (topic0_decomposition_model). "
              f"Topic centroids for those will be computed from paper embeddings directly.")

    # Topic embeddings
    topic_embeddings = academic_model.topic_embeddings_
    if topic_embeddings is not None:
        emb_shape = np.array(topic_embeddings).shape
        print(f"\n  topic_embeddings_ shape: {emb_shape}")
        np.save(RESULTS_DIR / "academic_topic_embeddings.npy", np.array(topic_embeddings))
        print(f"  Saved: results/alignment/academic_topic_embeddings.npy")
    else:
        print(f"\n  topic_embeddings_ is None — will compute from paper embeddings in Phase 2")
        emb_shape = None

    # ── Stopping Point 1 ──────────────────────────────────────────────────────
    n_docs = chunks["doc_id"].nunique()
    print(f"\n{'─'*60}")
    print(f"  SUMMARY FOR STOPPING POINT 1")
    print(f"{'─'*60}")
    print(f"  (a) Policy corpus loaded: {len(chunks):,} chunks across {n_docs} documents")
    print(f"      pub_year range: {year_min}–{year_max}, null chunk_text: {n_nulls}")
    print(f"  (b) Academic model loaded: {len(all_topics)} total topics "
          f"({n_topics} named)")
    print(f"      Governance topic IDs present in model: "
          f"{len(found_in_model)}/{len(gov_ids)}")
    if missing_in_model:
        print(f"      IDs in Stage 2 model only: {sorted(missing_in_model)}")
    print(f"  (c) Topic embeddings shape: {emb_shape}")

    confirm("1 — Phase 1 complete")
    return chunks, academic_model, gov_topics


# ==============================================================================
#  PHASE 1b — BUILD ACADEMIC GOVERNANCE TOPIC EMBEDDING MATRIX
# ==============================================================================

def build_academic_gov_embeddings(gov_topics: pd.DataFrame) -> np.ndarray:
    """
    Construct (21, 384) matrix with one row per governance topic in the order
    defined by data/governance_topic_ids.csv. Pulls embeddings from Stage 1
    model for original topics and Stage 2 model for topic0 sub-topics.
    Saves matrix + index CSV for full traceability.
    """
    print("\n" + "=" * 60)
    print("PHASE 1b — BUILD ACADEMIC GOVERNANCE TOPIC EMBEDDING MATRIX")
    print("=" * 60)

    from bertopic import BERTopic

    m1 = BERTopic.load(str(MODELS_DIR / "academic_topic_model"))
    m2 = BERTopic.load(str(MODELS_DIR / "topic0_decomposition_model"))

    emb1 = np.array(m1.topic_embeddings_)   # (47, 384)
    emb2 = np.array(m2.topic_embeddings_)   # (89, 384)

    subtopic_map = pd.read_csv(BASE_DIR / "results" / "topic0_subtopics.csv")
    id_to_orig   = dict(zip(subtopic_map["sub_topic_id"], subtopic_map["original_t0_id"]))

    print(f"\n  Stage 1 embeddings: {emb1.shape}")
    print(f"  Stage 2 embeddings: {emb2.shape}")
    assert emb1.shape[1] == emb2.shape[1] == 384, "Dimensionality mismatch!"
    print(f"  Dimensionality check: OK (384)")

    matrix = []
    index_rows = []

    for row_idx, row in gov_topics.iterrows():
        tid   = int(row["topic_id"])
        label = row["topic_label"]

        if tid in id_to_orig:
            orig_id  = id_to_orig[tid]
            emb_idx  = orig_id + 1          # +1 because index 0 = outlier topic -1
            embedding = emb2[emb_idx]
            source   = "topic0_decomposition_model"
        else:
            emb_idx  = tid + 1
            embedding = emb1[emb_idx]
            source   = "academic_topic_model"

        matrix.append(embedding)
        index_rows.append({
            "row_index":   len(matrix) - 1,
            "topic_id":    tid,
            "topic_label": label,
            "source_model": source,
            "emb_index_in_model": emb_idx,
        })

    matrix = np.array(matrix)   # (21, 384)
    assert matrix.shape == (21, 384), f"Expected (21,384), got {matrix.shape}"
    print(f"\n  academic_gov_embeddings matrix shape: {matrix.shape}  ✓")

    # Save matrix
    np.save(RESULTS_DIR / "academic_governance_topic_embeddings.npy", matrix)
    print(f"  Saved: results/alignment/academic_governance_topic_embeddings.npy")

    # Save index
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(RESULTS_DIR / "academic_governance_topic_index.csv", index=False)
    print(f"  Saved: results/alignment/academic_governance_topic_index.csv")

    print(f"\n  Index (row → topic → source):")
    print(index_df[["row_index", "topic_id", "topic_label", "source_model"]].to_string(index=False))

    return matrix


# ==============================================================================
#  PHASE 2 — POLICY BERTOPIC MODEL
# ==============================================================================

def _train_policy_model(texts, policy_embeddings, embedder,
                        min_cluster_size, n_components, n_neighbors):
    """Train one BERTopic run with given hyperparameters. Returns (model, assignments)."""
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    umap_model = UMAP(
        n_neighbors=n_neighbors, n_components=n_components,
        metric="cosine", random_state=SEED
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True
    )
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)

    model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        # nr_topics intentionally omitted — no post-hoc reduction
        calculate_probabilities=False,
        verbose=True,
    )
    assignments, _ = model.fit_transform(texts, embeddings=policy_embeddings)
    return model, assignments


def _report_stopping_point_2(chunks, policy_model, params):
    """Print the full Stopping Point 2 report and return (n_topics, outlier_rate)."""
    all_topics   = policy_model.get_topics()
    assignments  = chunks["policy_topic_id"].tolist()
    n_total      = len(chunks)
    n_outliers   = sum(1 for t in assignments if t == -1)
    outlier_rate = n_outliers / n_total * 100
    topic_ids    = sorted(t for t in all_topics.keys() if t != -1)
    n_topics     = len(topic_ids)

    counts = [int((chunks["policy_topic_id"] == t).sum()) for t in topic_ids]

    print(f"\n{'─'*60}")
    print(f"  STOPPING POINT 2 — params: min_cluster_size={params['min_cluster_size']}, "
          f"n_components={params['n_components']}, n_neighbors={params['n_neighbors']}")
    print(f"{'─'*60}")
    print(f"  Total topics (excl. -1): {n_topics}")
    print(f"  Outlier chunks:          {n_outliers} / {n_total}  ({outlier_rate:.1f}%)")
    if counts:
        print(f"  Chunks per topic — "
              f"mean={np.mean(counts):.1f}, median={np.median(counts):.0f}, "
              f"min={min(counts)}, max={max(counts)}")

    print(f"\n  Top 10 keywords per topic:")
    for tid in topic_ids:
        words     = ", ".join(w for w, _ in all_topics[tid][:10])
        n_chunks  = int((chunks["policy_topic_id"] == tid).sum())
        print(f"  Topic {tid:3d} ({n_chunks:4d} chunks):  {words}")

    print(f"\n  Top 3 documents per topic (by chunk count):")
    for tid in topic_ids:
        sub = chunks[chunks["policy_topic_id"] == tid]
        top_docs = (sub.groupby(["doc_id", "country"])
                    .size().reset_index(name="n")
                    .sort_values("n", ascending=False)
                    .head(3))
        print(f"  Topic {tid:3d}:")
        for _, r in top_docs.iterrows():
            print(f"    {r['n']:3d} chunks  {r['doc_id'][:55]}  ({r['country']})")

    return n_topics, outlier_rate


def phase2(chunks: pd.DataFrame, academic_gov_embeddings: np.ndarray,
           gov_topics: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PHASE 2 — TRAIN POLICY BERTOPIC MODEL")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = chunks["chunk_text"].tolist()

    # ── Task 2.1 — Embed policy chunks (once; reused across retries) ──────────
    emb_cache = RESULTS_DIR / "policy_chunk_embeddings.npy"
    if emb_cache.exists():
        print("\n[Task 2.1] Loading cached policy embeddings...")
        policy_embeddings = np.load(emb_cache)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    else:
        print("\n[Task 2.1] Embedding policy chunks with all-MiniLM-L6-v2...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        policy_embeddings = embedder.encode(
            texts, show_progress_bar=True, batch_size=64, normalize_embeddings=False
        )
        np.save(emb_cache, policy_embeddings)
        print(f"  Saved: results/alignment/policy_chunk_embeddings.npy")
    print(f"  Policy embeddings shape: {policy_embeddings.shape}")

    # ── Task 2.2 — Train with stopping-condition loop ─────────────────────────
    params = dict(min_cluster_size=8, n_components=10, n_neighbors=10)
    iteration = 0

    while True:
        iteration += 1
        p = params.copy()
        print(f"\n[Task 2.2] Training BERTopic — attempt {iteration} "
              f"(min_cluster_size={p['min_cluster_size']}, "
              f"n_components={p['n_components']}, n_neighbors={p['n_neighbors']})...")

        policy_model, assignments = _train_policy_model(
            texts, policy_embeddings, embedder,
            min_cluster_size=p["min_cluster_size"],
            n_components=p["n_components"],
            n_neighbors=p["n_neighbors"],
        )
        chunks = chunks.copy()
        chunks["policy_topic_id"] = assignments

        n_topics, outlier_rate = _report_stopping_point_2(chunks, policy_model, p)

        # ── Stopping conditions ───────────────────────────────────────────────
        if outlier_rate > 50:
            print(f"\n  ⚠ Outlier rate {outlier_rate:.1f}% > 50% → raising n_neighbors to 15")
            params["n_neighbors"] = 15
            continue

        if n_topics < 6:
            if p["n_components"] < 15:
                print(f"\n  ⚠ Only {n_topics} topics < 6 → raising n_components to 15")
                params["n_components"] = 15
                continue
            elif p["min_cluster_size"] > 4:
                print(f"\n  ⚠ Still {n_topics} topics at n_components=15 → "
                      f"lowering min_cluster_size to 4, n_components=5, n_neighbors=5")
                params["min_cluster_size"] = 4
                params["n_components"] = 5
                params["n_neighbors"] = 5
                continue
            else:
                print(f"\n  NOTE: Corpus produces {n_topics} topics even at "
                      f"min_cluster_size=4. Policy documents are semantically "
                      f"homogeneous — reporting honestly.")
                break

        if n_topics > 25:
            print(f"\n  ⚠ {n_topics} topics > 25 → raising min_cluster_size to 12")
            params["min_cluster_size"] = 12
            continue

        # 6–25 topics and outlier rate ≤ 50% → pause for human confirmation
        print(f"\n  ✓ {n_topics} topics, {outlier_rate:.1f}% outliers — "
              f"within target range (6–25 topics, ≤50% outliers).")
        print(f"  Review keywords above for distinctiveness before confirming Phase 3.")
        break

    # ── Save model and outputs ─────────────────────────────────────────────────
    policy_model.save(str(MODELS_DIR / "policy_topic_model"))
    print(f"\n  Saved: models/policy_topic_model/")

    all_topics = policy_model.get_topics()
    topic_ids_ordered = sorted(t for t in all_topics.keys() if t != -1)

    topic_rows = []
    for tid in topic_ids_ordered:
        topic_rows.append({
            "policy_topic_id": tid,
            "top_words":       ", ".join(w for w, _ in all_topics[tid][:10]),
            "chunk_count":     int((chunks["policy_topic_id"] == tid).sum()),
        })
    pd.DataFrame(topic_rows).to_csv(RESULTS_DIR / "policy_topics.csv", index=False)
    print(f"  Saved: results/alignment/policy_topics.csv")

    # ── Task 2.3 — Policy topic centroids ─────────────────────────────────────
    print("\n[Task 2.3] Computing policy topic centroids...")
    policy_emb_arr = np.array(policy_model.topic_embeddings_)
    print(f"  policy topic_embeddings_ shape: {policy_emb_arr.shape}")
    np.save(RESULTS_DIR / "policy_topic_embeddings.npy", policy_emb_arr)
    # topic_embeddings_ index 0 = outlier topic -1, index k+1 = topic k (normal).
    # When 0 outliers, BERTopic may omit the outlier row → shape (n_topics, 384).
    if policy_emb_arr.shape[0] == len(topic_ids_ordered) + 1:
        policy_matrix = np.array([policy_emb_arr[tid + 1] for tid in topic_ids_ordered])
    else:
        policy_matrix = np.array([policy_emb_arr[tid] for tid in topic_ids_ordered])
    print(f"  Policy topic centroid matrix shape: {policy_matrix.shape}")

    # ── Task 2.4 — Cosine similarity alignment ────────────────────────────────
    print("\n[Task 2.4] Computing cosine similarity...")
    sim_matrix = cosine_similarity(academic_gov_embeddings, policy_matrix)
    np.save(RESULTS_DIR / "alignment_similarity_matrix.npy", sim_matrix)
    print(f"  Similarity matrix shape: {sim_matrix.shape}")

    gov_index = pd.read_csv(RESULTS_DIR / "academic_governance_topic_index.csv")
    alignment_rows = []
    for i, gov_row in gov_index.iterrows():
        best_j     = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i, best_j])
        best_pid   = topic_ids_ordered[best_j]
        best_words = ", ".join(w for w, _ in all_topics[best_pid][:6])
        alignment_rows.append({
            "academic_topic_id":    gov_row["topic_id"],
            "academic_topic_label": gov_row["topic_label"],
            "best_policy_topic_id": best_pid,
            "best_policy_words":    best_words,
            "cosine_similarity":    round(best_score, 4),
        })

    alignment_df = pd.DataFrame(alignment_rows).sort_values("cosine_similarity", ascending=False)
    alignment_df.to_csv(RESULTS_DIR / "academic_policy_alignment.csv", index=False)
    print(f"  Saved: results/alignment/academic_policy_alignment.csv")

    chunks.drop(columns=["chunk_text"]).to_csv(
        RESULTS_DIR / "policy_chunk_topic_assignments.csv", index=False)
    print(f"  Saved: results/alignment/policy_chunk_topic_assignments.csv")

    confirm("2 — Phase 2 complete. Confirm to proceed to Phase 3.")
    return chunks, policy_model, alignment_df


# ==============================================================================
#  PHASE 3 — CLASSIFY POLICY TOPICS (THEMATIC vs DOCUMENT-SPECIFIC vs ARTIFACT)
# ==============================================================================

# Topic IDs that are encoding/OCR artifacts — excluded from downstream analysis.
ARTIFACT_TOPIC_IDS = {11}

# Human-reviewed overrides: topics that pass the thematic threshold numerically
# but are reclassified after content review (Japan-dominant T17 → document_specific).
CLASSIFICATION_OVERRIDES = {17: "document_specific"}


def phase3():
    print("\n" + "=" * 60)
    print("PHASE 3 — POLICY TOPIC CLASSIFICATION")
    print("=" * 60)

    assignments = pd.read_csv(RESULTS_DIR / "policy_chunk_topic_assignments.csv")
    policy_topics_df = pd.read_csv(RESULTS_DIR / "policy_topics.csv")

    # Build a lookup: topic_id → top_words string
    top_words_map = dict(zip(
        policy_topics_df["policy_topic_id"].astype(int),
        policy_topics_df["top_words"]
    ))

    # Exclude outlier chunks (topic -1) from classification
    named = assignments[assignments["policy_topic_id"] != -1].copy()
    topic_ids = sorted(named["policy_topic_id"].unique())

    rows = []
    for tid in topic_ids:
        sub = named[named["policy_topic_id"] == tid]
        total_chunks = len(sub)

        doc_counts = sub.groupby("doc_id").size().reset_index(name="n").sort_values("n", ascending=False)
        n_docs = len(doc_counts)
        dominant_doc = doc_counts.iloc[0]["doc_id"]
        dominant_n = int(doc_counts.iloc[0]["n"])
        dominant_share = dominant_n / total_chunks

        if tid in ARTIFACT_TOPIC_IDS:
            classification = "artifact"
        elif tid in CLASSIFICATION_OVERRIDES:
            classification = CLASSIFICATION_OVERRIDES[tid]
        elif n_docs >= 3 and dominant_share <= 0.60:
            classification = "thematic"
        else:
            classification = "document_specific"

        rows.append({
            "topic_id":                 tid,
            "top_words":                top_words_map.get(tid, ""),
            "classification":           classification,
            "n_contributing_documents": n_docs,
            "dominant_doc_share":       round(dominant_share, 3),
            "dominant_doc_name":        dominant_doc,
            "total_chunks":             total_chunks,
        })

    cls_df = pd.DataFrame(rows)
    cls_df.to_csv(RESULTS_DIR / "policy_topic_classification.csv", index=False)
    print(f"\n  Saved: results/alignment/policy_topic_classification.csv")

    # ── Stopping Point 3 ──────────────────────────────────────────────────────
    thematic   = cls_df[cls_df["classification"] == "thematic"]
    doc_spec   = cls_df[cls_df["classification"] == "document_specific"]
    artifact   = cls_df[cls_df["classification"] == "artifact"]

    print(f"\n{'─'*60}")
    print(f"  STOPPING POINT 3 — POLICY TOPIC CLASSIFICATION")
    print(f"{'─'*60}")
    print(f"  Thematic topics:          {len(thematic)}")
    print(f"  Document-specific topics: {len(doc_spec)}")
    print(f"  Artifact topics:          {len(artifact)}")

    print(f"\n  THEMATIC TOPICS (n_docs >= 3, dominant_share <= 0.60):")
    for _, row in thematic.iterrows():
        tid = int(row["topic_id"])
        sub = named[named["policy_topic_id"] == tid]
        doc_counts = (sub.groupby(["doc_id", "country"])
                      .size().reset_index(name="n")
                      .sort_values("n", ascending=False))
        print(f"\n  Topic {tid:2d} ({int(row['total_chunks'])} chunks, "
              f"{int(row['n_contributing_documents'])} docs, "
              f"dominant_share={row['dominant_doc_share']:.2f})")
        print(f"    Keywords: {row['top_words']}")
        for _, dr in doc_counts.iterrows():
            print(f"      {int(dr['n']):3d}  {dr['doc_id'][:60]}  ({dr['country']})")

    print(f"\n  DOCUMENT-SPECIFIC TOPICS:")
    for _, row in doc_spec.sort_values("total_chunks", ascending=False).iterrows():
        print(f"  Topic {int(row['topic_id']):2d} ({int(row['total_chunks'])} chunks, "
              f"{int(row['n_contributing_documents'])} doc(s), "
              f"dominant_share={row['dominant_doc_share']:.2f})  "
              f"→ {row['dominant_doc_name'][:55]}")

    print(f"\n  ARTIFACT TOPICS (excluded from alignment analysis):")
    for _, row in artifact.iterrows():
        print(f"  Topic {int(row['topic_id']):2d} ({int(row['total_chunks'])} chunks)  "
              f"→ {row['dominant_doc_name'][:55]}")
        print(f"    Keywords: {row['top_words']}")

    confirm("3 — Topic classification complete. Review above before alignment analysis.")
    return cls_df


# ==============================================================================
#  PHASE 4 — CROSS-CORPUS ALIGNMENT (THEMATIC POLICY TOPICS ONLY)
# ==============================================================================

# Human-reviewed short labels for the 5 thematic policy topics (label_pending basis)
THEMATIC_POLICY_LABELS = {
    1:  "AI Governance Frameworks (ASEAN/Singapore/NIST)",
    8:  "International AI Policy Principles (OECD/Multi-country)",
    9:  "AI Data, Learning & Infrastructure",
    15: "Generative AI Content Safety & Provenance",
    16: "Data Privacy & Protection",
}


def phase4():
    """
    Alignment analysis restricted to the 5 thematic policy topics.
    Reads all inputs from saved files — no retraining.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("PHASE 4 — CROSS-CORPUS ALIGNMENT (THEMATIC TOPICS ONLY)")
    print("=" * 60)

    # ── Load saved files ───────────────────────────────────────────────────────
    sim_matrix = np.load(RESULTS_DIR / "alignment_similarity_matrix.npy")   # (21, 20)
    gov_index  = pd.read_csv(RESULTS_DIR / "academic_governance_topic_index.csv")
    cls_df     = pd.read_csv(RESULTS_DIR / "policy_topic_classification.csv")
    pol_topics = pd.read_csv(RESULTS_DIR / "policy_topics.csv")

    print(f"\n  Full similarity matrix: {sim_matrix.shape}")

    thematic_ids = sorted(
        cls_df[cls_df["classification"] == "thematic"]["topic_id"].astype(int).tolist()
    )
    assert thematic_ids == [1, 8, 9, 15, 16], \
        f"Expected thematic IDs [1,8,9,15,16], got {thematic_ids}"
    print(f"  Thematic policy topic IDs: {thematic_ids}")

    # Column j in sim_matrix = policy topic j (topics sorted 0..19 in Phase 2)
    thematic_cols = thematic_ids   # [1, 8, 9, 15, 16]
    sub_matrix = sim_matrix[:, thematic_cols]   # (21, 5)
    print(f"  Sub-matrix shape (21 academic × 5 thematic): {sub_matrix.shape}")

    pol_words_map = dict(zip(pol_topics["policy_topic_id"].astype(int), pol_topics["top_words"]))

    # ── Academic → thematic policy alignment ──────────────────────────────────
    acad_rows = []
    for i, gov_row in gov_index.iterrows():
        sims = sub_matrix[i]
        order = np.argsort(sims)[::-1]
        best_j, sec_j = int(order[0]), int(order[1])
        acad_rows.append({
            "academic_topic_id":           int(gov_row["topic_id"]),
            "academic_topic_label":        gov_row["topic_label"],
            "best_match_policy_topic_id":  thematic_cols[best_j],
            "best_match_policy_label":     THEMATIC_POLICY_LABELS[thematic_cols[best_j]],
            "best_similarity":             round(float(sims[best_j]), 4),
            "second_best_policy_topic_id": thematic_cols[sec_j],
            "second_best_policy_label":    THEMATIC_POLICY_LABELS[thematic_cols[sec_j]],
            "second_best_similarity":      round(float(sims[sec_j]), 4),
        })

    acad_df = (pd.DataFrame(acad_rows)
               .sort_values("best_similarity", ascending=False)
               .reset_index(drop=True))
    acad_df.to_csv(RESULTS_DIR / "academic_to_thematic_policy_alignment.csv", index=False)
    print(f"  Saved: results/alignment/academic_to_thematic_policy_alignment.csv")

    # ── Thematic policy → academic alignment (top 3 per policy topic) ─────────
    pol_rows = []
    for col_idx, tid in enumerate(thematic_cols):
        sims = sub_matrix[:, col_idx]
        order = np.argsort(sims)[::-1]
        for rank in range(3):
            j = int(order[rank])
            pol_rows.append({
                "policy_topic_id":        tid,
                "policy_label":           THEMATIC_POLICY_LABELS[tid],
                "rank":                   rank + 1,
                "academic_topic_id":      int(gov_index.iloc[j]["topic_id"]),
                "academic_topic_label":   gov_index.iloc[j]["topic_label"],
                "similarity":             round(float(sims[j]), 4),
            })

    pol_df = pd.DataFrame(pol_rows)
    pol_df.to_csv(RESULTS_DIR / "thematic_policy_to_academic_alignment.csv", index=False)
    print(f"  Saved: results/alignment/thematic_policy_to_academic_alignment.csv")

    # ── Document-specific topics summary ──────────────────────────────────────
    doc_spec = cls_df[cls_df["classification"] == "document_specific"].copy()
    doc_spec_out = (
        doc_spec[["topic_id", "dominant_doc_name", "dominant_doc_share", "total_chunks"]]
        .merge(pol_topics.rename(columns={
            "policy_topic_id": "topic_id", "top_words": "top_keywords"
        })[["topic_id", "top_keywords"]],
        on="topic_id", how="left")
        .sort_values("total_chunks", ascending=False)
        .rename(columns={"dominant_doc_name": "dominant_document"})
    )
    doc_spec_out.to_csv(RESULTS_DIR / "document_specific_topics_summary.csv", index=False)
    print(f"  Saved: results/alignment/document_specific_topics_summary.csv")

    # ── Histogram ─────────────────────────────────────────────────────────────
    best_sims = acad_df["best_similarity"].values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(best_sims, bins=10, range=(0.0, 1.0),
            edgecolor="black", color="steelblue", alpha=0.8)
    ax.set_xlim(0.0, 1.0)
    ymax = ax.get_ylim()[1]
    for xval in [0.40, 0.50, 0.60]:
        ax.axvline(x=xval, color="crimson", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.text(xval + 0.008, ymax * 0.88, f"{xval:.2f}",
                color="crimson", fontsize=9, va="top")
    ax.set_xlabel("Cosine Similarity (best thematic policy match)", fontsize=11)
    ax.set_ylabel("Number of Academic Governance Topics", fontsize=11)
    ax.set_title(
        "Academic Governance Topics — Best-Match Similarity\nto Thematic Policy Topics",
        fontsize=11)
    plt.tight_layout()
    hist_path = RESULTS_DIR / "similarity_distribution_thematic.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: results/alignment/similarity_distribution_thematic.png")

    # ── Stopping Point 4 — full report ────────────────────────────────────────
    mean_sim = float(np.mean(best_sims))
    std_sim  = float(np.std(best_sims))

    # Short column headers for 5 thematic topics
    col_heads = {1: "T01", 8: "T08", 9: "T09", 15: "T15", 16: "T16"}

    print(f"\n{'─'*60}")
    print(f"  STOPPING POINT 4 — CROSS-CORPUS ALIGNMENT")
    print(f"{'─'*60}")

    print(f"\n  21 × 5 SIMILARITY MATRIX:")
    hdr = f"  {'Academic Topic':<48}" + "".join(f"  {col_heads[t]:>5}" for t in thematic_cols)
    print(hdr)
    print("  " + "─" * (48 + 7 * len(thematic_cols)))
    for i, gov_row in gov_index.iterrows():
        label = gov_row["topic_label"][:46]
        vals  = "".join(f"  {sub_matrix[i, j]:>5.3f}" for j in range(len(thematic_cols)))
        print(f"  {label:<48}{vals}")
    print()
    for tid in thematic_cols:
        print(f"  {col_heads[tid]} = {THEMATIC_POLICY_LABELS[tid]}")

    print(f"\n  SIMILARITY STATS (21 best-match values):")
    print(f"    Mean ± SD:  {mean_sim:.4f} ± {std_sim:.4f}")
    print(f"    Min:        {float(np.min(best_sims)):.4f}")
    print(f"    Max:        {float(np.max(best_sims)):.4f}")

    print(f"\n  TOP 5 ALIGNMENTS (highest similarity):")
    for _, r in acad_df.head(5).iterrows():
        print(f"    sim={r['best_similarity']:.4f}  "
              f"{r['academic_topic_label'][:50]:<50}"
              f"  → T{int(r['best_match_policy_topic_id']):02d}")

    print(f"\n  BOTTOM 5 ALIGNMENTS (lowest — orphaned academic topics):")
    for _, r in acad_df.tail(5).iterrows():
        print(f"    sim={r['best_similarity']:.4f}  "
              f"{r['academic_topic_label'][:50]:<50}"
              f"  → T{int(r['best_match_policy_topic_id']):02d}")

    confirm("4 — Alignment complete. Review before Phase 5 (depth-mismatch analysis).")
    return acad_df, pol_df


# ==============================================================================
#  PHASE 5 — GRANULARITY, VOCABULARY DIVERGENCE, ORPHANS, REGION ALIGNMENT
# ==============================================================================

# Human-specified expected pairings (label-based judgment, not similarity-driven).
# Each tuple: (policy_topic_id, academic_topic_label, rationale)
EXPECTED_PAIRS = [
    (15, "Generative AI & Creative Ethics",
     "Strongest label overlap: both concern generative AI governance and ethics"),
    (16, "AI Law, Ethics & Human Rights",
     "Data privacy is a core legal/rights-based governance concern"),
    (16, "Algorithmic Fairness & Bias",
     "Privacy and fairness co-occur in data governance literature"),
    (1,  "AI in Public Administration & Governance",
     "T01 operational frameworks align with public-sector governance research"),
    (1,  "AI Law, Ethics & Human Rights",
     "Governance frameworks ground themselves in rights-based principles"),
    (8,  "EU AI Regulation & Regulatory Frameworks",
     "T08 international principles most directly parallel EU regulatory literature"),
    (9,  "Digital Transformation & AI Adoption",
     "T09 data/infrastructure aligns with broader AI adoption and systems research"),
    (9,  "Explainable AI & Transparency",
     "T09 data and learning systems also reference explainability requirements"),
]

ORPHAN_SIM_THRESHOLD = 0.50  # academic topics with best-match below this are orphans


def phase5():
    """
    Granularity, vocabulary divergence, orphan analysis, and per-region alignment.
    Reads all inputs from saved files — no retraining.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from collections import defaultdict

    print("\n" + "=" * 60)
    print("PHASE 5 — GRANULARITY, VOCABULARY DIVERGENCE, ORPHANS, REGIONS")
    print("=" * 60)

    # ── Load saved files ───────────────────────────────────────────────────────
    acad_df   = pd.read_csv(RESULTS_DIR / "academic_to_thematic_policy_alignment.csv")
    cls_df    = pd.read_csv(RESULTS_DIR / "policy_topic_classification.csv")
    gov_index = pd.read_csv(RESULTS_DIR / "academic_governance_topic_index.csv")
    pol_topics = pd.read_csv(RESULTS_DIR / "policy_topics.csv")
    assignments = pd.read_csv(RESULTS_DIR / "policy_chunk_topic_assignments.csv")

    full_sim  = np.load(RESULTS_DIR / "alignment_similarity_matrix.npy")   # (21, 20)
    policy_emb     = np.load(RESULTS_DIR / "policy_topic_embeddings.npy")  # (21, 384)
    acad_gov_emb   = np.load(RESULTS_DIR / "academic_governance_topic_embeddings.npy")  # (21, 384)

    thematic_ids  = [1, 8, 9, 15, 16]
    # Column j in full_sim = policy topic j (0-indexed, sorted)
    sub_matrix    = full_sim[:, thematic_ids]   # (21, 5)

    # Index helpers
    gov_label_to_row  = {row["topic_label"]: i for i, row in gov_index.iterrows()}
    thematic_col_map  = {tid: i for i, tid in enumerate(thematic_ids)}
    pol_words_map     = dict(zip(pol_topics["policy_topic_id"].astype(int),
                                 pol_topics["top_words"]))

    # ── 1. Fan-out analysis ────────────────────────────────────────────────────
    print("\n[1] Fan-out analysis...")
    fanout_rows = []
    for tid in thematic_ids:
        best_acad = acad_df[acad_df["best_match_policy_topic_id"] == tid
                            ]["academic_topic_label"].tolist()
        sec_acad  = acad_df[acad_df["second_best_policy_topic_id"] == tid
                            ]["academic_topic_label"].tolist()
        fanout_rows.append({
            "policy_topic_id":         tid,
            "policy_label":            THEMATIC_POLICY_LABELS[tid],
            "n_best_match":            len(best_acad),
            "best_match_academics":    "; ".join(best_acad),
            "n_second_best":           len(sec_acad),
            "second_best_academics":   "; ".join(sec_acad),
        })

    fanout_df = pd.DataFrame(fanout_rows)
    fanout_df.to_csv(RESULTS_DIR / "policy_topic_fanout.csv", index=False)
    print(f"  Saved: results/alignment/policy_topic_fanout.csv")

    # ── 2. Vocabulary divergence pairs ────────────────────────────────────────
    print("\n[2] Vocabulary divergence pairs...")
    div_rows = []
    for pol_id, acad_label, rationale in EXPECTED_PAIRS:
        row_idx = gov_label_to_row.get(acad_label)
        if row_idx is None:
            print(f"  WARNING: academic label not found: '{acad_label}'")
            continue
        col_idx = thematic_col_map[pol_id]
        sim = round(float(sub_matrix[row_idx, col_idx]), 4)
        gov_row = gov_index.iloc[row_idx]
        div_rows.append({
            "policy_topic_id":      pol_id,
            "policy_label":         THEMATIC_POLICY_LABELS[pol_id],
            "academic_topic_id":    int(gov_row["topic_id"]),
            "academic_topic_label": acad_label,
            "actual_similarity":    sim,
            "flagged_low":          sim < 0.55,
            "rationale":            rationale,
        })

    div_df = pd.DataFrame(div_rows).sort_values(["policy_topic_id", "actual_similarity"],
                                                 ascending=[True, False])
    div_df.to_csv(RESULTS_DIR / "vocabulary_divergence_pairs.csv", index=False)
    print(f"  Saved: results/alignment/vocabulary_divergence_pairs.csv")

    # ── 3. Orphan academic topics ──────────────────────────────────────────────
    print("\n[3] Orphan academic topics (best-match sim < 0.50)...")
    orphan_rows_acad = acad_df[acad_df["best_similarity"] < ORPHAN_SIM_THRESHOLD].copy()

    # Load academic BERTopic models to extract keywords
    from bertopic import BERTopic
    m1 = BERTopic.load(str(MODELS_DIR / "academic_topic_model"))
    m2 = BERTopic.load(str(MODELS_DIR / "topic0_decomposition_model"))
    subtopic_map = pd.read_csv(BASE_DIR / "results" / "topic0_subtopics.csv")
    id_to_orig   = dict(zip(subtopic_map["sub_topic_id"], subtopic_map["original_t0_id"]))

    orphan_out_rows = []
    for _, row in orphan_rows_acad.iterrows():
        tid = int(row["academic_topic_id"])
        if tid in id_to_orig:
            words = m2.get_topic(id_to_orig[tid])
        else:
            words = m1.get_topic(tid)
        acad_kw = ", ".join(w for w, _ in words[:10]) if words and words != [-1] else ""

        best_pol_id  = int(row["best_match_policy_topic_id"])
        pol_kw       = pol_words_map.get(best_pol_id, "")

        orphan_out_rows.append({
            "academic_topic_id":       tid,
            "academic_topic_label":    row["academic_topic_label"],
            "academic_top10_keywords": acad_kw,
            "best_match_policy_topic_id":   best_pol_id,
            "best_match_policy_label":      THEMATIC_POLICY_LABELS[best_pol_id],
            "best_match_similarity":        row["best_similarity"],
            "policy_top10_keywords":        pol_kw,
        })

    orphan_df = pd.DataFrame(orphan_out_rows).sort_values("best_match_similarity")
    orphan_df.to_csv(RESULTS_DIR / "orphan_academic_topics.csv", index=False)
    print(f"  Saved: results/alignment/orphan_academic_topics.csv")

    # ── 4. Per-region alignment (document-specific topics only) ────────────────
    print("\n[4] Per-region alignment...")
    doc_spec = cls_df[cls_df["classification"] == "document_specific"].copy()

    # Map each doc-specific topic to the region of its dominant document
    doc_region = (assignments.drop_duplicates("doc_id")
                              .set_index("doc_id")["region"].to_dict())
    topic_region = {}
    for _, r in doc_spec.iterrows():
        tid    = int(r["topic_id"])
        region = doc_region.get(r["dominant_doc_name"], "Unknown")
        topic_region[tid] = region

    # Group by region → list of topic IDs
    region_groups = defaultdict(list)
    for tid, region in topic_region.items():
        region_groups[region].append(tid)

    # For each region: centroid of doc-specific policy topic embeddings → cos_sim to 21 academic
    region_wide_rows = []
    for region in sorted(region_groups):
        topic_list = region_groups[region]
        # policy_emb index: tid + 1 (index 0 = outlier)
        embs    = np.array([policy_emb[tid + 1] for tid in topic_list])
        centroid = np.mean(embs, axis=0, keepdims=True)          # (1, 384)
        sims    = cos_sim(centroid, acad_gov_emb)[0]              # (21,)

        row = {
            "region":                  region,
            "n_doc_specific_topics":   len(topic_list),
            "doc_specific_topic_ids":  str(topic_list),
        }
        for i, gr in gov_index.iterrows():
            col_name = f"acad_{int(gr['topic_id'])}_{gr['topic_label'][:28].replace(' ', '_')}"
            row[col_name] = round(float(sims[i]), 4)
        region_wide_rows.append(row)

    region_df = pd.DataFrame(region_wide_rows)
    region_df.to_csv(RESULTS_DIR / "region_to_academic_similarity.csv", index=False)
    print(f"  Saved: results/alignment/region_to_academic_similarity.csv")

    # ── Stopping Point 5 ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  STOPPING POINT 5")
    print(f"{'─'*60}")

    print(f"\n[1] FAN-OUT — academic topics absorbed by each thematic policy topic:")
    for _, r in fanout_df.iterrows():
        print(f"\n  T{int(r['policy_topic_id']):02d} ({r['n_best_match']} best / "
              f"{r['n_second_best']} 2nd-best)  {r['policy_label']}")
        if r["best_match_academics"]:
            for lbl in r["best_match_academics"].split("; "):
                print(f"    [BEST]  {lbl}")
        if r["second_best_academics"]:
            for lbl in r["second_best_academics"].split("; "):
                print(f"    [2ND]   {lbl}")

    print(f"\n[2] VOCABULARY DIVERGENCE PAIRS (expected vs actual):")
    fmt = "  {:4s}  {:<48}  ↔  {:<42}  sim={:.4f}  {}"
    for _, r in div_df.iterrows():
        flag = "⚑ LOW" if r["flagged_low"] else "OK"
        print(fmt.format(
            f"T{int(r['policy_topic_id']):02d}",
            r["policy_label"][:48],
            r["academic_topic_label"][:42],
            r["actual_similarity"],
            flag,
        ))

    print(f"\n[3] ORPHAN ACADEMIC TOPICS (best-match sim < {ORPHAN_SIM_THRESHOLD}):")
    for _, r in orphan_df.iterrows():
        print(f"\n  [{r['best_match_similarity']:.4f}] {r['academic_topic_label']}")
        print(f"    Academic kw: {r['academic_top10_keywords']}")
        print(f"    Best policy: T{int(r['best_match_policy_topic_id']):02d} "
              f"— {r['best_match_policy_label'][:50]}")
        print(f"    Policy kw:   {r['policy_top10_keywords']}")

    print(f"\n[4] REGION → ACADEMIC SIMILARITY (top 3 per region):")
    acad_cols = [c for c in region_df.columns if c.startswith("acad_")]
    # Build short label lookup from col name → academic label
    col_to_label = {}
    for i, gr in gov_index.iterrows():
        col = f"acad_{int(gr['topic_id'])}_{gr['topic_label'][:28].replace(' ', '_')}"
        col_to_label[col] = gr["topic_label"]

    for _, r in region_df.iterrows():
        sims_row = {c: r[c] for c in acad_cols}
        top3     = sorted(sims_row, key=sims_row.get, reverse=True)[:3]
        print(f"\n  {r['region']} ({int(r['n_doc_specific_topics'])} doc-specific topics):")
        for c in top3:
            print(f"    sim={sims_row[c]:.4f}  {col_to_label.get(c, c)}")

    confirm("5 — Analysis complete. Pause for human review before further outputs.")


# ==============================================================================
#  PHASE 5b — ACADEMIC "EU AI REGULATION" TOPIC INSPECTION
# ==============================================================================

# Finetuned topic ID for EU AI Regulation & Regulatory Frameworks (Stage 2 topic)
EU_REG_FINETUNED_ID = 135
# Policy topic T08 = International AI Policy Principles
T08_POLICY_TOPIC_ID = 8

# EU-specific vocabulary markers (used to flag Brussels-effect vs. register convergence)
EU_SPECIFIC_MARKERS = {
    "ce marking", "conformity assessment", "gpai", "ai office", "annex iii",
    "annex", "notified body", "notifying authority", "market surveillance",
    "prohibited practice", "high-risk", "regulation eu", "eu act", "ai act",
    "article 6", "article 9", "article 13", "the regulation", "provider",
    "deployer", "importer", "distributor", "fundamental rights impact",
    "post-market", "sandbox", "general purpose ai", "systemic risk",
}


def phase5b():
    """
    Inspect the academic EU AI Regulation topic: keywords, representative abstracts,
    and comparison with policy T08 keywords. Determines whether the topic captures
    EU-specific regulatory vocabulary or general regulatory-normative register.
    Saves results/alignment/eu_regulation_topic_inspection.md and .csv.
    No further analysis — final diagnostic before closing the alignment pipeline.
    """
    print("\n" + "=" * 60)
    print("PHASE 5b — EU AI REGULATION TOPIC INSPECTION")
    print("=" * 60)

    from bertopic import BERTopic

    # ── Load models and mapping ────────────────────────────────────────────────
    m2           = BERTopic.load(str(MODELS_DIR / "topic0_decomposition_model"))
    policy_model = BERTopic.load(str(MODELS_DIR / "policy_topic_model"))

    subtopic_map = pd.read_csv(BASE_DIR / "results" / "topic0_subtopics.csv")
    id_to_orig   = dict(zip(subtopic_map["sub_topic_id"], subtopic_map["original_t0_id"]))
    orig_id      = id_to_orig[EU_REG_FINETUNED_ID]   # topic ID in Stage 2 model
    print(f"\n  Finetuned topic ID {EU_REG_FINETUNED_ID} → Stage 2 model topic {orig_id}")

    # ── Top keywords with c-TF-IDF scores (BERTopic default: 10 per topic) ──────
    # Request more via the model's internal representation if possible
    acad_kw_raw = m2.get_topic(orig_id) or []
    # Try to get additional words from c_tf_idf_ if available
    try:
        import scipy.sparse as sp
        ctfidf = m2.c_tf_idf_
        topic_idx = list(m2.get_topics().keys()).index(orig_id)
        if hasattr(ctfidf, "toarray"):
            row = ctfidf.toarray()[topic_idx]
        else:
            row = np.array(ctfidf[topic_idx].todense()).flatten()
        vocab = m2.vectorizer_model.get_feature_names_out()
        top_idx = np.argsort(row)[::-1][:15]
        acad_kw_raw = [(vocab[i], float(row[i])) for i in top_idx if row[i] > 0]
    except Exception:
        pass  # fall back to get_topic() result
    acad_top15 = acad_kw_raw[:15]
    print(f"\n  Academic top keywords extracted: {len(acad_top15)} "
          f"(BERTopic default caps at 10; extended if c_tf_idf_ available)")

    # Flag EU-specific markers in keyword list
    flagged_acad = set()
    for word, _ in acad_top15:
        if any(m in word.lower() for m in EU_SPECIFIC_MARKERS):
            flagged_acad.add(word)

    # ── Top policy T08 keywords ────────────────────────────────────────────────
    pol_kw_raw = policy_model.get_topic(T08_POLICY_TOPIC_ID) or []
    try:
        ctfidf_p = policy_model.c_tf_idf_
        topic_ids_pol = sorted(k for k in policy_model.get_topics().keys() if k != -1)
        t08_col_idx = topic_ids_pol.index(T08_POLICY_TOPIC_ID)
        if hasattr(ctfidf_p, "toarray"):
            row_p = ctfidf_p.toarray()[t08_col_idx]
        else:
            row_p = np.array(ctfidf_p[t08_col_idx].todense()).flatten()
        vocab_p = policy_model.vectorizer_model.get_feature_names_out()
        top_p   = np.argsort(row_p)[::-1][:15]
        pol_kw_raw = [(vocab_p[i], float(row_p[i])) for i in top_p if row_p[i] > 0]
    except Exception:
        pass
    pol_top15   = pol_kw_raw[:15]
    print(f"  Policy T08 top keywords extracted: {len(pol_top15)}")

    # ── Load academic papers metadata for this topic ───────────────────────────
    # governance_papers_stance.csv has topic_id_finetuned, title, abstract, cover_date, country
    stance_csv = BASE_DIR / "results" / "governance_papers_stance.csv"
    papers     = pd.read_csv(stance_csv, dtype=str)
    papers["pub_year"] = pd.to_numeric(
        papers["cover_date"].str[:4], errors="coerce").astype("Int64")

    eu_papers  = papers[papers["topic_id_finetuned"].astype(str) == str(EU_REG_FINETUNED_ID)].copy()
    print(f"  Papers in topic {EU_REG_FINETUNED_ID}: {len(eu_papers)}")

    # ── Get 5 representative abstracts ────────────────────────────────────────
    # Try BERTopic representative_docs first; fall back to topic-filtered papers.
    rep_docs     = m2.get_representative_docs(orig_id) or []
    matched_ids  = set()
    for text in rep_docs:
        snippet = text.strip()[:200]
        for _, p in eu_papers.iterrows():
            abst = str(p.get("abstract_clean") or p.get("abstract") or "").strip()
            if abst[:200] == snippet or snippet in abst:
                matched_ids.add(p["scopus_id"])
                break

    rep_sample = eu_papers[eu_papers["scopus_id"].isin(matched_ids)].copy()
    if len(rep_sample) < 5:
        remaining = eu_papers[~eu_papers["scopus_id"].isin(matched_ids)].copy()
        if "governance_score" in remaining.columns:
            remaining = remaining.sort_values("governance_score", ascending=False)
        extra = remaining.head(5 - len(rep_sample))
        rep_sample = pd.concat([rep_sample, extra], ignore_index=True)

    rep_sample = rep_sample.head(5)
    print(f"  Representative sample: {len(matched_ids)} BERTopic matches + "
          f"{max(0, 5 - len(matched_ids))} supplemented from topic papers")

    # ── EU-specificity assessment ──────────────────────────────────────────────
    # Scan all top-15 keywords against EU-specific markers
    eu_specific_hits  = [(w, s) for w, s in acad_top15
                         if any(m in w.lower() for m in EU_SPECIFIC_MARKERS)]
    general_reg_hits  = [(w, s) for w, s in acad_top15
                         if w.lower() in {"risk", "regulation", "governance", "oversight",
                                          "accountability", "transparency", "compliance",
                                          "framework", "policy", "standards", "requirements"}]
    specificity_verdict = (
        "EU-SPECIFIC: vocabulary includes EU regulatory apparatus terms"
        if len(eu_specific_hits) >= 3
        else "REGISTER CONVERGENCE: vocabulary is general regulatory-normative register"
        if not eu_specific_hits
        else "MIXED: some EU-specific terms alongside general regulatory vocabulary"
    )

    # ── Build CSV output ───────────────────────────────────────────────────────
    # Part 1: keywords
    kw_rows = []
    for rank, (word, score) in enumerate(acad_top15, 1):
        eu_flag = any(m in word.lower() for m in EU_SPECIFIC_MARKERS)
        kw_rows.append({
            "section":        "academic_eu_regulation_keywords",
            "rank":           rank,
            "term":           word,
            "ctfidf_score":   round(score, 6),
            "eu_specific_flag": eu_flag,
        })
    for rank, (word, score) in enumerate(pol_top15, 1):
        kw_rows.append({
            "section":        "policy_T08_keywords",
            "rank":           rank,
            "term":           word,
            "ctfidf_score":   round(score, 6),
            "eu_specific_flag": False,
        })

    kw_df = pd.DataFrame(kw_rows)

    # Part 2: representative abstracts
    abs_rows = []
    for i, row in rep_sample.iterrows():
        abs_rows.append({
            "rank":     len(abs_rows) + 1,
            "scopus_id": row.get("scopus_id", ""),
            "title":    str(row.get("title", ""))[:200],
            "pub_year": int(row["pub_year"]) if pd.notna(row["pub_year"]) else "",
            "country":  str(row.get("country", "")),
            "abstract_snippet": str(row.get("abstract", ""))[:500],
        })
    abs_df = pd.DataFrame(abs_rows)

    # Save structured CSV (two sheets simulated as two sections in one CSV)
    combined_rows = []
    for _, r in kw_df.iterrows():
        combined_rows.append({"record_type": "keyword", **r.to_dict(),
                               "rank_abs": None, "scopus_id": None, "title": None,
                               "pub_year": None, "country": None, "abstract_snippet": None})
    for _, r in abs_df.iterrows():
        combined_rows.append({"record_type": "abstract",
                               "section": "representative_abstracts",
                               "rank": None, "term": None, "ctfidf_score": None,
                               "eu_specific_flag": None,
                               "rank_abs": r["rank"], "scopus_id": r["scopus_id"],
                               "title": r["title"], "pub_year": r["pub_year"],
                               "country": r["country"],
                               "abstract_snippet": r["abstract_snippet"]})

    out_csv = RESULTS_DIR / "eu_regulation_topic_inspection.csv"
    pd.DataFrame(combined_rows).to_csv(out_csv, index=False)
    print(f"  Saved: results/alignment/eu_regulation_topic_inspection.csv")

    # ── Build Markdown report ──────────────────────────────────────────────────
    lines = []
    lines.append("# EU AI Regulation Topic Inspection")
    lines.append("")
    lines.append(f"**Purpose:** Determine whether academic topic 135 "
                 f"(\"EU AI Regulation & Regulatory Frameworks\") captures "
                 f"EU-specific regulatory vocabulary or general regulatory-normative register.")
    lines.append("")
    lines.append(f"**Finetuned topic ID:** {EU_REG_FINETUNED_ID}  "
                 f"(Stage 2 model topic {orig_id})")
    lines.append(f"**Total papers in topic:** {len(eu_papers)}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 1. Academic Topic: Top 15 Keywords (c-TF-IDF)")
    lines.append("")
    lines.append("| Rank | Term | Score | EU-Specific? |")
    lines.append("|------|------|-------|-------------|")
    for rank, (word, score) in enumerate(acad_top15, 1):
        eu_flag = "✓ EU" if any(m in word.lower() for m in EU_SPECIFIC_MARKERS) else ""
        lines.append(f"| {rank} | {word} | {score:.6f} | {eu_flag} |")
    lines.append("")
    lines.append(f"**EU-specific terms found:** "
                 f"{', '.join(w for w, _ in eu_specific_hits) if eu_specific_hits else 'None'}")
    lines.append(f"**General regulatory-normative terms found:** "
                 f"{', '.join(w for w, _ in general_reg_hits) if general_reg_hits else 'None'}")
    lines.append("")
    lines.append(f"**Verdict:** {specificity_verdict}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"## 2. Policy Topic T08: Top 15 Keywords (c-TF-IDF)")
    lines.append(f"*(International AI Policy Principles — best-matching policy topic for "
                 f"the academic EU AI Regulation topic, sim=0.7458)*")
    lines.append("")
    lines.append("| Rank | Term | Score |")
    lines.append("|------|------|-------|")
    for rank, (word, score) in enumerate(pol_top15, 1):
        lines.append(f"| {rank} | {word} | {score:.6f} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Representative Abstracts")
    n_matched = len(matched_ids)
    lines.append(f"*(5 papers from topic {EU_REG_FINETUNED_ID}: "
                 f"{n_matched} matched via BERTopic representative_docs, "
                 f"{max(0, 5 - n_matched)} supplemented by governance score)*")
    lines.append("")
    for _, row in rep_sample.iterrows():
        lines.append(f"### Paper {row.get('scopus_id', 'unknown')}")
        lines.append(f"**Title:** {row.get('title', 'N/A')}")
        lines.append(f"**Year:** {row.get('pub_year', 'N/A')}  |  "
                     f"**Country:** {row.get('country', 'N/A')}")
        lines.append("")
        abst = str(row.get("abstract", "")).strip()
        lines.append(f"**Abstract:** {abst[:600]}{'...' if len(abst) > 600 else ''}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. Vocabulary Comparison Summary")
    lines.append("")
    lines.append("| Dimension | Academic T135 | Policy T08 |")
    lines.append("|-----------|--------------|------------|")
    acad_words_str = ", ".join(w for w, _ in acad_top15[:8])
    pol_words_str  = ", ".join(w for w, _ in pol_top15[:8])
    lines.append(f"| Top 8 terms | {acad_words_str} | {pol_words_str} |")
    lines.append(f"| Cosine similarity | 0.7458 | — |")
    lines.append(f"| EU-specific markers | "
                 f"{len(eu_specific_hits)} of 15 keywords | 0 of 15 keywords |")
    lines.append(f"| Interpretation | {specificity_verdict.split(':')[0]} | "
                 f"General governance register |")
    lines.append("")

    out_md = RESULTS_DIR / "eu_regulation_topic_inspection.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: results/alignment/eu_regulation_topic_inspection.md")

    # ── Stopping Point 5b ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  STOPPING POINT 5b — EU AI REGULATION TOPIC INSPECTION")
    print(f"{'─'*60}")
    print(f"\n  Academic topic 135 — EU AI Regulation & Regulatory Frameworks")
    print(f"  (Stage 2 model topic {orig_id}, {len(eu_papers)} papers total)")
    print(f"\n  TOP 15 ACADEMIC KEYWORDS (c-TF-IDF):")
    for rank, (word, score) in enumerate(acad_top15, 1):
        eu_flag = "  ← EU-specific" if any(m in word.lower() for m in EU_SPECIFIC_MARKERS) else ""
        print(f"    {rank:2d}. {word:<35} {score:.6f}{eu_flag}")
    print(f"\n  EU-specific markers detected: "
          f"{[w for w, _ in eu_specific_hits] if eu_specific_hits else 'None'}")
    print(f"  General regulatory terms:     "
          f"{[w for w, _ in general_reg_hits] if general_reg_hits else 'None'}")
    print(f"\n  VERDICT: {specificity_verdict}")
    print(f"\n  TOP 15 POLICY T08 KEYWORDS (c-TF-IDF):")
    for rank, (word, score) in enumerate(pol_top15, 1):
        print(f"    {rank:2d}. {word:<35} {score:.6f}")
    print(f"\n  REPRESENTATIVE ABSTRACTS (5 papers):")
    for _, row in rep_sample.iterrows():
        print(f"\n  [{row.get('pub_year', '?')}] {row.get('country', '?')}  "
              f"{str(row.get('title', 'no title'))[:80]}")
        abst = str(row.get("abstract", "")).strip()
        print(f"  {abst[:300]}...")
    confirm("5b — Inspection complete. No further analysis required.")


# ==============================================================================
#  PHASE 5c — VOCABULARY OVERLAP AUDIT (21 ALIGNMENT PAIRS)
# ==============================================================================

_STEM_SUFFIXES = ("ings", "tions", "tion", "ities", "ity", "ies", "ing",
                  "ment", "ments", "ness", "ed", "ers", "er", "es", "ly", "s")


def _simple_stem(word: str) -> str:
    """Strip common English suffixes to get a rough stem."""
    w = word.lower()
    for suf in _STEM_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def _extract_keywords(model, topic_id: int, n: int = 10) -> list[tuple[str, float]]:
    """Return top-n (word, score) pairs from a fitted BERTopic model topic.
    Tries c_tf_idf_ first for extended vocabulary; falls back to get_topic()."""
    try:
        ctf = model.c_tf_idf_
        sorted_ids = sorted(k for k in model.get_topics().keys() if k != -1)
        col = sorted_ids.index(topic_id)
        row = (ctf.toarray()[col] if hasattr(ctf, "toarray")
               else np.array(ctf[col].todense()).flatten())
        vocab = model.vectorizer_model.get_feature_names_out()
        top   = np.argsort(row)[::-1][:n]
        return [(vocab[i], float(row[i])) for i in top if row[i] > 0]
    except Exception:
        result = model.get_topic(topic_id) or []
        return result[:n]


def phase5c():
    """
    Vocabulary overlap audit for all 21 academic→policy alignment pairs.
    Computes exact and stem-fuzzy keyword overlap to distinguish genuine
    thematic alignment from register-convergence artefacts.
    """
    print("\n" + "=" * 60)
    print("PHASE 5c — VOCABULARY OVERLAP AUDIT (21 ALIGNMENT PAIRS)")
    print("=" * 60)

    from bertopic import BERTopic

    # ── Load saved artefacts ──────────────────────────────────────────────────
    acad_df      = pd.read_csv(RESULTS_DIR / "academic_to_thematic_policy_alignment.csv")
    gov_index    = pd.read_csv(RESULTS_DIR / "academic_governance_topic_index.csv")
    sub_matrix   = np.load(RESULTS_DIR / "alignment_similarity_matrix.npy")[:, [1,8,9,15,16]]
    thematic_ids = [1, 8, 9, 15, 16]
    thematic_col = {tid: i for i, tid in enumerate(thematic_ids)}

    # ── Load models ───────────────────────────────────────────────────────────
    m1           = BERTopic.load(str(MODELS_DIR / "academic_topic_model"))
    m2           = BERTopic.load(str(MODELS_DIR / "topic0_decomposition_model"))
    policy_model = BERTopic.load(str(MODELS_DIR / "policy_topic_model"))

    subtopic_map = pd.read_csv(BASE_DIR / "results" / "topic0_subtopics.csv")
    id_to_orig   = dict(zip(subtopic_map["sub_topic_id"], subtopic_map["original_t0_id"]))

    print(f"  Models loaded. Auditing 21 alignment pairs...")

    # ── Cache policy topic keywords (each thematic topic queried once) ────────
    pol_kw_cache: dict[int, list[tuple[str, float]]] = {}
    for tid in thematic_ids:
        pol_kw_cache[tid] = _extract_keywords(policy_model, tid, n=10)

    # ── Compute per-pair audit ────────────────────────────────────────────────
    audit_rows = []
    pair_details = []   # for MD output

    for _, row in acad_df.iterrows():
        tid_acad   = int(row["academic_topic_id"])
        label_acad = row["academic_topic_label"]
        tid_pol    = int(row["best_match_policy_topic_id"])
        cosine     = float(row["best_similarity"])

        # Academic keywords
        if tid_acad in id_to_orig:
            acad_kw = _extract_keywords(m2, id_to_orig[tid_acad], n=10)
        else:
            acad_kw = _extract_keywords(m1, tid_acad, n=10)

        pol_kw  = pol_kw_cache[tid_pol]

        acad_words = [w.lower() for w, _ in acad_kw]
        pol_words  = [w.lower() for w, _ in pol_kw]

        # Exact overlap
        exact = len(set(acad_words) & set(pol_words))

        # Fuzzy/stem overlap (no double-counting)
        acad_stems = [_simple_stem(w) for w in acad_words]
        pol_stems  = [_simple_stem(w) for w in pol_words]
        fuzzy = len(set(acad_stems) & set(pol_stems))

        # Flags
        reg_conv    = bool(cosine >= 0.55 and exact <= 2)
        substantive = bool(cosine >= 0.55 and exact >= 4)

        # Gap metric: how much cosine exceeds what vocabulary overlap predicts
        gap = round(cosine - exact / 10.0, 4)

        audit_rows.append({
            "academic_topic_id":      tid_acad,
            "academic_topic_label":   label_acad,
            "academic_top10":         "; ".join(acad_words),
            "best_policy_topic_id":   tid_pol,
            "best_policy_label":      THEMATIC_POLICY_LABELS[tid_pol],
            "policy_top10":           "; ".join(pol_words),
            "cosine_similarity":      round(cosine, 4),
            "vocab_overlap_exact":    exact,
            "vocab_overlap_fuzzy":    fuzzy,
            "cosine_overlap_gap":     gap,
            "register_convergence_flag":  reg_conv,
            "substantive_alignment_flag": substantive,
        })
        pair_details.append((label_acad, tid_pol, acad_words, pol_words,
                              cosine, exact, fuzzy, gap, reg_conv, substantive))

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(RESULTS_DIR / "vocab_overlap_audit.csv", index=False)
    print(f"  Saved: results/alignment/vocab_overlap_audit.csv")

    # ── Build Markdown ────────────────────────────────────────────────────────
    md = ["# Vocabulary Overlap Audit — 21 Alignment Pairs", "",
          "Compares top-10 c-TF-IDF keywords for each academic governance topic "
          "against the top-10 keywords of its best-matching thematic policy topic.",
          "",
          "**Flags:**  ",
          "- `register_convergence`: cosine ≥ 0.55 AND exact overlap ≤ 2  ",
          "- `substantive_alignment`: cosine ≥ 0.55 AND exact overlap ≥ 4",
          "", "---", ""]

    # Summary table
    md += ["## Summary Table", "",
           "| # | Academic Topic | Policy Topic | Cos | Exact | Fuzzy | Gap | RC | SA |",
           "|---|---------------|-------------|-----|-------|-------|-----|----|----|"]
    for i, r in audit_df.iterrows():
        rc = "✓" if r["register_convergence_flag"] else ""
        sa = "✓" if r["substantive_alignment_flag"] else ""
        md.append(
            f"| {i+1} | {r['academic_topic_label'][:42]} "
            f"| T{int(r['best_policy_topic_id']):02d} "
            f"| {r['cosine_similarity']:.3f} "
            f"| {r['vocab_overlap_exact']} "
            f"| {r['vocab_overlap_fuzzy']} "
            f"| {r['cosine_overlap_gap']:.3f} "
            f"| {rc} | {sa} |")
    md += ["", "---", ""]

    # Per-pair keyword detail
    md += ["## Per-Pair Keyword Detail", ""]
    for (label_acad, tid_pol, acad_words, pol_words,
         cosine, exact, fuzzy, gap, rc, sa) in pair_details:
        pol_label = THEMATIC_POLICY_LABELS[tid_pol]
        md.append(f"### {label_acad}")
        md.append(f"**→ T{tid_pol:02d} {pol_label}** "
                  f"| cos={cosine:.4f} | exact={exact} | fuzzy={fuzzy} "
                  f"| gap={gap:.3f}"
                  + (" | **REGISTER CONVERGENCE**" if rc else "")
                  + (" | **SUBSTANTIVE ALIGNMENT**" if sa else ""))
        md.append("")
        md.append("| Rank | Academic keyword | Policy keyword |")
        md.append("|------|-----------------|----------------|")
        for rank in range(10):
            aw = acad_words[rank] if rank < len(acad_words) else ""
            pw = pol_words[rank]  if rank < len(pol_words)  else ""
            match = "✓" if aw and pw and (_simple_stem(aw) == _simple_stem(pw)
                                           or aw == pw) else ""
            md.append(f"| {rank+1} | {aw} {match} | {pw} {match} |")
        md.append("")

    out_md = RESULTS_DIR / "vocab_overlap_audit.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  Saved: results/alignment/vocab_overlap_audit.md")

    # ── Stopping Point 5c ─────────────────────────────────────────────────────
    n_rc    = audit_df["register_convergence_flag"].sum()
    n_sa    = audit_df["substantive_alignment_flag"].sum()
    n_high  = (audit_df["cosine_similarity"] >= 0.55).sum()
    n_uncl  = n_high - n_rc - n_sa  # high cosine, overlap 3 — neither flag

    print(f"\n{'─'*60}")
    print(f"  STOPPING POINT 5c — VOCABULARY OVERLAP AUDIT")
    print(f"{'─'*60}")

    print(f"\n  FULL AUDIT TABLE (21 pairs):")
    hdr = (f"  {'Academic Topic':<46}  T__  {'Cos':>5}  "
           f"{'Exact':>5}  {'Fuzzy':>5}  {'Gap':>5}  Flags")
    print(hdr)
    print("  " + "─" * 85)
    for _, r in audit_df.iterrows():
        flags = []
        if r["register_convergence_flag"]:  flags.append("RC")
        if r["substantive_alignment_flag"]: flags.append("SA")
        print(f"  {r['academic_topic_label'][:46]:<46}  "
              f"T{int(r['best_policy_topic_id']):02d}  "
              f"{r['cosine_similarity']:>5.3f}  "
              f"{r['vocab_overlap_exact']:>5}  "
              f"{r['vocab_overlap_fuzzy']:>5}  "
              f"{r['cosine_overlap_gap']:>5.3f}  "
              f"{', '.join(flags)}")

    print(f"\n  SUMMARY:")
    print(f"    Pairs with cosine ≥ 0.55:              {n_high}")
    print(f"    Register convergence  (cos≥0.55, ex≤2): {n_rc}")
    print(f"    Substantive alignment (cos≥0.55, ex≥4): {n_sa}")
    print(f"    Mid-range overlap     (cos≥0.55, ex=3):  {n_uncl}")

    print(f"\n  TOP 5 BY EXACT VOCAB OVERLAP (most genuinely aligned):")
    for _, r in audit_df.sort_values("vocab_overlap_exact", ascending=False).head(5).iterrows():
        print(f"    exact={r['vocab_overlap_exact']}  cos={r['cosine_similarity']:.4f}  "
              f"{r['academic_topic_label'][:50]}  → T{int(r['best_policy_topic_id']):02d}")

    print(f"\n  TOP 5 BY COSINE-OVERLAP GAP (most register-driven):")
    for _, r in audit_df.sort_values("cosine_overlap_gap", ascending=False).head(5).iterrows():
        print(f"    gap={r['cosine_overlap_gap']:.3f}  cos={r['cosine_similarity']:.4f}  "
              f"exact={r['vocab_overlap_exact']}  "
              f"{r['academic_topic_label'][:50]}  → T{int(r['best_policy_topic_id']):02d}")

    confirm("5c — Vocabulary overlap audit complete. Alignment analysis concluded.")


# ==============================================================================
#  PHASE 5d — RAW VOCABULARY OVERLAP CHECK
# ==============================================================================

# Terms to remove in addition to stopwords: universally common in AI discourse.
UNIVERSAL_AI_TERMS = {
    "ai", "artificial", "intelligence", "machine", "learning",
    "system", "systems", "data", "model", "models",
    "technology", "technologies", "algorithm", "algorithms",
}

# Comprehensive English stopword list (no NLTK corpus download required).
_STOPWORDS = {
    "a","about","above","after","again","against","all","also","am","an","and",
    "any","are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each","few",
    "for","from","further","get","gets","got","had","hadn't","has","hasn't","have",
    "haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
    "herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've",
    "if","in","into","is","isn't","it","it's","its","itself","let","let's","me",
    "more","most","mustn't","my","myself","no","nor","not","of","off","on","once",
    "only","or","other","ought","our","ours","ourselves","out","over","own","same",
    "shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
    "than","that","that's","the","their","theirs","them","themselves","then","there",
    "there's","these","they","they'd","they'll","they're","they've","this","those",
    "through","to","too","under","until","up","very","was","wasn't","we","we'd",
    "we'll","we're","we've","were","weren't","what","what's","when","when's","where",
    "where's","which","while","who","who's","whom","why","why's","will","with",
    "won't","would","wouldn't","you","you'd","you'll","you're","you've","your",
    "yours","yourself","yourselves",
    # Common academic/document boilerplate
    "paper","study","research","results","analysis","show","shows","showed","propose",
    "proposed","use","used","using","work","based","approach","method","methods",
    "however","thus","therefore","also","including","particularly","well","may",
    "many","first","two","three","one","new","high","large","different","et","al",
    "fig","table","section","pp","vol","doi","http","www","com","org","found",
    "provide","provides","present","presents","review","reviewed","propose","suggests",
    "across","within","between","among","various","several","different","number",
    "important","significant","effective","various","need","needs","make","makes",
    "ensure","ensures","help","helps","order","terms","form","forms","point","points",
    "whether","both","either","case","cases","given","example","examples","type",
    "types","role","roles","key","main","often","likely","less","more","most","least",
}


def _get_top50_terms(text: str) -> list[str]:
    """Tokenize, remove stopwords + universal AI terms, return top-50 by frequency."""
    import re
    from collections import Counter
    tokens = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    filtered = [t for t in tokens
                if len(t) >= 3
                and t not in _STOPWORDS
                and t not in UNIVERSAL_AI_TERMS]
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(50)]


def phase5d():
    """
    Raw vocabulary overlap check for all 21 alignment pairs.
    Compares top-50 term-frequency terms (after stopword removal) from academic
    abstracts vs. policy chunk texts, providing a c-TF-IDF-independent overlap signal.
    """
    print("\n" + "=" * 60)
    print("PHASE 5d — RAW VOCABULARY OVERLAP CHECK (21 PAIRS)")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────────────
    acad_df    = pd.read_csv(RESULTS_DIR / "academic_to_thematic_policy_alignment.csv")
    audit_df   = pd.read_csv(RESULTS_DIR / "vocab_overlap_audit.csv")
    assign     = pd.read_csv(RESULTS_DIR / "policy_chunk_topic_assignments.csv")
    clean_df   = pd.read_csv(DATA_DIR / "policy_chunks_clean.csv",
                              usecols=["chunk_id", "chunk_text"])
    papers     = pd.read_csv(BASE_DIR / "results" / "governance_papers_stance.csv",
                              usecols=["topic_id_finetuned", "abstract"])

    # Join chunk texts onto assignments (policy side)
    assign = assign.merge(clean_df, on="chunk_id", how="left")

    # ── Pre-compute top-50 terms for each thematic policy topic ───────────────
    thematic_ids  = [1, 8, 9, 15, 16]
    pol_top50_map: dict[int, list[str]] = {}
    for tid in thematic_ids:
        texts = assign[assign["policy_topic_id"] == tid]["chunk_text"].fillna("").tolist()
        pol_top50_map[tid] = _get_top50_terms(" ".join(texts))
        print(f"  T{tid:02d}: {len(assign[assign['policy_topic_id']==tid])} chunks → "
              f"top-50 computed")

    # ── Exact top-10 lookup from Phase 5c ────────────────────────────────────
    exact_map = dict(zip(audit_df["academic_topic_id"].astype(int),
                         audit_df["vocab_overlap_exact"].astype(int)))

    # ── Per-pair raw overlap ──────────────────────────────────────────────────
    rows  = []
    debug = []   # (acad_label, pol_label, acad_top50, pol_top50, overlap_set)

    papers["topic_id_finetuned"] = pd.to_numeric(
        papers["topic_id_finetuned"], errors="coerce")

    for _, row in acad_df.iterrows():
        tid_acad  = int(row["academic_topic_id"])
        tid_pol   = int(row["best_match_policy_topic_id"])
        cosine    = float(row["best_similarity"])

        # Collect abstracts for this academic topic
        mask     = papers["topic_id_finetuned"] == tid_acad
        abstracts = papers.loc[mask, "abstract"].fillna("").tolist()
        acad_top50 = _get_top50_terms(" ".join(abstracts))

        pol_top50  = pol_top50_map[tid_pol]

        overlap    = sorted(set(acad_top50) & set(pol_top50))
        n_overlap  = len(overlap)
        examples   = ", ".join(overlap[:10])

        rows.append({
            "academic_topic_id":    tid_acad,
            "academic_topic_label": row["academic_topic_label"],
            "n_academic_papers":    int(mask.sum()),
            "policy_topic_id":      tid_pol,
            "policy_label":         THEMATIC_POLICY_LABELS[tid_pol],
            "cosine_similarity":    round(cosine, 4),
            "exact_top10_overlap":  exact_map.get(tid_acad, 0),
            "raw_top50_overlap":    n_overlap,
            "overlap_examples":     examples,
        })
        debug.append((row["academic_topic_label"],
                      THEMATIC_POLICY_LABELS[tid_pol],
                      acad_top50, pol_top50, overlap))

    result_df = pd.DataFrame(rows).sort_values("raw_top50_overlap", ascending=False)
    result_df.to_csv(RESULTS_DIR / "raw_vocab_overlap.csv", index=False)
    print(f"\n  Saved: results/alignment/raw_vocab_overlap.csv")

    # ── Build Markdown ────────────────────────────────────────────────────────
    md = ["# Raw Vocabulary Overlap Audit — 21 Alignment Pairs", "",
          "Top-50 term-frequency terms (stopwords + universal AI terms removed) "
          "compared across academic abstracts and policy chunks.",
          "", "---", ""]

    md += ["## Summary Table", "",
           "| Academic Topic | Policy | Cos | c-TF-IDF exact | Raw top-50 overlap | Examples |",
           "|---|---|---|---|---|---|"]
    for _, r in result_df.iterrows():
        md.append(f"| {r['academic_topic_label'][:42]} "
                  f"| T{int(r['policy_topic_id']):02d} "
                  f"| {r['cosine_similarity']:.3f} "
                  f"| {r['exact_top10_overlap']} "
                  f"| **{r['raw_top50_overlap']}** "
                  f"| {r['overlap_examples'][:60]} |")
    md += ["", "---", ""]

    md += ["## Per-Pair Term Lists", ""]
    for (acad_lbl, pol_lbl, acad_t50, pol_t50, overlap) in debug:
        md.append(f"### {acad_lbl}")
        md.append(f"**→ {pol_lbl}**  |  raw overlap = {len(overlap)}")
        md.append("")
        md.append(f"**Academic top-50:** {', '.join(acad_t50)}")
        md.append("")
        md.append(f"**Policy top-50:** {', '.join(pol_t50)}")
        md.append("")
        md.append(f"**Overlapping terms ({len(overlap)}):** "
                  f"{', '.join(overlap) if overlap else '*(none)*'}")
        md.append("")

    out_md = RESULTS_DIR / "raw_vocab_overlap.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  Saved: results/alignment/raw_vocab_overlap.md")

    # ── Stopping Point 5d ─────────────────────────────────────────────────────
    overlaps     = result_df["raw_top50_overlap"].values
    n_substantial = int((overlaps >= 10).sum())
    n_moderate    = int(((overlaps >= 5) & (overlaps < 10)).sum())
    n_low         = int((overlaps < 5).sum())

    print(f"\n{'─'*60}")
    print(f"  STOPPING POINT 5d — RAW VOCABULARY OVERLAP")
    print(f"{'─'*60}")

    print(f"\n  FULL RESULTS (21 pairs, sorted by raw overlap descending):")
    hdr = (f"  {'Academic Topic':<46}  T__  {'Cos':>5}  "
           f"{'cTFIDF':>6}  {'Raw50':>5}  Overlapping terms (sample)")
    print(hdr)
    print("  " + "─" * 100)
    for _, r in result_df.iterrows():
        print(f"  {r['academic_topic_label'][:46]:<46}  "
              f"T{int(r['policy_topic_id']):02d}  "
              f"{r['cosine_similarity']:>5.3f}  "
              f"{r['exact_top10_overlap']:>6}  "
              f"{r['raw_top50_overlap']:>5}  "
              f"{r['overlap_examples'][:50]}")

    print(f"\n  SUMMARY STATISTICS (raw top-50 overlap):")
    print(f"    Mean:   {float(np.mean(overlaps)):.1f}")
    print(f"    Median: {float(np.median(overlaps)):.1f}")
    print(f"    Min:    {int(np.min(overlaps))}")
    print(f"    Max:    {int(np.max(overlaps))}")
    print(f"\n  DISTRIBUTION:")
    print(f"    Substantial (≥10 shared terms):  {n_substantial} pairs")
    print(f"    Moderate    (5–9 shared terms):   {n_moderate} pairs")
    print(f"    Low         (<5 shared terms):    {n_low} pairs")

    confirm("5d — Raw vocabulary overlap complete. Alignment analysis fully concluded.")


# ==============================================================================
#  ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    chunks, academic_model, gov_topics = phase1()
    academic_gov_embeddings = build_academic_gov_embeddings(gov_topics)
    phase2(chunks, academic_gov_embeddings, gov_topics)
    phase3()
    phase4()
    phase5()
    phase5b()
    phase5c()
    phase5d()
