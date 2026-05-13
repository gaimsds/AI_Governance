# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

This repo maps AI governance discourse across 41,067 Scopus publications (2015–2025) and 35 national AI strategy documents. The core pipeline runs a two-stage BERTopic model, zero-shot stance classification via `facebook/bart-large-mnli`, and a symmetric cross-corpus alignment analysis. Findings are in `reports/6. Uneven Science–Policy Translation Shapes Global AI Governance.pdf`.

## Environment Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. GPU (MPS or CUDA) strongly recommended for `4d_sentiment_analysis.py` (~7 min on Apple Silicon).

Scopus collection requires a `.env` file in the project root (gitignored):
```
SCOPUS_API_KEY=your_key_here
```
Also requires GWU institutional network access or VPN.

## Running Scripts

Scripts are numbered by execution order. Run from the project root:

```bash
# Core pipeline (Steps 1–4)
python src/1_data_collection_scopus.py
python src/1_data_collection_policyframeworks.py
python src/2_data_cleaning_scopus.py
python src/2_policy_text_extraction.py
python src/2b_institution_geocoding_scopus.py
python src/2c_coauthorship_edges_scopus.py
python src/3_modelling.py
python src/4_model_finetuning.py
python src/4b_update_governance_scores.py
python src/4c_data_integrity_fix.py
python src/4d_sentiment_analysis.py

# Policy stance temporal analysis
python src/7_policy_stance_temporal.py

# Symmetric alignment pipeline (Steps 8–11, run in order)
python src/8_prepare_alignment_inputs.py
python src/10_clean_policy_corpus.py      # must run before 9
python src/9_symmetric_alignment.py
python src/11_validation_sample.py

# Dashboard
streamlit run src/5_streamlit_dashboard.py  # http://localhost:8501

# Visualizations (rebuild notebook only if structure needs regenerating)
python src/6_create_visuals.py
# Then open src/6_exploratory_visuals.ipynb interactively
```

Pre-fitted models are in `models/` — Steps 1–4 can be skipped by loading them directly.

## Architecture

### Pipeline Stages

| Script | Stage | Key output |
|--------|-------|------------|
| `1_*` | Data collection | Raw CSVs in `data_raw/`, policy PDFs |
| `2_*` | Cleaning & geocoding | `data_clean/scopus_cleaned.xlsx`, institution nodes/edges, policy chunks |
| `3_modelling.py` | BERTopic Stage 1 | 46 topics, `results/scopus_topic_assignments.csv` |
| `4_model_finetuning.py` | BERTopic Stage 2 | 133 topics (88 sub-topics from catch-all Topic 0) |
| `4b_update_governance_scores.py` | Expert scoring | `results/governance_topics.csv` (21 topics ≥ 0.5) |
| `4c_data_integrity_fix.py` | Validation | `results/governance_papers.csv` (3,186 papers) |
| `4d_sentiment_analysis.py` | Stance classification | `results/governance_papers_stance.csv` |
| `7_policy_stance_temporal.py` | Policy stance over time | `results/policy_stance_temporal.csv` (1,399 chunks) |
| `8_prepare_alignment_inputs.py` | Alignment input prep | Policy chunks CSV, academic embeddings |
| `10_clean_policy_corpus.py` | Policy corpus cleaning | `data/policy_chunks_clean.csv` (1,086 chunks, 30 docs) |
| `9_symmetric_alignment.py` | Symmetric alignment | `results/alignment/` — full analysis outputs |
| `11_validation_sample.py` | Validation sample | `results/validation/` — 400-doc stance evaluation set |

### Two-Stage BERTopic Design

Stage 1: `all-MiniLM-L6-v2` embeddings → UMAP(n_components=10, n_neighbors=15) → HDBSCAN(min_cluster_size=50) → 46 topics. 53% of papers fell into catch-all Topic 0. Stage 2 re-clusters Topic 0 papers with min_cluster_size=15, producing 88 sub-topics merged back to 133 total. Stage 2 topic IDs are stored as `finetuned_id = original_topic0_subid + 100`.

### Governance Scoring

Topics are manually scored 0.0–1.0 against OECD AI Principles, EU AI Act risk categories, and UNESCO Ethics Recommendation. Topics scoring ≥ 0.5 are governance-relevant. Scoring is finalized in `4b_update_governance_scores.py`.

### Policy Corpus Cleaning (`10_clean_policy_corpus.py`)

Drops non-English source documents via an explicit `DOC_LANGUAGE_MAP`: France (fr), Brazil (pt), Colombia (es), Mexico (es), Chile (machine-translated). Retains official English translations for China (CSET), South Korea (CSET), and Japan (Cabinet Office). Applies six OCR artifact heuristics per chunk:
- `non_alpha_ratio > 0.20`
- `avg_word_len < 2.5` or `> 12`
- `numeric_token_ratio > 0.25`
- `reversed_text_score > 0.15` (reversed common English words)
- UTF-8 replacement character rate ≥ 2%
- `standalone_al_ratio > 0.04` (catches I→l font encoding artifacts in ASEAN PDFs)

### Symmetric Alignment Design (`9_symmetric_alignment.py`)

Policy BERTopic: `all-MiniLM-L6-v2` → UMAP(n_components=5, n_neighbors=5, metric="cosine") → HDBSCAN(min_cluster_size=12, metric="euclidean", cluster_selection_method="eom") → 20 policy topics. Adaptive stopping rules: <6 topics → raise n_components to 15; >25 → raise min_cluster_size to 12; outlier rate >50% → raise n_neighbors to 15.

Topics are classified as thematic (n_contributing_docs ≥ 3 AND dominant_doc_share ≤ 0.60), document-specific, or artifact. `CLASSIFICATION_OVERRIDES = {17: "document_specific"}` and `ARTIFACT_TOPIC_IDS = {11}` are baked in as constants to survive reruns.

Cross-corpus similarity is a 21×5 cosine similarity matrix (21 academic governance topics × 5 thematic policy topics). Key diagnostics:
- **Fan-out**: how many academic topics each policy topic absorbs as best match
- **Vocabulary divergence**: c-TF-IDF keyword overlap for expected high-similarity pairs
- **Register convergence flag**: cosine ≥ 0.55 AND exact c-TF-IDF overlap ≤ 2
- **Substantive alignment flag**: cosine ≥ 0.55 AND exact c-TF-IDF overlap ≥ 4
- **Raw vocabulary overlap**: top-50 frequency terms after removing universal AI terms and stopwords

### Validation Sample (`11_validation_sample.py`)

400 documents across 4 strata (100 each), sampled with seed=42:
- **A**: Academic, 2015–2021
- **B**: Academic, 2022–2025
- **C**: Policy, 2017–2021
- **D**: Policy, 2022–2025

Each stratum: 35 risk_focused, 35 opportunity_focused, 30 balanced. Shuffled across strata before assigning coding IDs V0001–V0400.

### Paths Convention

All scripts use `Path(__file__).parent.parent` as `BASE_DIR`. Run as `python src/script.py` from the project root.

## Key Data Files

- `data_clean/scopus_cleaned.xlsx` — 41,067 cleaned papers
- `data/policy_chunks_clean.csv` — 1,086 clean policy chunks (30 docs, post-cleaning)
- `data/policy_chunks.csv` — 1,399 raw policy chunks (35 docs, pre-cleaning)
- `results/governance_papers_stance.csv` — 3,186 governance papers with stance labels
- `results/policy_stance_temporal.csv` — 1,399 policy chunks with stance predictions
- `results/alignment/academic_to_thematic_policy_alignment.csv` — best/second-best policy match per academic topic
- `results/alignment/vocab_overlap_audit.csv` — c-TF-IDF overlap with convergence flags
- `results/alignment/raw_vocab_overlap.csv` — raw top-50 vocabulary overlap
- `results/alignment/region_to_academic_similarity.csv` — per-region alignment scores
- `results/validation/validation_sample.csv` — 400-doc sample with metadata (internal)
- `results/validation/validation_sample_for_coders.csv` — coder-facing file (coding_id, text, empty label/notes)

## Notes

- `1_data_collection_openalex.py` and `2_data_cleaning_openalex.py` are exploratory and not used in the final analysis.
- `src/lib/` contains JavaScript/CSS for Streamlit network visualizations (vis.js, tom-select), not Python utilities.
- `models/academic_topic_model` and `models/policy_topic_model` are gitignored (generated artifacts).
- `data/` files are gitignored (generated from `data_clean/` sources).
- `results/alignment/*.npy` embedding arrays are gitignored; recomputed by `9_symmetric_alignment.py`.
- The dashboard Overview tab shows "7.7%" — known discrepancy; correct figure is "7.8%" per the paper.
- `gensim` is absent from `requirements.txt`; coherence scoring in `3_modelling.py` skips gracefully if not installed.
