# Uneven Science–Policy Translation Shapes Global AI Governance


![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Academic%20Use-green)

---

## Overview

This project maps how AI governance is discussed, framed, and translated (or not) into policy across 138 countries. Using a two-stage BERTopic pipeline and zero-shot stance classification, we analyze 41,067 peer-reviewed AI publications (Scopus, 2015–2025) and 35 national AI strategy documents to identify dominant governance themes, regional specialization patterns, and knowledge-policy gaps.

**Key findings:**
- Only 7.8% of AI research addresses governance — a growing but still minority concern
- Post-ChatGPT, governance framing shifted from risk-dominant to opportunity-dominant (40.0% vs 29.4%)
- 11 governance themes present in academic discourse are absent from all national AI strategies
- EU AI Act vocabulary structures Global South policy documents, while China, Japan, and Australia maintain independent frameworks

---

## Research Objectives

| RO | Focus | Question |
|----|-------|----------|
| **RO1** | Themes & Framing | What are the dominant AI governance themes, how have they evolved, and are they framed as risks, opportunities, or balanced concerns? |
| **RO2** | Spatial Distribution | How is AI governance discourse distributed across world regions and contributing institutions, and what patterns of specialization exist? |
| **RO3** | Academic–Policy Alignment | How do academic and policy corpora compare in governance topic coverage, and where do significant knowledge-policy gaps exist? |

---

## Authors

| Name | Institution | Program |
|------|-------------|---------|
| Tambudzai G. Charumbira | George Washington University, CCAS | M.S. Data Science |
| Joshua Gray | George Washington University, CCAS | M.S. Data Science |

**Capstone Project — Spring 2026**
Supervised by: Professor Junjun Yin

---

## Pipeline Overview

```
Scopus Corpus (41,067 papers)          Policy Corpus (35 documents)
        │                                        │
        ▼                                        ▼
  Embedding (MiniLM-L6-v2)              Chunking (500-word segments)
        │                                        │
        ▼                                        │
  BERTopic Stage 1 → 46 topics                   │
        │                                        │
        ▼                                        │
  Stage 2 Decomposition → 133 topics             │
        │                                        │
        ▼                                        │
  Governance Scoring (OECD/EU/UNESCO)            │
        │                                        │
        ▼                                        ▼
  21 Governance Topics ◄──── Cross-Corpus Alignment ────►
        │
        ▼
  Zero-Shot Stance Classification (BART-large-MNLI)
        │
        ▼
  Geographic Attribution & Co-authorship Networks
```

---

## Project Structure

```
AI-Research-Analysis-Capstone/
│
├── src/                                    # All Python scripts (numbered by execution order)
│   ├── lib/                                # Shared utility functions and helper modules
│   ├── 1_data_collection_scopus.py         # Collect papers via Scopus API
│   ├── 1_data_collection_openalex.py       # Collect papers via OpenAlex API (exploratory)
│   ├── 1_data_collection_policyframeworks.py # Download and extract policy documents
│   ├── 2_data_cleaning_scopus.py           # Clean, deduplicate, and standardize Scopus data
│   ├── 2_data_cleaning_openalex.py         # Clean OpenAlex data (exploratory)
│   ├── 2_policy_text_extraction.py         # Extract and chunk policy documents
│   ├── 2b_institution_geocoding_scopus.py  # Geocode 40,968 institutions via Nominatim
│   ├── 2c_coauthorship_edges_scopus.py     # Build institution and country co-authorship networks
│   ├── 3_modelling.py                      # BERTopic Stage 1: embedding + topic discovery
│   ├── 4_model_finetuning.py               # Hyperparameter tuning + Stage 2 decomposition
│   ├── 4b_update_governance_scores.py      # Expert governance relevance scoring (0–1 scale)
│   ├── 4c_data_integrity_fix.py            # Data validation and integrity checks
│   ├── 4d_sentiment_analysis.py            # Zero-shot stance classification (BART-large-MNLI)
│   ├── 5_streamlit_dashboard.py            # Interactive visualization dashboard
│   ├── 6_create_visuals.py                 # Generate all publication-ready charts (Altair)
│   ├── 6_exploratory_visuals.ipynb         # Exploratory visual analysis notebook
│   ├── sample_api_test.py                  # API connection testing utility
│   └── wos_setup.py                        # Web of Science setup (not used in final analysis)
│
├── data_raw/                               # Raw data as collected (never modified)
│   ├── policy_frameworks/                  # Original policy document PDFs
│   ├── scopus_pre_chatgpt_raw.csv          # Scopus pre-ChatGPT collection (2015–2021)
│   ├── scopus_post_chatgpt_raw.csv         # Scopus post-ChatGPT collection (2022–2025)
│   ├── scopus_combined.csv                 # Combined raw Scopus export
│   ├── scopus_collection.log               # API collection log
│   ├── scopus_geo_summary.csv              # Geographic summary from collection
│   └── openalex_institution_geo.parquet    # OpenAlex institution data (exploratory)
│
├── data_clean/                             # Processed datasets ready for analysis
│   ├── VOSViewer/                          # VOSviewer input files for network analysis
│   ├── scopus_cleaned.csv                  # Final cleaned academic corpus (n = 41,067)
│   ├── scopus_abstracts_nlp.csv            # Abstracts prepared for NLP pipeline
│   ├── scopus_institutions.csv             # Geocoded institution list (n = 40,968)
│   ├── scopus_country_nodes.csv            # Country-level node attributes (n = 138)
│   ├── scopus_country_edges.csv            # Country co-authorship pairs (n = 2,410)
│   ├── scopus_institution_edges.csv        # Institution co-authorship pairs (n = 116,351)
│   ├── policy_corpus.csv                   # Processed policy corpus (n = 35 documents)
│   ├── policy_corpus_english_only.csv      # English-language policy subset
│   ├── geocoding_cache.json                # Nominatim geocoding cache
│   ├── geocoding_failures.csv              # Unresolved institutions (n = 18)
│   └── *.log, *.txt                        # Processing logs and reports
│
├── models/                                 # Saved BERTopic models and embeddings
│   ├── final_bertopic_model/               # Final fitted model (Stage 1)
│   ├── topic0_decomposition_model/         # Stage 2 decomposition model
│   ├── M1_Base_model/                      # Benchmark: all-MiniLM-L6-v2
│   ├── M2_Quality_model/                   # Benchmark: all-mpnet-base-v2
│   └── M3_Multilingual_model/              # Benchmark: paraphrase-multilingual-MiniLM-L12-v2
│
├── results/                                # Analysis outputs
│   ├── topics_overview.csv                 # 46 Stage 1 topics
│   ├── topic0_subtopics.csv                # 88 Stage 2 sub-topics
│   ├── topics_finetuned.csv                # Final 133 topics
│   ├── governance_topics.csv               # 21 governance-relevant topics with scores
│   ├── governance_papers.csv               # 3,186 governance papers
│   ├── governance_papers_stance.csv        # Governance papers with stance labels
│   ├── model_comparison.csv                # Embedding model benchmark results
│   ├── hyperparameter_results.csv          # UMAP/HDBSCAN tuning results
│   ├── stance_by_topic.csv                 # Stance distribution per topic
│   ├── stance_by_region.csv                # Stance distribution per region
│   ├── stance_by_period.csv                # Pre/post-ChatGPT stance shift
│   ├── cross_corpus_alignment_v2.csv       # Academic–policy topic alignment
│   ├── policy_document_topics_v2.csv       # Policy document topic assignments
│   ├── period_governance_distribution.csv  # Temporal governance analysis
│   ├── region_governance_distribution.csv  # Regional governance analysis
│   └── *.log, *.txt                        # Processing logs and summaries
│
├── charts/                                 # All visualizations (numbered)
│   ├── 1–19. [numbered charts].png         # Publication-ready figures
│   ├── vosviewer_*.png                     # VOSviewer network visualizations
│   └── coauthorship_*.png                  # Co-authorship network maps
│
├── reports/                                # Written deliverables
│   ├── 1. Capstone Proposal.docx
│   ├── 2. Literature Review.docx
│   ├── 3. CCAS Showcase Abstract.docx
│   ├── 4. CCAS Showcase Poster.pptx
│   └── 5. DSA Capstone Presentation.pptx
│
├── .env                                    # API keys (gitignored)
├── .gitignore
├── collection_log.txt                      # Data collection audit trail
├── requirements.txt                        # Python dependencies
└── README.md                              # This file
```

---

## Data Sources

| Source | Coverage | Role in Project |
|--------|----------|-----------------|
| **Scopus** | 78M+ works | Primary academic corpus (41,067 AI publications, 2015–2025) |
| **OECD AI Policy Observatory** | National AI strategies | Primary policy corpus source |
| **Official government portals** | National AI strategies | Supplementary policy documents |

All data collection is for **non-commercial academic research purposes only**, in compliance with Elsevier's terms of use.

### Academic Corpus
- **41,067** peer-reviewed AI publications from Scopus
- **138** countries represented
- Pre-ChatGPT (2015–2021): n = 8,088
- Post-ChatGPT (2022–2025): n = 32,979
- Records with <50 abstract words excluded after deduplication

### Policy Corpus
- **35** national and international AI strategy documents
- Regions: North America, Europe, Asia-Pacific, Latin America, Africa & Middle East + international bodies (OECD, UNESCO, EU)
- Chunked into 500-word segments with 50-word overlap → **1,557** chunks

---

## Methodology

| Step | Method | Tool/Model | Output |
|------|--------|------------|--------|
| 1. Embedding | Sentence transformation | all-MiniLM-L6-v2 | 384-dim vectors for 41,067 abstracts |
| 2. Topic modeling | BERTopic Stage 1 | UMAP + HDBSCAN | 46 topics (53% in catch-all) |
| 3. Decomposition | BERTopic Stage 2 | HDBSCAN (min_cluster=15) | 88 sub-topics → 133 total |
| 4. Governance scoring | Expert coding | OECD/EU AI Act/UNESCO | 21 topics, 3,186 papers (7.8%) |
| 5. Stance classification | Zero-shot NLI | facebook/bart-large-mnli | Risk / Opportunity / Balanced |
| 6. Geographic attribution | Geocoding + networks | OpenStreetMap Nominatim | 40,968 institutions (99.96%) |
| 7. Cross-corpus alignment | Keyword matching | Custom pipeline | 10 shared, 11 academic-only |

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- GWU institutional network access or VPN (required for Scopus API)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure API Keys

Create a `.env` file in the project root (gitignored):

```
SCOPUS_API_KEY=your_scopus_key_here
```

---

## Reproducing Results

Scripts are numbered in execution order:

```bash
# 1. Data collection
python src/1_data_collection_scopus.py
python src/1_data_collection_policyframeworks.py

# 2. Data cleaning and geographic processing
python src/2_data_cleaning_scopus.py
python src/2_policy_text_extraction.py
python src/2b_institution_geocoding_scopus.py
python src/2c_coauthorship_edges_scopus.py

# 3. Topic modeling
python src/3_modelling.py

# 4. Fine-tuning, governance scoring, and stance classification
python src/4_model_finetuning.py
python src/4b_update_governance_scores.py
python src/4c_data_integrity_fix.py
python src/4d_sentiment_analysis.py

# 5. Dashboard
streamlit run src/5_streamlit_dashboard.py

# 6. Visualizations
python src/6_create_visuals.py
```

Pre-fitted models are saved in `models/` and can be loaded directly to skip Steps 1–4.

---

## Key References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *JMLR*, 3, 993–1022.
- Bradford, A. (2020). *The Brussels Effect: How the EU Rules the World*. Oxford University Press.
- Corrêa, N. K. et al. (2023). Worldwide AI ethics: A review of 200 guidelines. *Patterns*, 4(10).
- Egger, R., & Yu, J. (2022). Topic modeling comparison. *Frontiers in Sociology*, 7, 886498.
- European Parliament. (2024). Regulation (EU) 2024/1689 (EU AI Act).
- Grootendorst, M. (2022). BERTopic: Neural topic modeling. arXiv:2203.05794.
- Jobin, A., Ienca, M., & Vayena, E. (2019). The global landscape of AI ethics guidelines. *Nature Machine Intelligence*, 1, 389–399.
- Lewis, M. et al. (2020). BART: Denoising sequence-to-sequence pre-training. *ACL 2020*, 7871–7880.
- OECD. (2019). Recommendation of the Council on Artificial Intelligence.
- UNESCO. (2021). Recommendation on the Ethics of Artificial Intelligence.
- Van Eck, N. J., & Waltman, L. (2010). Software survey: VOSviewer. *Scientometrics*, 84(2), 523–538.
- Wang, W. et al. (2020). MiniLM: Deep self-attention distillation. *NeurIPS 2020*.
- Yin, W., Hay, J., & Roth, D. (2019). Benchmarking zero-shot text classification. *EMNLP 2019*.

---

## License

This project is for academic research purposes only. Data collected via the Scopus API is subject to Elsevier's terms of use and may not be redistributed.

---

*Capstone completed: Spring 2026 | George Washington University, Columbian College of Arts and Sciences*
