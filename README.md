# Global AI Governance Narratives
### A Spatial NLP Analysis of Academic and Policy Discourse

![Status](https://img.shields.io/badge/Status-Work%20In%20Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Academic%20Use-green)

---

## 📌 Overview

This project investigates how artificial intelligence governance is discussed across academic literature and policy documents globally. Using Natural Language Processing (NLP) and bibliometric analysis, we map the geographic and temporal distribution of AI governance discourse — examining how narratives shifted before and after the emergence of large-scale AI systems like ChatGPT.

The study draws on a dual corpus of peer-reviewed academic papers and formal policy documents, spanning **2015 to 2025**, and applies topic modelling and spatial analysis to identify regional differences in how governments, researchers, and institutions frame AI governance.

---

## 🎯 Research Goals

- **Goal 1:** Collect and clean a large-scale corpus of AI governance literature from multiple scholarly databases
- **Goal 2:** Apply topic modelling (BERTopic / LDA) to identify dominant themes in AI governance discourse
- **Goal 3:** Perform geographic analysis using author affiliation data to map where AI governance research originates
- **Goal 4:** Conduct temporal analysis comparing pre-ChatGPT (2015–2021) vs post-ChatGPT (2022–2025) discourse
- **Goal 5:** Visualise findings through an interactive dashboard

---

## 👥 Authors

| Name                 | Institution                        | Program               |
|----------------------|------------------------------------|-----------------------|
| Tambudzai Charumbira | George Washington University, CCAS | Ms Data Science       |
| Joshua Gray          | George Washington University, CCAS | Ms Data Science  |

**Capstone Project — Spring 2026**  
Supervised by: Professor Junjun Yin

---

## 🗂️ Project Structure

```
AI-Research-Analysis-Capstone/
│
├── code/                                 # All Python scripts (numbered in order of execution)
│   ├── 1_data_collection_openalex.py     # Collect papers via OpenAlex API (free, open)
│   ├── 1_data_collection_scopus.py       # Collect papers via Scopus API (GWU institutional)
│   ├── 1_data_collection_webofscience.py # Collect papers via Web of Science API (pending access)
│   ├── 2_data_cleaning.py                # Merge, deduplicate and standardise all sources
│   ├── 3_modelling.py                    # Topic modelling (BERTopic / LDA)
│   ├── 4_model_finetuning.py             # Hyperparameter tuning and model optimisation
│   ├── 5_streamlit_dashboard.py          # Interactive visualisation dashboard
│   └── 6_stats_plots.py                  # Statistical analysis and static plots
│
├── data_raw/                             # Raw data as collected from APIs (never modified)
├── data_clean/                           # Cleaned and processed datasets
├── data_backend_metadata/                # API metadata, query logs, and collection records
├── models/                               # Saved topic models and embeddings
├── reports/                              # Written reports, methodology notes, findings
├── results/                              # Output files: charts, tables, geo summaries
├── src/                                  # Shared utility functions and helper modules
│
├── .env                                  # API keys — NOT committed to GitHub (see .gitignore)
├── .gitignore                            # Files excluded from version control
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

> **Note:** `data_raw/`, `data_clean/`, `models/`, `reports/`, and `results/` are intentionally empty at this stage. They are preserved in the repository using `.gitkeep` placeholder files and will be populated as the project progresses.

---

## 🗄️ Data Sources

This project collects data from three scholarly databases. The table below summarises their role in the project:

| Source | Coverage | Access | Role in Project |
|--------|----------|--------|-----------------|
| **OpenAlex** | 240M+ works | Free, open API | Primary corpus collection |
| **Scopus** | 78M+ works | GWU institutional API | Secondary corpus & cross-validation |
| **Web of Science** | 20M+ works | Pending institutional access | Tertiary source (if access granted) |

All data collection is for **non-commercial academic research purposes only**, in compliance with each provider's terms of use.

### Key Fields Collected
- Title and abstract
- Author names and institutional affiliations
- Country of first author (for geographic analysis)
- Publication year (for temporal analysis)
- DOI and citation count
- Keywords and topics

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- PyCharm (recommended) or any Python IDE
- GWU institutional network access or VPN (required for Scopus API)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure API Keys

Create a `.env` file in the project root (this file is gitignored and will never be uploaded):

```
SCOPUS_API_KEY=your_scopus_key_here
OPENALEX_EMAIL=your_gwu_email@gwu.edu
```

Get your free keys here:
- **Scopus:** [dev.elsevier.com](https://dev.elsevier.com)
- **OpenAlex:** [openalex.org/settings/api](https://openalex.org/settings/api)

---

## 🚦 Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Literature review & citation verification | ✅ Complete |
| 2 | API access & environment setup | ✅ Complete |
| 3 | Data collection scripts | 🔄 In Progress |
| 4 | Data cleaning & merging | ⏳ Pending |
| 5 | Topic modelling | ⏳ Pending |
| 6 | Geographic & temporal analysis | ⏳ Pending |
| 7 | Dashboard & visualisations | ⏳ Pending |
| 8 | Final report | ⏳ Pending |

---

## 📚 Key References

- Jobin et al. (2019). The global landscape of AI ethics guidelines. *Nature Machine Intelligence*
- Mittelstadt (2019). Principles alone cannot guarantee ethical AI. *Nature Machine Intelligence*
- Bareis & Katzenbach (2022). Talking AI into Being. *Science, Technology, & Human Values*
- Grootendorst (2022). BERTopic: Neural topic modelling. *arXiv*

---

## 📄 License

This project is for academic research purposes only. Data collected via Scopus and Web of Science APIs is subject to Elsevier and Clarivate's respective terms of use and may not be redistributed.

---

*Last updated: February 2026*