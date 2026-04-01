# =============================================================================
#  4b_update_governance_scores.py
#  Running this ONCE after 4_model_finetuning.py
#  It updates the governance scores for the sub-topics found inside Topic 0
#  and rebuilds governance_papers.csv with the complete governance corpus
# =============================================================================

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# -----------------------------------------------------------------------------
#  GOVERNANCE SCORES FOR TOPIC 0 SUB-TOPICS
#  Based on keyword inspection of topic0_subtopics.csv
# -----------------------------------------------------------------------------

SUBTOPIC_UPDATES = {
    # sub_topic_id : (new_label, new_governance_score)
    103: ("AI Law, Ethics & Human Rights",               0.95),
    129: ("AI in Public Administration & Governance",    0.90),
    135: ("EU AI Regulation & Regulatory Frameworks",    1.00),
    177: ("Explainable AI & Transparency",               0.80),
    114: ("Generative AI & Creative Ethics",             0.65),
    148: ("Smart City & Urban AI Governance",            0.70),
    151: ("AI Governance during COVID-19",               0.60),
    183: ("UAV & Drone Governance",                      0.60),
    168: ("Digital Transformation & AI Adoption",        0.55),
    181: ("AI Patent Law & Intellectual Property",       0.55),
    124: ("AI Research Landscape (Bibliometrics)",       0.50),
    # Everything else stays at 0.1
}

DEFAULT_SCORE = 0.1  # for sub-topics not listed above

# -----------------------------------------------------------------------------
#  LOAD AND UPDATE
# -----------------------------------------------------------------------------

print("Loading topics_finetuned.csv...")
topics_df = pd.read_csv(RESULTS_DIR / "topics_finetuned.csv")

# Update sub-topic labels and scores
updated = 0
for idx, row in topics_df.iterrows():
    if row["source"] != "topic0_subtopic":
        continue

    tid = int(row["topic_id"])

    if tid in SUBTOPIC_UPDATES:
        new_label, new_score = SUBTOPIC_UPDATES[tid]
        topics_df.at[idx, "topic_label"]      = new_label
        topics_df.at[idx, "governance_score"] = new_score
        print(f"  [{tid}] → {new_label}  (score={new_score})")
        updated += 1
    else:
        topics_df.at[idx, "governance_score"] = DEFAULT_SCORE

print(f"\nUpdated {updated} sub-topics with governance scores")

# Save updated topics
topics_df.to_csv(RESULTS_DIR / "topics_finetuned.csv", index=False)
print("✅ topics_finetuned.csv saved")

# -----------------------------------------------------------------------------
#  REBUILD GOVERNANCE FILES
# -----------------------------------------------------------------------------

print("\nRebuilding governance files...")

# Governance topics (score >= 0.5)
gov_topics = topics_df[topics_df["governance_score"] >= 0.5].copy()
gov_topics.to_csv(RESULTS_DIR / "governance_topics.csv", index=False)
print(f"✅ governance_topics.csv — {len(gov_topics)} topics")

# Load full paper assignments
scopus_df = pd.read_csv(RESULTS_DIR / "scopus_topics_finetuned.csv")

# Merge updated scores in
score_map = dict(zip(topics_df["topic_id"], topics_df["governance_score"]))
label_map = dict(zip(topics_df["topic_id"], topics_df["topic_label"]))

scopus_df["governance_score"]         = scopus_df["topic_id_finetuned"].map(score_map)
scopus_df["topic_label_finetuned"]    = scopus_df["topic_id_finetuned"].map(label_map)

# Rebuild governance papers
gov_papers = scopus_df[scopus_df["governance_score"] >= 0.5].copy()
gov_papers.to_csv(RESULTS_DIR / "governance_papers.csv", index=False)
print(f"✅ governance_papers.csv — {len(gov_papers):,} papers")

# Rebuild region distribution
region_gov = (
    gov_papers[gov_papers["topic_id_finetuned"] != -1]
    .groupby(["region", "topic_id_finetuned", "topic_label_finetuned"])
    .size()
    .reset_index(name="count")
)
region_totals = scopus_df.groupby("region").size().reset_index(name="total")
region_gov = region_gov.merge(region_totals, on="region")
region_gov["proportion"] = (region_gov["count"] / region_gov["total"]).round(4)
region_gov.to_csv(RESULTS_DIR / "region_governance_distribution.csv", index=False)
print("✅ region_governance_distribution.csv rebuilt")

# Rebuild period distribution
period_gov = (
    gov_papers[gov_papers["topic_id_finetuned"] != -1]
    .groupby(["period", "topic_id_finetuned", "topic_label_finetuned"])
    .size()
    .reset_index(name="count")
)
period_totals = scopus_df.groupby("period").size().reset_index(name="total")
period_gov = period_gov.merge(period_totals, on="period")
period_gov["proportion"] = (period_gov["count"] / period_gov["total"]).round(4)
period_gov.to_csv(RESULTS_DIR / "period_governance_distribution.csv", index=False)
print("✅ period_governance_distribution.csv rebuilt")

# -----------------------------------------------------------------------------
#  SUMMARY
# -----------------------------------------------------------------------------

print("\n" + "=" * 50)
print("  GOVERNANCE CORPUS — UPDATED SUMMARY")
print("=" * 50)
print(f"  Total governance topics:  {len(gov_topics)}")
print(f"  Total governance papers:  {len(gov_papers):,}")
print()
print("  Papers by region:")
for region, count in gov_papers["region"].value_counts().items():
    print(f"    {region:<25} {count:,}")
print()
print("  Papers by period:")
for period, count in gov_papers["period"].value_counts().items():
    print(f"    {period:<20} {count:,}")
print()
print("  Top governance topics by paper count:")
top = gov_topics.sort_values("paper_count", ascending=False).head(10)
for _, r in top.iterrows():
    print(f"    [{int(r['topic_id']):3d}] {r['paper_count']:>5,}  {r['topic_label'][:50]}")
print("=" * 50)
print("🎉 Done! Governance corpus is now complete.")
print("   Next step: run 5_streamlit_dashboard.py")