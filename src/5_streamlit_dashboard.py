# =============================================================================
#  5_streamlit_dashboard.py
#  Project: Mapping Global AI Governance Narratives
#  Authors: Tambudzai Gundani & Joshua Gray
#  Date:    April 2026
# =============================================================================
#
#  WHAT THIS SCRIPT DOES:
#  ----------------------
#  Interactive Streamlit dashboard visualising all research findings.
#  Organised across five tabs mirroring the research objectives.
#
#  TABS:
#  -----
#  1. Overview        — corpus stats, pipeline summary, key numbers
#  2. Topic Landscape — RO1: dominant governance themes + temporal shifts
#  3. Regional Atlas  — RO2: spatial distribution of governance discourse
#  4. Policy Alignment— RO3: academic vs policy corpus comparison
#  5. Stance Analysis — sentiment: risk vs opportunity framing
#
#  RUN:
#  ----
#  pip install streamlit plotly
#  streamlit run src/5_streamlit_dashboard.py
#
#  Then open: http://localhost:8501
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# -----------------------------------------------------------------------------
#  PAGE CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Governance Narratives",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
#  PATHS
# -----------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# -----------------------------------------------------------------------------
#  COLOUR PALETTE
# -----------------------------------------------------------------------------

BLUE        = "#1F4E79"
MIDBLUE     = "#2E75B6"
LIGHTBLUE   = "#BDD7EE"
GREEN       = "#1E7145"
AMBER       = "#C55A11"
RED         = "#C00000"
PURPLE      = "#7030A0"
TEAL        = "#008080"

REGION_COLOURS = {
    "Europe":               MIDBLUE,
    "Asia-Pacific":         TEAL,
    "North America":        GREEN,
    "Africa & Middle East": AMBER,
    "Latin America":        PURPLE,
    "Other / Unclassified": "#888888",
}

STANCE_COLOURS = {
    "risk_focused":        RED,
    "opportunity_focused": GREEN,
    "balanced":            "#888888",
}

PERIOD_COLOURS = {
    "pre_chatgpt":  "#888888",
    "post_chatgpt": MIDBLUE,
}

# -----------------------------------------------------------------------------
#  DATA LOADING
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    """Load all results files. Cache so they only load once."""

    def safe_read(filename):
        path = RESULTS_DIR / filename
        if path.exists():
            return pd.read_csv(path, dtype=str, low_memory=False)
        return pd.DataFrame()

    data = {
        "papers_stance":    safe_read("governance_papers_stance.csv"),
        "stance_topic":     safe_read("stance_by_topic.csv"),
        "stance_region":    safe_read("stance_by_region.csv"),
        "stance_period":    safe_read("stance_by_period.csv"),
        "stance_policy":    safe_read("stance_policy_comparison.csv"),
        "gov_topics":       safe_read("governance_topics.csv"),
        "gov_papers":       safe_read("governance_papers.csv"),
        "region_gov":       safe_read("region_governance_distribution.csv"),
        "period_gov":       safe_read("period_governance_distribution.csv"),
        "alignment":        safe_read("cross_corpus_alignment_v2.csv"),
        "policy_docs":      safe_read("policy_document_topics_v2.csv"),
        "model_comparison": safe_read("model_comparison.csv"),
        "topics_overview":  safe_read("topics_overview.csv"),
        "topics_finetuned": safe_read("topics_finetuned.csv"),
    }

    # Fix numeric columns
    numeric_cols = {
        "papers_stance":  ["governance_score", "score_risk", "score_opportunity",
                            "score_balanced", "stance_confidence", "topic_id_finetuned"],
        "stance_topic":   ["balanced", "opportunity_focused", "risk_focused",
                            "avg_risk_score", "avg_opportunity_score", "paper_count",
                            "topic_id_finetuned"],
        "stance_region":  ["balanced", "opportunity_focused", "risk_focused",
                            "avg_risk_score", "avg_opportunity_score", "paper_count"],
        "stance_period":  ["balanced", "opportunity_focused", "risk_focused",
                            "avg_risk_score", "avg_opportunity_score", "paper_count"],
        "stance_policy":  ["academic_papers", "academic_risk_score",
                            "academic_opportunity_score", "policy_count",
                            "topic_id_finetuned"],
        "gov_topics":     ["governance_score", "paper_count"],
        "region_gov":     ["count", "total", "proportion", "topic_id_finetuned"],
        "period_gov":     ["count", "total", "proportion", "topic_id_finetuned"],
        "alignment":      ["scopus_count", "policy_count", "governance_score"],
    }

    for key, cols in numeric_cols.items():
        if key in data and not data[key].empty:
            for col in cols:
                if col in data[key].columns:
                    data[key][col] = pd.to_numeric(
                        data[key][col], errors="coerce"
                    )

    return data


# -----------------------------------------------------------------------------
#  SIDEBAR
# -----------------------------------------------------------------------------

def render_sidebar(data):
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/George_Washington_University_monogram.svg/120px-George_Washington_University_monogram.svg.png", width=60)
        st.markdown("### AI Governance Narratives")
        st.markdown("*Tambudzai Gundani & Joshua Gray*")
        st.markdown("GWU Masters Capstone · 2026")
        st.divider()

        st.markdown("**Corpus Statistics**")
        papers = data["papers_stance"]
        if not papers.empty:
            st.metric("Governance Papers", f"{len(papers):,}")
            st.metric("Governance Topics", "21")
            st.metric("Countries", f"{papers['country'].nunique()}")
            st.metric("Policy Documents", "35")
        st.divider()

        st.markdown("**Filters**")
        regions = ["All Regions"] + sorted([
            r for r in data["papers_stance"]["region"].dropna().unique()
            if r != "Other / Unclassified"
        ]) + ["Other / Unclassified"]
        selected_region = st.selectbox("Region", regions)

        periods = ["All Periods", "pre_chatgpt", "post_chatgpt"]
        period_labels = {
            "All Periods": "All Periods",
            "pre_chatgpt": "Pre-ChatGPT (2015–2021)",
            "post_chatgpt": "Post-ChatGPT (2022–2025)"
        }
        selected_period = st.selectbox(
            "Time Period",
            periods,
            format_func=lambda x: period_labels.get(x, x)
        )

        st.divider()
        st.caption("GitHub: JGray-21/AI-Research-Analysis-Capstone")

    return selected_region, selected_period


def filter_papers(papers, region, period):
    """Apply sidebar filters to papers dataframe."""
    filtered = papers.copy()
    if region != "All Regions":
        filtered = filtered[filtered["region"] == region]
    if period != "All Periods":
        filtered = filtered[filtered["period"] == period]
    return filtered


# =============================================================================
#  TAB 1 — OVERVIEW
# =============================================================================

def render_overview(data):
    st.markdown("## 📊 Project Overview")
    st.markdown(
        "This dashboard presents findings from a spatial NLP analysis of **41,067 "
        "peer-reviewed AI publications** (Scopus, 2015–2025) and **35 national AI "
        "strategy documents** across six world regions. BERTopic topic modelling "
        "identified 21 governance-relevant themes, which were then analysed for "
        "regional distribution, temporal trends, policy alignment, and stance framing."
    )

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Papers", "41,067", "Scopus 2015–2025")
    col2.metric("Governance Papers", "3,186", "7.7% of corpus")
    col3.metric("Topics Discovered", "133", "21 governance-relevant")
    col4.metric("Policy Frameworks", "35", "6 world regions")
    col5.metric("Countries", "138", "First author affiliation")

    st.divider()

    col_left, col_right = st.columns(2)

    # Model comparison chart
    with col_left:
        st.markdown("### BERTopic Model Comparison")
        mc = data["model_comparison"]
        if not mc.empty:
            fig = go.Figure()
            colours = [MIDBLUE if str(b).lower() == "true" else LIGHTBLUE
                       for b in mc["best"]]
            fig.add_trace(go.Bar(
                x=mc["model"],
                y=mc["coherence_cv"].astype(float),
                name="Coherence (c_v)",
                marker_color=colours,
                text=[f"{v:.3f}" for v in mc["coherence_cv"].astype(float)],
                textposition="outside",
            ))
            fig.update_layout(
                height=320,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis_title="Coherence Score",
                showlegend=False,
                plot_bgcolor="white",
                yaxis=dict(range=[0, 0.85]),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("M1 (MiniLM) selected as best model. Composite score = coherence × diversity × (1 − outlier rate).")

    # Governance topic sizes
    with col_right:
        st.markdown("### Governance Topics by Paper Count")
        gt = data["gov_topics"].sort_values("paper_count", ascending=True)
        if not gt.empty:
            # Shorten labels
            gt["short_label"] = gt["topic_label"].str[:40]
            fig = px.bar(
                gt, x="paper_count", y="short_label",
                orientation="h",
                color="governance_score",
                color_continuous_scale=[[0, LIGHTBLUE], [1, BLUE]],
                labels={"paper_count": "Papers", "short_label": "",
                        "governance_score": "Gov. Score"},
            )
            fig.update_layout(
                height=500,
                margin=dict(t=10, b=20, l=10, r=20),
                plot_bgcolor="white",
                coloraxis_colorbar=dict(title="Score", thickness=12),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Pipeline summary table
    st.markdown("### Pipeline Summary")
    pipeline = pd.DataFrame([
        {"Script": "1_data_collection_scopus", "Output": "41,067 papers, 138 countries", "Status": "✅"},
        {"Script": "1_data_collection_policyframeworks", "Output": "35 policy frameworks downloaded", "Status": "✅"},
        {"Script": "2_data_cleaning_scopus", "Output": "scopus_cleaned.csv — NLP cleaned", "Status": "✅"},
        {"Script": "2b_institution_geocoding", "Output": "40,950 institutions geocoded (99.96%)", "Status": "✅"},
        {"Script": "2c_coauthorship_edges", "Output": "116,351 edges, 2,410 country pairs", "Status": "✅"},
        {"Script": "2_policy_text_extraction", "Output": "621,492 words from 35 PDFs", "Status": "✅"},
        {"Script": "3_modelling", "Output": "46 topics — M1 best (coherence=0.716)", "Status": "✅"},
        {"Script": "4_model_finetuning", "Output": "Topic 0 decomposed → 133 total topics", "Status": "✅"},
        {"Script": "4b_update_governance_scores", "Output": "21 governance topics scored", "Status": "✅"},
        {"Script": "4c_data_integrity_fix", "Output": "Country, policy alignment, cross-corpus rebuild", "Status": "✅"},
        {"Script": "4d_sentiment_analysis", "Output": "3,186 papers: 37.7% opp | 30.6% risk | 31.7% balanced", "Status": "✅"},
    ])
    st.dataframe(pipeline, use_container_width=True, hide_index=True)


# =============================================================================
#  TAB 2 — TOPIC LANDSCAPE
# =============================================================================

def render_topics(data, region, period):
    st.markdown("## 🔍 Topic Landscape (RO1)")
    st.markdown(
        "Dominant AI governance themes discovered via BERTopic across "
        "41,067 academic abstracts. Topics are ordered by paper count."
    )

    period_gov = data["period_gov"].copy()
    region_gov = data["region_gov"].copy()

    # Filter by region/period if selected
    if region != "All Regions":
        region_gov = region_gov[region_gov["region"] == region]
    if period != "All Periods":
        period_gov = period_gov[period_gov["period"] == period]

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Governance Topics — Paper Count")
        gt = data["gov_topics"].sort_values("paper_count", ascending=True).copy()
        gt["short_label"] = gt["topic_label"].str[:45]
        fig = px.bar(
            gt, x="paper_count", y="short_label", orientation="h",
            color="governance_score",
            color_continuous_scale=[[0, "#BDD7EE"], [0.5, MIDBLUE], [1, BLUE]],
            labels={"paper_count": "Number of Papers",
                    "short_label": "",
                    "governance_score": "Governance\nRelevance Score"},
            hover_data={"top_words": True},
        )
        fig.update_layout(
            height=560, margin=dict(t=10, b=20, l=10, r=20),
            plot_bgcolor="white",
            coloraxis_colorbar=dict(title="Score", thickness=12),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Pre vs Post ChatGPT")
        st.caption("Proportion of each period's papers on each governance topic")

        pg = period_gov.copy()
        if not pg.empty:
            pg["period_label"] = pg["period"].map({
                "pre_chatgpt": "Pre-ChatGPT\n(2015–2021)",
                "post_chatgpt": "Post-ChatGPT\n(2022–2025)"
            })
            # Pivot for grouped bar
            pivot = pg.pivot_table(
                index="topic_label_finetuned",
                columns="period",
                values="proportion",
                fill_value=0
            ).reset_index()
            pivot = pivot.sort_values("post_chatgpt", ascending=False).head(12)

            fig = go.Figure()
            if "pre_chatgpt" in pivot.columns:
                fig.add_trace(go.Bar(
                    name="Pre-ChatGPT",
                    x=(pivot["pre_chatgpt"] * 100).round(2),
                    y=pivot["topic_label_finetuned"].str[:35],
                    orientation="h",
                    marker_color="#AAAAAA",
                ))
            if "post_chatgpt" in pivot.columns:
                fig.add_trace(go.Bar(
                    name="Post-ChatGPT",
                    x=(pivot["post_chatgpt"] * 100).round(2),
                    y=pivot["topic_label_finetuned"].str[:35],
                    orientation="h",
                    marker_color=MIDBLUE,
                ))
            fig.update_layout(
                barmode="group",
                height=560,
                margin=dict(t=10, b=20, l=10, r=20),
                plot_bgcolor="white",
                xaxis_title="% of period's papers",
                legend=dict(orientation="h", y=-0.08),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Growth table
    st.markdown("### Post-ChatGPT Growth in Governance Topics")
    pg_all = data["period_gov"].copy()
    if not pg_all.empty:
        pre  = pg_all[pg_all["period"] == "pre_chatgpt"][
            ["topic_label_finetuned", "count"]
        ].rename(columns={"count": "pre_count"})
        post = pg_all[pg_all["period"] == "post_chatgpt"][
            ["topic_label_finetuned", "count"]
        ].rename(columns={"count": "post_count"})
        growth = pre.merge(post, on="topic_label_finetuned", how="outer").fillna(0)
        growth["pre_count"]  = growth["pre_count"].astype(int)
        growth["post_count"] = growth["post_count"].astype(int)
        growth["growth_%"] = growth.apply(
            lambda r: round((r["post_count"] - r["pre_count"]) / r["pre_count"] * 100)
            if r["pre_count"] > 0 else 999, axis=1
        )
        growth = growth.sort_values("growth_%", ascending=False)
        growth.columns = ["Topic", "Pre-ChatGPT Papers", "Post-ChatGPT Papers", "Growth (%)"]
        st.dataframe(growth, use_container_width=True, hide_index=True)


# =============================================================================
#  TAB 3 — REGIONAL ATLAS
# =============================================================================

def render_regional(data, region, period):
    st.markdown("## 🌍 Regional Atlas (RO2)")
    st.markdown(
        "Spatial distribution of AI governance research across five world regions. "
        "Proportions show each topic's share of that region's total AI publications."
    )

    rg = data["region_gov"].copy()
    papers = data["papers_stance"].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Papers per Region")
        region_totals = papers.groupby("region").size().reset_index(name="count")
        region_totals = region_totals[region_totals["region"] != "Other / Unclassified"]
        region_totals = region_totals.sort_values("count", ascending=False)
        fig = px.bar(
            region_totals, x="region", y="count",
            color="region",
            color_discrete_map=REGION_COLOURS,
            labels={"region": "", "count": "Governance Papers"},
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=340, showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Dominant Topic per Region")
        if not rg.empty:
            dominant = (
                rg.sort_values("proportion", ascending=False)
                .groupby("region")
                .first()
                .reset_index()
            )[["region", "topic_label_finetuned", "proportion"]]
            dominant["proportion_%"] = (dominant["proportion"] * 100).round(2)
            dominant.columns = ["Region", "Dominant Governance Topic", "Proportion (%)"]
            st.dataframe(dominant, use_container_width=True, hide_index=True)

    st.markdown("### Regional Governance Topic Heatmap")
    st.caption("Proportion (%) of each region's papers covering each governance topic")
    if not rg.empty:
        pivot = rg.pivot_table(
            index="region",
            columns="topic_label_finetuned",
            values="proportion",
            fill_value=0
        ) * 100
        pivot = pivot.round(2)
        # Shorten column names
        pivot.columns = [c[:30] for c in pivot.columns]
        fig = px.imshow(
            pivot,
            color_continuous_scale=[[0, "white"], [0.3, LIGHTBLUE], [1, BLUE]],
            labels=dict(color="% of region"),
            aspect="auto",
            text_auto=".2f",
        )
        fig.update_layout(
            height=280,
            margin=dict(t=10, b=60, l=10, r=10),
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cybersecurity Governance — Regional Spotlight")
    st.markdown(
        "Africa & Middle East produces **2.02%** of its AI papers on cybersecurity "
        "governance — more than **3× higher** than Europe (0.63%). This reflects "
        "distinct regional threat priorities and governance concerns."
    )
    if not rg.empty:
        cyber = rg[rg["topic_label_finetuned"] == "AI Cybersecurity & IoT Threats"].copy()
        if not cyber.empty:
            cyber["proportion_%"] = (cyber["proportion"] * 100).round(3)
            cyber = cyber.sort_values("proportion_%", ascending=False)
            fig = px.bar(
                cyber, x="region", y="proportion_%",
                color="region",
                color_discrete_map=REGION_COLOURS,
                labels={"region": "", "proportion_%": "% of Region's Papers"},
                text="proportion_%",
            )
            fig.update_traces(textposition="outside",
                              texttemplate="%{text:.2f}%")
            fig.update_layout(
                height=300, showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Country table
    st.markdown("### Top Countries in Governance Corpus")
    country_counts = (
        papers[papers["country"] != "Unknown"]["country"]
        .value_counts()
        .reset_index()
    )
    country_counts.columns = ["Country", "Governance Papers"]
    country_counts = country_counts.head(20)
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.dataframe(country_counts, use_container_width=True, hide_index=True)
    with col_b:
        fig = px.bar(
            country_counts.head(15), x="Governance Papers", y="Country",
            orientation="h",
            color="Governance Papers",
            color_continuous_scale=[[0, LIGHTBLUE], [1, BLUE]],
        )
        fig.update_layout(
            height=400, showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
#  TAB 4 — POLICY ALIGNMENT
# =============================================================================

def render_policy(data):
    st.markdown("## 📜 Academic vs Policy Alignment (RO3)")
    st.markdown(
        "Comparison of topic coverage between 3,186 academic governance papers "
        "and 35 national AI strategy documents (1,557 text chunks). "
        "Shared topics appear in both corpora; academic-only topics are absent from policy frameworks."
    )

    align = data["alignment"]
    policy_docs = data["policy_docs"]

    # Alignment summary metrics
    if not align.empty:
        gov_align = align[align["governance_score"] >= 0.5]
        shared    = (gov_align["alignment"] == "shared").sum()
        acad_only = (gov_align["alignment"] == "academic_only").sum()
        pol_only  = (gov_align["alignment"] == "policy_only").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Shared Topics", shared, "in both corpora")
        col2.metric("Academic-Only", acad_only, "absent from policy")
        col3.metric("Policy-Only", pol_only, "absent from academia")

    st.markdown("### Shared vs Academic-Only Governance Topics")
    if not align.empty:
        gov_align = align[
            align["governance_score"] >= 0.5
        ].copy().sort_values("scopus_count", ascending=False)
        gov_align = gov_align[gov_align["alignment"].isin(["shared","academic_only"])]
        gov_align["short_label"] = gov_align["topic_label"].str[:40]
        gov_align["alignment_label"] = gov_align["alignment"].map({
            "shared":        "Shared (academic + policy)",
            "academic_only": "Academic Only",
        })
        colour_map = {
            "Shared (academic + policy)": MIDBLUE,
            "Academic Only":              "#AAAAAA",
        }
        fig = px.bar(
            gov_align, x="scopus_count", y="short_label",
            color="alignment_label",
            color_discrete_map=colour_map,
            orientation="h",
            labels={"scopus_count": "Academic Papers",
                    "short_label": "",
                    "alignment_label": ""},
            hover_data={"policy_count": True},
        )
        fig.update_layout(
            height=520,
            margin=dict(t=10, b=20, l=10, r=10),
            plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.08),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Policy vs Academic Coverage — Shared Topics")
    if not align.empty:
        shared = align[
            (align["alignment"] == "shared") &
            (align["governance_score"] >= 0.5)
        ].copy()
        if not shared.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Academic Papers",
                x=shared["topic_label"].str[:35],
                y=shared["scopus_count"],
                marker_color=MIDBLUE,
            ))
            fig.add_trace(go.Bar(
                name="Policy Chunks",
                x=shared["topic_label"].str[:35],
                y=shared["policy_count"],
                marker_color=AMBER,
            ))
            fig.update_layout(
                barmode="group",
                height=380,
                margin=dict(t=10, b=100, l=10, r=10),
                plot_bgcolor="white",
                xaxis=dict(tickangle=45, tickfont=dict(size=9)),
                yaxis_title="Count",
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Note: EU AI Regulation (407 policy chunks vs 100 papers) and "
                "Smart City Governance (123 vs 63) show policy leading academic research — "
                "governments are ahead of researchers on these themes."
            )

    # Policy document topic coverage
    st.markdown("### Policy Framework Topic Coverage by Country")
    if not policy_docs.empty:
        policy_docs["topic_label_finetuned"] = policy_docs["topic_label_finetuned"].fillna("Unassigned")
        pivot_pol = policy_docs.pivot_table(
            index="country",
            columns="topic_label_finetuned",
            values="chunk_count",
            aggfunc="sum",
            fill_value=0
        )
        # Shorten column names
        pivot_pol.columns = [c[:25] for c in pivot_pol.columns]
        fig = px.imshow(
            pivot_pol,
            color_continuous_scale=[[0, "white"], [0.3, "#FCE4D6"], [1, AMBER]],
            labels=dict(color="Chunks"),
            aspect="auto",
            text_auto=True,
        )
        fig.update_layout(
            height=500,
            margin=dict(t=10, b=80, l=10, r=10),
            xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "EU regulatory vocabulary dominates Global South policy documents — "
            "Zimbabwe, Kenya, ASEAN, and African Union all mirror EU AI Act language."
        )


# =============================================================================
#  TAB 5 — STANCE ANALYSIS
# =============================================================================

def render_stance(data, region, period):
    st.markdown("## 💬 Stance Analysis — Risk vs Opportunity (RO1)")
    st.markdown(
        "Zero-shot stance classification using `facebook/bart-large-mnli` on "
        "3,186 governance paper abstracts. Each abstract is classified as "
        "**risk-focused**, **opportunity-focused**, or **balanced**."
    )

    papers = filter_papers(data["papers_stance"], region, period)

    # Overall metrics
    if not papers.empty:
        counts = papers["stance"].value_counts()
        total  = len(papers)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Papers Analysed", f"{total:,}")
        col2.metric("Opportunity-Focused",
                    f"{counts.get('opportunity_focused',0):,}",
                    f"{counts.get('opportunity_focused',0)/total*100:.1f}%")
        col3.metric("Balanced",
                    f"{counts.get('balanced',0):,}",
                    f"{counts.get('balanced',0)/total*100:.1f}%")
        col4.metric("Risk-Focused",
                    f"{counts.get('risk_focused',0):,}",
                    f"{counts.get('risk_focused',0)/total*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Stance by Governance Topic")
        st_topic = data["stance_topic"].copy()
        if not st_topic.empty:
            st_topic = st_topic.sort_values("avg_risk_score", ascending=True)
            st_topic["short_label"] = st_topic["topic_label_finetuned"].str[:40]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Risk-Focused",
                x=st_topic["risk_focused"] * 100,
                y=st_topic["short_label"],
                orientation="h",
                marker_color=RED,
            ))
            fig.add_trace(go.Bar(
                name="Opportunity-Focused",
                x=st_topic["opportunity_focused"] * 100,
                y=st_topic["short_label"],
                orientation="h",
                marker_color=GREEN,
            ))
            fig.add_trace(go.Bar(
                name="Balanced",
                x=st_topic["balanced"] * 100,
                y=st_topic["short_label"],
                orientation="h",
                marker_color="#CCCCCC",
            ))
            fig.update_layout(
                barmode="stack",
                height=560,
                margin=dict(t=10, b=20, l=10, r=20),
                plot_bgcolor="white",
                xaxis_title="Percentage of Papers (%)",
                legend=dict(orientation="h", y=-0.08),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Pre vs Post ChatGPT Stance Shift")
        sp = data["stance_period"].copy()
        if not sp.empty:
            sp["period_label"] = sp["period"].map({
                "pre_chatgpt":  "Pre-ChatGPT\n(2015–2021)",
                "post_chatgpt": "Post-ChatGPT\n(2022–2025)"
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Risk-Focused",
                x=sp["period_label"],
                y=sp["risk_focused"] * 100,
                marker_color=RED,
                text=(sp["risk_focused"] * 100).round(1),
                texttemplate="%{text}%",
                textposition="inside",
            ))
            fig.add_trace(go.Bar(
                name="Opportunity-Focused",
                x=sp["period_label"],
                y=sp["opportunity_focused"] * 100,
                marker_color=GREEN,
                text=(sp["opportunity_focused"] * 100).round(1),
                texttemplate="%{text}%",
                textposition="inside",
            ))
            fig.add_trace(go.Bar(
                name="Balanced",
                x=sp["period_label"],
                y=sp["balanced"] * 100,
                marker_color="#CCCCCC",
                text=(sp["balanced"] * 100).round(1),
                texttemplate="%{text}%",
                textposition="inside",
            ))
            fig.update_layout(
                barmode="stack",
                height=360,
                margin=dict(t=10, b=20, l=10, r=20),
                plot_bgcolor="white",
                yaxis_title="% of Papers",
                legend=dict(orientation="h", y=-0.12),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(
                "📌 **Key finding:** Pre-ChatGPT governance research was "
                "risk-dominant (34.9% risk vs 29.7% opportunity). "
                "Post-ChatGPT, framing flipped to opportunity-dominant "
                "(40.0% opportunity vs 29.4% risk), despite a surge in "
                "total governance research volume."
            )

        st.markdown("### Stance by Region")
        sr = data["stance_region"].copy()
        if not sr.empty:
            sr = sr.sort_values("avg_risk_score", ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Risk", x=sr["region"],
                y=sr["risk_focused"] * 100,
                marker_color=RED,
            ))
            fig.add_trace(go.Bar(
                name="Opportunity", x=sr["region"],
                y=sr["opportunity_focused"] * 100,
                marker_color=GREEN,
            ))
            fig.add_trace(go.Bar(
                name="Balanced", x=sr["region"],
                y=sr["balanced"] * 100,
                marker_color="#CCCCCC",
            ))
            fig.update_layout(
                barmode="group",
                height=320,
                margin=dict(t=10, b=20, l=10, r=10),
                plot_bgcolor="white",
                yaxis_title="% of Papers",
                xaxis=dict(tickangle=15),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Europe leads on risk framing (33.3%). "
                "Africa & Middle East most opportunity-focused (45.0%). "
                "North America most balanced."
            )

    # Stance × topic deep dive
    st.markdown("### Academic Stance on Shared Topics")
    st.caption("How academics frame topics that also appear in national policy frameworks")
    sp_comp = data["stance_policy"].copy()
    if not sp_comp.empty:
        sp_comp["risk_%"]         = (sp_comp["academic_risk_score"] * 100).round(1)
        sp_comp["opportunity_%"]  = (sp_comp["academic_opportunity_score"] * 100).round(1)
        sp_comp["balanced_%"]     = (
            100 - sp_comp["risk_%"] - sp_comp["opportunity_%"]
        ).round(1)
        display = sp_comp[[
            "topic_label_finetuned", "academic_papers",
            "risk_%", "opportunity_%", "balanced_%",
            "academic_dominant_stance", "policy_count"
        ]].copy()
        display.columns = [
            "Topic", "Academic Papers", "Risk (%)",
            "Opportunity (%)", "Balanced (%)",
            "Dominant Stance", "Policy Chunks"
        ]
        st.dataframe(display, use_container_width=True, hide_index=True)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    data = load_data()
    region, period = render_sidebar(data)

    st.title("🌍 Mapping Global AI Governance Narratives")
    st.markdown(
        "**A Spatial NLP Analysis of Academic and Policy Discourse** — "
        "Tambudzai Gundani & Joshua Gray · GWU Masters Capstone · 2026"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Topic Landscape",
        "🌍 Regional Atlas",
        "📜 Policy Alignment",
        "💬 Stance Analysis",
    ])

    with tab1:
        render_overview(data)
    with tab2:
        render_topics(data, region, period)
    with tab3:
        render_regional(data, region, period)
    with tab4:
        render_policy(data)
    with tab5:
        render_stance(data, region, period)


if __name__ == "__main__":
    main()