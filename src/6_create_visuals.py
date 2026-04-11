"""
6_create_visuals.py
Project: Mapping Global AI Governance Narratives
Authors: Tambudzai Gundani & Joshua Gray

Run this script ONCE to generate 6_exploratory_visuals.ipynb
Then open that notebook in PyCharm and run cells interactively.

Usage:
    python src/6_create_visuals.py

Requirements:
    pip install altair vega_datasets pycountry networkx nbformat
"""

import json
from pathlib import Path

OUT_PATH = Path(__file__).parent / "6_exploratory_visuals.ipynb"


# ── helpers ──────────────────────────────────────────────────────────────────

def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip()
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip()
    }


# =============================================================================
#  CELLS
# =============================================================================

cells = []

# ── TITLE ────────────────────────────────────────────────────────────────────
cells.append(md("""# Mapping Global AI Governance Narratives — Exploratory Visuals
**Authors:** Tambudzai Gundani & Joshua Gray | GWU Masters Capstone | 2026

This notebook contains **20 Altair visualisations** (5 per Research Objective + Stance Analysis)
exploring AI governance discourse across 41,067 academic papers and 35 national AI policy frameworks.

**Sections:**
1. 🔍 RO1 — Dominant Governance Themes
2. 🌍 RO2 — Regional & Spatial Distribution
3. 📜 RO3 — Academic vs Policy Alignment
4. 💬 Stance Analysis — Risk vs Opportunity
"""))

# ── SETUP ────────────────────────────────────────────────────────────────────
cells.append(md("## ⚙️ Setup & Data Loading"))

cells.append(code("""
# Install if needed:
# pip install altair vega_datasets pycountry networkx nbformat

import warnings
warnings.filterwarnings('ignore')

import altair as alt
import pandas as pd
import numpy as np
import networkx as nx
from vega_datasets import data as vega_data
from pathlib import Path

# Remove Altair 5000-row limit
alt.data_transformers.disable_max_rows()

# ── PATHS ──────────────────────────────────────────────────────────────────
def find_root():
    for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
        if (p / 'results').exists() and (p / 'data_clean').exists():
            return p
    raise FileNotFoundError('Could not find project root with results/ and data_clean/')

ROOT       = find_root()
RESULTS    = ROOT / 'results'
DATA_CLEAN = ROOT / 'data_clean'
print(f'Project root: {ROOT}')

# ── COLOUR PALETTE ──────────────────────────────────────────────────────────
BLUE   = '#1F4E79'
MID    = '#2E75B6'
LIGHT  = '#BDD7EE'
GREEN  = '#1E7145'
AMBER  = '#C55A11'
RED    = '#C00000'
PURPLE = '#7030A0'
TEAL   = '#008080'

REGION_DOMAIN  = ['Europe','Asia-Pacific','North America','Africa & Middle East','Latin America','Other / Unclassified']
REGION_RANGE   = [MID, TEAL, GREEN, AMBER, PURPLE, '#888888']

STANCE_DOMAIN  = ['risk_focused','opportunity_focused','balanced']
STANCE_RANGE   = [RED, GREEN, '#AAAAAA']
"""))

cells.append(code("""
# ── LOAD ALL DATA ────────────────────────────────────────────────────────────
def sr(f, **kw):
    p = RESULTS / f
    if not p.exists(): p = DATA_CLEAN / f
    return pd.read_csv(p, dtype=str, low_memory=False, **kw) if p.exists() else pd.DataFrame()

papers   = sr('governance_papers_stance.csv')
s_topic  = sr('stance_by_topic.csv')
s_region = sr('stance_by_region.csv')
s_period = sr('stance_by_period.csv')
s_policy = sr('stance_policy_comparison.csv')
gov_top  = sr('governance_topics.csv')
reg_gov  = sr('region_governance_distribution.csv')
per_gov  = sr('period_governance_distribution.csv')
align    = sr('cross_corpus_alignment_v2.csv')
pol_docs = sr('policy_document_topics_v2.csv')
edges    = sr('scopus_country_edges.csv')
nodes    = sr('scopus_country_nodes.csv')

# Numeric conversions
for df, cols in [
    (papers,   ['governance_score','score_risk','score_opportunity','score_balanced','topic_id_finetuned']),
    (s_topic,  ['balanced','opportunity_focused','risk_focused','avg_risk_score','avg_opportunity_score','paper_count','topic_id_finetuned']),
    (s_region, ['balanced','opportunity_focused','risk_focused','avg_risk_score','avg_opportunity_score','paper_count']),
    (s_period, ['balanced','opportunity_focused','risk_focused','avg_risk_score','avg_opportunity_score','paper_count']),
    (s_policy, ['academic_papers','academic_risk_score','academic_opportunity_score','policy_count']),
    (gov_top,  ['governance_score','paper_count']),
    (reg_gov,  ['count','total','proportion']),
    (per_gov,  ['count','total','proportion']),
    (align,    ['scopus_count','policy_count','governance_score']),
    (edges,    ['weight']),
    (nodes,    ['latitude','longitude','total_papers']),
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

# Derived columns
papers['year'] = papers['cover_date'].astype(str).str[:4]
papers['year'] = pd.to_numeric(papers['year'], errors='coerce')

print('✅ All data loaded')
print(f'   Governance papers: {len(papers):,}')
print(f'   Country edges:     {len(edges):,}')
print(f'   Country nodes:     {len(nodes):,}')
"""))

cells.append(code("""
# ── ISO COUNTRY CODE MAPPING (for world map) ─────────────────────────────────
# ISO 3166-1 numeric codes used by vega world_110m topojson
COUNTRY_ISO = {
    'China':756,'United States':840,'India':356,'United Kingdom':826,
    'Germany':276,'South Korea':410,'Italy':380,'Australia':36,
    'Spain':724,'Canada':124,'France':250,'Japan':392,'Brazil':76,
    'Netherlands':528,'Saudi Arabia':682,'Malaysia':458,'Pakistan':586,
    'Turkey':792,'Portugal':620,'Poland':616,'Switzerland':756,
    'Sweden':752,'Belgium':56,'Iran':364,'Greece':300,'Egypt':818,
    'Nigeria':566,'South Africa':710,'Singapore':702,'Taiwan':158,
    'Hong Kong':344,'Indonesia':360,'Mexico':484,'Colombia':170,
    'Chile':152,'Argentina':32,'Russia':643,'Ukraine':804,
    'Czech Republic':203,'Denmark':208,'Finland':246,'Norway':578,
    'Austria':40,'Romania':642,'Israel':376,'Jordan':400,'Iraq':368,
    'Morocco':504,'Bangladesh':50,'Qatar':634,'United Arab Emirates':784,
    'UAE':784,'Kenya':404,'Ethiopia':231,'Ghana':288,'Rwanda':646,
    'Zimbabwe':716,'Algeria':12,'Tunisia':788,'Algeria':12,
    'New Zealand':554,'Philippines':608,'Thailand':764,'Vietnam':704,
    'Nepal':524,'Peru':604,'Ecuador':218,'Uruguay':858,'Hungary':348,
    'Slovakia':703,'Bulgaria':100,'Croatia':191,'Serbia':688,
    'Lithuania':440,'Ireland':372,'Slovenia':705,'Venezuela':862,
    'Bolivia':68,'Paraguay':600,'Kosovo':383,'Albania':8,
    'North Macedonia':807,'Montenegro':499,'Cyprus':196,
    'Sudan':729,'Tanzania':834,'Uganda':800,'Zambia':894,
    'Senegal':686,'Ivory Coast':384,'Angola':24,'Mozambique':508,
    'Cameroon':120,'Libya':434,'Somalia':706,
    'Sri Lanka':144,'Myanmar':104,'Cambodia':116,'Laos':418,
    'Mongolia':496,'Kazakhstan':398,'Uzbekistan':860,
    'Azerbaijan':31,'Georgia':268,'Armenia':51,'Belarus':112,
    'Moldova':498,'Estonia':233,'Latvia':428,'Iceland':352,
    'Luxembourg':442,'Malta':470,'Liechtenstein':438,'Monaco':492,
    'Andorra':20,'San Marino':674,'Bosnia and Herzegovina':70,
    'Bahrain':48,'Kuwait':414,'Oman':512,'Lebanon':422,'Syria':760,
    'Yemen':887,'Afghanistan':4,'Kyrgyzstan':417,'Tajikistan':762,
    'Turkmenistan':795,'Cuba':192,'Dominican Republic':214,
    'Guatemala':320,'Honduras':340,'El Salvador':222,'Nicaragua':558,
    'Costa Rica':188,'Panama':591,'Jamaica':388,'Trinidad and Tobago':780,
}
# Fix China (should be 156)
COUNTRY_ISO['China'] = 156

# Build reverse map
ISO_COUNTRY = {v: k for k, v in COUNTRY_ISO.items()}

print(f'✅ Country ISO map: {len(COUNTRY_ISO)} entries')
"""))

# =============================================================================
#  SECTION 1 — RO1: TOPIC LANDSCAPE
# =============================================================================

cells.append(md("""---
## 🔍 RO1 — Dominant Governance Themes

> *What are the dominant AI risk and governance themes in academic discourse, and how have they evolved over time?*

**5 visualisations:**
1. Governance Topic Bubble Chart (risk score × opportunity score)
2. Topic Evolution Heatmap (year × topic)
3. Growth Slope Chart (pre vs post ChatGPT)
4. Stacked Area — Topic Volume 2015–2025
5. Topic Governance Score vs Paper Count
"""))

cells.append(md("### Visual 1.1 — Governance Topic Bubble Chart\nEach bubble = one governance topic. Position = framing (x=risk, y=opportunity), size = paper count, colour = governance relevance score."))
cells.append(code("""
bubble_data = s_topic.copy()
bubble_data['short_label'] = bubble_data['topic_label_finetuned'].str.replace('AI ', '').str[:30]

# Size scale: paper_count → bubble area
chart_bubble = alt.Chart(bubble_data).mark_circle(opacity=0.85, stroke='white', strokeWidth=1).encode(
    x=alt.X('avg_risk_score:Q',
            scale=alt.Scale(domain=[0, 0.75]),
            axis=alt.Axis(title='Average Risk Score →', grid=True, gridOpacity=0.3)),
    y=alt.Y('avg_opportunity_score:Q',
            scale=alt.Scale(domain=[0, 0.75]),
            axis=alt.Axis(title='Average Opportunity Score →', grid=True, gridOpacity=0.3)),
    size=alt.Size('paper_count:Q',
                  scale=alt.Scale(range=[100, 4000]),
                  legend=alt.Legend(title='Papers')),
    color=alt.Color('avg_risk_score:Q',
                    scale=alt.Scale(scheme='redblue', reverse=True),
                    legend=alt.Legend(title='Risk Score')),
    tooltip=[
        alt.Tooltip('topic_label_finetuned:N', title='Topic'),
        alt.Tooltip('paper_count:Q', title='Papers'),
        alt.Tooltip('avg_risk_score:Q', title='Avg Risk Score', format='.3f'),
        alt.Tooltip('avg_opportunity_score:Q', title='Avg Opp Score', format='.3f'),
        alt.Tooltip('dominant_stance:N', title='Dominant Stance'),
    ]
).properties(width=640, height=480)

labels = alt.Chart(bubble_data).mark_text(
    fontSize=9, dy=-14, fontWeight='bold', color='#333'
).encode(
    x='avg_risk_score:Q',
    y='avg_opportunity_score:Q',
    text='short_label:N'
)

# Diagonal guide line (equal risk/opportunity)
diag = pd.DataFrame({'x': [0, 0.75], 'y': [0, 0.75]})
diag_line = alt.Chart(diag).mark_line(
    strokeDash=[4, 4], color='#999', opacity=0.5
).encode(x='x:Q', y='y:Q')

v1 = (diag_line + chart_bubble + labels).properties(
    title=alt.TitleParams(
        'Governance Topics: Risk vs Opportunity Framing',
        subtitle='Bubble size = paper count | Position = avg stance scores | Colour = risk intensity',
        fontSize=15, subtitleFontSize=11
    )
).configure_view(strokeWidth=0).configure_axis(labelFontSize=10)

v1
"""))

cells.append(md("### Visual 1.2 — Topic Evolution Heatmap (2015–2025)\nHow has each governance topic grown year by year?"))
cells.append(code("""
# Build year × topic matrix
year_topic = papers.groupby(['year', 'topic_label_finetuned']).size().reset_index(name='count')
year_topic = year_topic[year_topic['year'].between(2015, 2025)]
year_topic['short_label'] = year_topic['topic_label_finetuned'].str[:40]

v2 = alt.Chart(year_topic).mark_rect().encode(
    x=alt.X('year:O',
            axis=alt.Axis(title='Year', labelAngle=0)),
    y=alt.Y('short_label:N',
            sort=alt.SortField('count', order='descending'),
            axis=alt.Axis(title='')),
    color=alt.Color('count:Q',
                    scale=alt.Scale(scheme='blues'),
                    legend=alt.Legend(title='Papers')),
    tooltip=[
        alt.Tooltip('topic_label_finetuned:N', title='Topic'),
        alt.Tooltip('year:O', title='Year'),
        alt.Tooltip('count:Q', title='Papers'),
    ]
).properties(
    width=580, height=460,
    title=alt.TitleParams(
        'AI Governance Topic Activity by Year (2015–2025)',
        subtitle='Darker = more papers | Vertical bands show ChatGPT launch effect (2023+)',
        fontSize=15, subtitleFontSize=11
    )
)

# Add ChatGPT marker
chatgpt_mark = alt.Chart(pd.DataFrame({'year': ['2023']})).mark_rule(
    color=RED, strokeWidth=2, strokeDash=[4, 3]
).encode(x='year:O')

chatgpt_label = alt.Chart(pd.DataFrame({'year': ['2023'], 'y': [0.5], 'text': ['← ChatGPT']})).mark_text(
    angle=0, color=RED, fontSize=10, dx=35, dy=-180
).encode(x='year:O', text='text:N')

(v2 + chatgpt_mark).configure_view(step=28).configure_axis(labelFontSize=9)
"""))

cells.append(md("### Visual 1.3 — Pre vs Post ChatGPT Growth (Slope Chart)\nWhich governance topics grew most dramatically after ChatGPT?"))
cells.append(code("""
pre  = per_gov[per_gov['period'] == 'pre_chatgpt'][['topic_label_finetuned','proportion']].rename(columns={'proportion':'pre'})
post = per_gov[per_gov['period'] == 'post_chatgpt'][['topic_label_finetuned','proportion']].rename(columns={'proportion':'post'})
slope_data = pre.merge(post, on='topic_label_finetuned')
slope_data['growth'] = ((slope_data['post'] - slope_data['pre']) / slope_data['pre'].clip(lower=0.0001) * 100).round(0)
slope_data['short']  = slope_data['topic_label_finetuned'].str[:35]
slope_data['color']  = slope_data['growth'].apply(lambda x: RED if x < 0 else MID)

# Melt for slope
melted = slope_data.melt(id_vars=['topic_label_finetuned','short','growth','color'],
                          value_vars=['pre','post'], var_name='period', value_name='proportion')
melted['period_label'] = melted['period'].map({'pre': 'Pre-ChatGPT\n(2015–21)', 'post': 'Post-ChatGPT\n(2022–25)'})
melted['proportion_pct'] = (melted['proportion'] * 100).round(3)

lines = alt.Chart(melted).mark_line(point=True, strokeWidth=2).encode(
    x=alt.X('period_label:O',
            axis=alt.Axis(title='', labelFontSize=12, labelFontWeight='bold'),
            sort=['Pre-ChatGPT\n(2015–21)', 'Post-ChatGPT\n(2022–25)']),
    y=alt.Y('proportion_pct:Q',
            axis=alt.Axis(title='% of period papers', grid=True, gridOpacity=0.3)),
    detail='topic_label_finetuned:N',
    color=alt.Color('growth:Q',
                    scale=alt.Scale(scheme='redblue', reverse=False, domain=[-100, 700]),
                    legend=alt.Legend(title='Growth %')),
    tooltip=[
        alt.Tooltip('topic_label_finetuned:N', title='Topic'),
        alt.Tooltip('proportion_pct:Q', title='% of papers', format='.3f'),
        alt.Tooltip('growth:Q', title='Growth (%)', format='+.0f'),
    ]
)

labels_right = alt.Chart(melted[melted['period'] == 'post']).mark_text(
    align='left', dx=6, fontSize=8
).encode(
    x=alt.X('period_label:O', sort=['Pre-ChatGPT\n(2015–21)', 'Post-ChatGPT\n(2022–25)']),
    y='proportion_pct:Q',
    text='short:N',
    color=alt.Color('growth:Q', scale=alt.Scale(scheme='redblue', reverse=False))
)

v3 = (lines + labels_right).properties(
    width=420, height=560,
    title=alt.TitleParams(
        'Governance Topic Trajectory: Pre vs Post ChatGPT',
        subtitle='Lines rising steeply = topics that surged after ChatGPT launch',
        fontSize=15, subtitleFontSize=11
    )
).configure_view(strokeWidth=0)

v3
"""))

cells.append(md("### Visual 1.4 — Stacked Area: Governance Research Volume 2015–2025"))
cells.append(code("""
# Papers per year per topic
area_data = papers.groupby(['year','topic_label_finetuned']).size().reset_index(name='count')
area_data = area_data[area_data['year'].between(2015, 2025)]

# Top 8 topics for readability
top8 = papers['topic_label_finetuned'].value_counts().head(8).index.tolist()
area_data8 = area_data[area_data['topic_label_finetuned'].isin(top8)].copy()
area_data8['short'] = area_data8['topic_label_finetuned'].str[:30]

v4 = alt.Chart(area_data8).mark_area(opacity=0.85).encode(
    x=alt.X('year:O',
            axis=alt.Axis(title='Year', labelAngle=0, labelFontSize=11)),
    y=alt.Y('count:Q',
            stack='zero',
            axis=alt.Axis(title='Governance Papers Published', grid=True)),
    color=alt.Color('short:N',
                    scale=alt.Scale(scheme='tableau10'),
                    legend=alt.Legend(title='Topic', orient='right')),
    tooltip=[
        alt.Tooltip('topic_label_finetuned:N', title='Topic'),
        alt.Tooltip('year:O', title='Year'),
        alt.Tooltip('count:Q', title='Papers'),
    ]
).properties(
    width=660, height=380,
    title=alt.TitleParams(
        'Growth in AI Governance Research Volume (2015–2025)',
        subtitle='Stacked areas show cumulative growth across top 8 governance topics',
        fontSize=15, subtitleFontSize=11
    )
)

chatgpt_rule = alt.Chart(pd.DataFrame({'year': [2023]})).mark_rule(
    color=RED, strokeWidth=2, strokeDash=[5,4]
).encode(x='year:O')

chatgpt_text = alt.Chart(pd.DataFrame({'year': [2023], 'y': [50]})).mark_text(
    text='ChatGPT', color=RED, fontSize=10, angle=90, dy=-60
).encode(x='year:O', y=alt.Y('y:Q'))

(v4 + chatgpt_rule + chatgpt_text).configure_view(strokeWidth=1).configure_axis(labelFontSize=10)
"""))

cells.append(md("### Visual 1.5 — Governance Relevance Score vs Paper Count"))
cells.append(code("""
v5 = alt.Chart(gov_top).mark_circle(opacity=0.9).encode(
    x=alt.X('paper_count:Q',
            scale=alt.Scale(type='log'),
            axis=alt.Axis(title='Paper Count (log scale)')),
    y=alt.Y('governance_score:Q',
            scale=alt.Scale(domain=[0.45, 1.05]),
            axis=alt.Axis(title='Governance Relevance Score')),
    size=alt.Size('paper_count:Q',
                  scale=alt.Scale(range=[80, 1800]),
                  legend=None),
    color=alt.Color('governance_score:Q',
                    scale=alt.Scale(scheme='goldgreen'),
                    legend=alt.Legend(title='Gov. Score')),
    tooltip=[
        alt.Tooltip('topic_label:N', title='Topic'),
        alt.Tooltip('paper_count:Q', title='Papers'),
        alt.Tooltip('governance_score:Q', title='Gov. Score', format='.2f'),
        alt.Tooltip('top_words:N', title='Keywords'),
    ]
).properties(width=560, height=420)

text5 = alt.Chart(gov_top).mark_text(fontSize=9, dy=-13, color='#333').encode(
    x=alt.X('paper_count:Q', scale=alt.Scale(type='log')),
    y=alt.Y('governance_score:Q', scale=alt.Scale(domain=[0.45, 1.05])),
    text=alt.Text('topic_label:N',
                  condition=alt.condition(
                      alt.datum.paper_count > 80,
                      alt.value(None),
                      alt.value(None)
                  ))
)

# Custom text for top topics
top_labels = gov_top[gov_top['paper_count'] > 60].copy()
top_labels['short'] = top_labels['topic_label'].str[:25]
text5b = alt.Chart(top_labels).mark_text(fontSize=8, dy=-13, color='#333').encode(
    x=alt.X('paper_count:Q', scale=alt.Scale(type='log')),
    y=alt.Y('governance_score:Q', scale=alt.Scale(domain=[0.45, 1.05])),
    text='short:N'
)

# Reference lines
ref_h = alt.Chart(pd.DataFrame({'y': [0.9]})).mark_rule(
    strokeDash=[4,3], color=AMBER, opacity=0.6
).encode(y='y:Q')
ref_v = alt.Chart(pd.DataFrame({'x': [100]})).mark_rule(
    strokeDash=[4,3], color=AMBER, opacity=0.6
).encode(x='x:Q')

(v5 + text5b + ref_h + ref_v).properties(
    title=alt.TitleParams(
        'Governance Relevance vs Research Volume',
        subtitle='Top-right = high relevance AND high volume (most impactful topics)',
        fontSize=15, subtitleFontSize=11
    )
).configure_view(strokeWidth=1).configure_axis(labelFontSize=10)
"""))

# =============================================================================
#  SECTION 2 — RO2: REGIONAL ATLAS
# =============================================================================

cells.append(md("""---
## 🌍 RO2 — Regional & Spatial Distribution

> *How is AI governance discourse distributed across world regions and countries?*

**5 visualisations:**
1. World Choropleth Map — governance papers by country
2. Co-authorship Network — country collaboration graph
3. Regional Governance Topic Heatmap
4. Dot Map — country positions sized by paper count
5. Radar Chart — regional governance profiles
"""))

cells.append(md("### Visual 2.1 — World Choropleth Map\n**Countries coloured by number of governance papers.** This requires `vega_datasets` with the world_110m topojson."))
cells.append(code("""
# Country paper counts
country_counts = papers[papers['country'] != 'Unknown'].groupby('country').size().reset_index(name='papers')
country_counts['id'] = country_counts['country'].map(COUNTRY_ISO)
country_counts = country_counts.dropna(subset=['id'])
country_counts['id'] = country_counts['id'].astype(int)

# Also add a human-readable name for tooltip
country_counts['country_label'] = country_counts['country']

world_topo = alt.topo_feature(vega_data.world_110m.url, 'countries')

# Background (grey for countries with no data)
background = alt.Chart(world_topo).mark_geoshape(
    fill='#f0f0f0', stroke='white', strokeWidth=0.4
)

# Choropleth
choropleth = alt.Chart(world_topo).mark_geoshape(
    stroke='white', strokeWidth=0.4
).encode(
    color=alt.Color('papers:Q',
                    scale=alt.Scale(scheme='blues', domain=[1, country_counts['papers'].max()]),
                    legend=alt.Legend(title='Governance Papers', orient='bottom',
                                       gradientLength=200, gradientThickness=12)),
    tooltip=[
        alt.Tooltip('country_label:N', title='Country'),
        alt.Tooltip('papers:Q', title='Governance Papers'),
    ]
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(country_counts, 'id', ['papers', 'country_label'])
)

v6 = (background + choropleth).project(
    type='naturalEarth1'
).properties(
    width=800, height=430,
    title=alt.TitleParams(
        'Global Distribution of AI Governance Research (2015–2025)',
        subtitle='Colour intensity = governance paper count | First author country affiliation',
        fontSize=15, subtitleFontSize=11
    )
).configure_view(strokeWidth=0)

v6
"""))

cells.append(md("### Visual 2.2 — Country Co-authorship Network\nCountry nodes sized by paper count, connected by international collaboration edges. Colour = world region."))
cells.append(code("""
# Filter to top N edges for visual clarity
TOP_EDGES = 80
top_e = edges.nlargest(TOP_EDGES, 'weight').copy()

# Build networkx graph
G = nx.Graph()
for _, row in top_e.iterrows():
    G.add_edge(row['country_a'], row['country_b'], weight=float(row['weight']))

# Add isolated top nodes
for c in papers[papers['country'] != 'Unknown']['country'].value_counts().head(20).index:
    G.add_node(c)

# Spring layout weighted by collaboration strength
pos = nx.spring_layout(G, k=2.5, seed=42, weight='weight', iterations=80)

# Region mapping
REGION_MAP = {}
if not nodes.empty and 'country' in nodes.columns and 'region' in nodes.columns:
    REGION_MAP = dict(zip(nodes['country'], nodes['region']))
else:
    # Fallback from papers
    rmap = papers[['country','region']].drop_duplicates()
    REGION_MAP = dict(zip(rmap['country'], rmap['region']))

country_papers = papers[papers['country'] != 'Unknown']['country'].value_counts().reset_index()
country_papers.columns = ['country', 'papers']
cp_map = dict(zip(country_papers['country'], country_papers['papers']))

# Node dataframe
node_df = pd.DataFrame([{
    'country': n,
    'x': pos[n][0],
    'y': pos[n][1],
    'papers': cp_map.get(n, 1),
    'region': REGION_MAP.get(n, 'Other / Unclassified'),
    'degree': sum(G[n][v]['weight'] for v in G[n]) if G.degree(n) > 0 else 1
} for n in G.nodes()])

# Edge dataframe (as individual rows with start/end)
edge_df = pd.DataFrame([{
    'x':  pos[u][0], 'y':  pos[u][1],
    'x2': pos[v][0], 'y2': pos[v][1],
    'weight': G[u][v]['weight'],
    'pair': f'{u} ↔ {v}'
} for u, v in G.edges() if u in pos and v in pos])

# Charts
edges_layer = alt.Chart(edge_df).mark_rule(opacity=0.25).encode(
    x=alt.X('x:Q', axis=None),
    y=alt.Y('y:Q', axis=None),
    x2='x2:Q', y2='y2:Q',
    strokeWidth=alt.StrokeWidth('weight:Q',
                                 scale=alt.Scale(range=[0.3, 4])),
    color=alt.value('#999999'),
    tooltip=[alt.Tooltip('pair:N', title='Collaboration'), alt.Tooltip('weight:Q', title='Papers')]
)

nodes_layer = alt.Chart(node_df).mark_circle(opacity=0.9, stroke='white', strokeWidth=1).encode(
    x=alt.X('x:Q', axis=None),
    y=alt.Y('y:Q', axis=None),
    size=alt.Size('papers:Q',
                  scale=alt.Scale(range=[40, 2200]),
                  legend=alt.Legend(title='Papers', orient='bottom')),
    color=alt.Color('region:N',
                    scale=alt.Scale(domain=REGION_DOMAIN, range=REGION_RANGE),
                    legend=alt.Legend(title='Region', orient='bottom-right')),
    tooltip=[
        alt.Tooltip('country:N', title='Country'),
        alt.Tooltip('region:N', title='Region'),
        alt.Tooltip('papers:Q', title='Governance Papers'),
        alt.Tooltip('degree:Q', title='Collaboration Strength', format='.0f'),
    ]
)

# Labels for top 18 nodes
top_nodes = node_df.nlargest(18, 'papers')
labels_layer = alt.Chart(top_nodes).mark_text(
    fontSize=8, fontWeight='bold', dy=-10, color='#222'
).encode(
    x='x:Q', y='y:Q',
    text='country:N'
)

v7 = (edges_layer + nodes_layer + labels_layer).properties(
    width=720, height=580,
    title=alt.TitleParams(
        'International AI Research Co-authorship Network',
        subtitle=f'Top {TOP_EDGES} country collaboration pairs | Node size = governance papers | Colour = region',
        fontSize=15, subtitleFontSize=11
    )
).configure_view(strokeWidth=0)

v7
"""))

cells.append(md("### Visual 2.3 — Regional Governance Topic Heatmap"))
cells.append(code("""
# Proportion of each region's papers on each governance topic
rg = reg_gov.copy()
rg['proportion_pct'] = (rg['proportion'] * 100).round(3)
rg['short_topic'] = rg['topic_label_finetuned'].str[:30]
rg = rg[rg['region'] != 'Other / Unclassified']

v8 = alt.Chart(rg).mark_rect().encode(
    x=alt.X('region:N',
            axis=alt.Axis(title='', labelAngle=-30, labelFontSize=11)),
    y=alt.Y('short_topic:N',
            sort=alt.SortField('proportion_pct', order='descending'),
            axis=alt.Axis(title='', labelFontSize=10)),
    color=alt.Color('proportion_pct:Q',
                    scale=alt.Scale(scheme='blues'),
                    legend=alt.Legend(title='% of Region Papers')),
    tooltip=[
        alt.Tooltip('topic_label_finetuned:N', title='Topic'),
        alt.Tooltip('region:N', title='Region'),
        alt.Tooltip('proportion_pct:Q', title='% of Region Papers', format='.3f'),
        alt.Tooltip('count:Q', title='Papers'),
    ]
).properties(
    width=380, height=460,
    title=alt.TitleParams(
        'Governance Research Focus by Region',
        subtitle='% = share of each region\\'s total AI papers on each governance topic',
        fontSize=15, subtitleFontSize=11
    )
)

# Add text labels for high values
text_v8 = alt.Chart(rg[rg['proportion_pct'] > 0.3]).mark_text(
    fontSize=8, color='white', fontWeight='bold'
).encode(
    x='region:N',
    y=alt.Y('short_topic:N', sort=alt.SortField('proportion_pct', order='descending')),
    text=alt.Text('proportion_pct:Q', format='.2f')
)

(v8 + text_v8).configure_view(step=24).configure_axis(labelFontSize=10)
"""))

cells.append(md("### Visual 2.4 — Dot Map: Countries Positioned by Coordinates\nEach country at its geographic centroid, sized by governance paper count."))
cells.append(code("""
# Use country nodes file which has centroids
if not nodes.empty and 'latitude' in nodes.columns:
    dot_data = nodes.copy()
    dot_data['papers'] = dot_data['country'].map(cp_map).fillna(0)
    dot_data = dot_data[dot_data['papers'] > 0]
    dot_data['region'] = dot_data['country'].map(REGION_MAP).fillna('Other / Unclassified')
else:
    # Build from papers
    dot_data = papers[papers['country'] != 'Unknown'].groupby('country').size().reset_index(name='papers')
    dot_data['region'] = dot_data['country'].map(REGION_MAP).fillna('Other / Unclassified')
    # Approximate centroids (major countries)
    CENTROIDS = {
        'United States': (-95, 38), 'China': (105, 35), 'United Kingdom': (-2, 54),
        'Germany': (10, 51), 'India': (78, 20), 'Australia': (134, -25),
        'France': (2, 47), 'Japan': (138, 36), 'Canada': (-95, 60),
        'Spain': (-4, 40), 'Italy': (12, 42), 'South Korea': (128, 37),
        'Brazil': (-52, -10), 'Netherlands': (5, 52), 'Saudi Arabia': (45, 24),
        'Malaysia': (110, 3), 'Singapore': (104, 1), 'Turkey': (35, 39),
        'Egypt': (30, 27), 'Nigeria': (8, 10), 'South Africa': (25, -29),
    }
    dot_data['longitude'] = dot_data['country'].map({k:v[0] for k,v in CENTROIDS.items()})
    dot_data['latitude']  = dot_data['country'].map({k:v[1] for k,v in CENTROIDS.items()})
    dot_data = dot_data.dropna(subset=['longitude', 'latitude'])

dot_data['longitude'] = pd.to_numeric(dot_data['longitude'], errors='coerce')
dot_data['latitude']  = pd.to_numeric(dot_data['latitude'], errors='coerce')
dot_data = dot_data.dropna(subset=['longitude','latitude','papers'])

v9 = alt.Chart(dot_data[dot_data['papers'] > 0]).mark_circle(opacity=0.8, stroke='white', strokeWidth=0.5).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    size=alt.Size('papers:Q',
                  scale=alt.Scale(range=[20, 3000]),
                  legend=alt.Legend(title='Governance Papers')),
    color=alt.Color('region:N',
                    scale=alt.Scale(domain=REGION_DOMAIN, range=REGION_RANGE),
                    legend=alt.Legend(title='Region')),
    tooltip=[
        alt.Tooltip('country:N', title='Country'),
        alt.Tooltip('region:N', title='Region'),
        alt.Tooltip('papers:Q', title='Governance Papers'),
    ]
).project(type='naturalEarth1').properties(
    width=760, height=400,
    title=alt.TitleParams(
        'AI Governance Research Output by Country',
        subtitle='Each dot = one country, positioned at geographic centroid | Size = paper count',
        fontSize=15, subtitleFontSize=11
    )
)

background2 = alt.Chart(world_topo).mark_geoshape(
    fill='#f5f5f5', stroke='#ddd', strokeWidth=0.3
).project('naturalEarth1')

(background2 + v9).configure_view(strokeWidth=0)
"""))

cells.append(md("### Visual 2.5 — Regional Stance Profiles (Faceted Bar)"))
cells.append(code("""
# Faceted: one panel per region showing stance breakdown
sr2 = s_region.copy()
sr2 = sr2[sr2['region'] != 'Other / Unclassified']
sr2_melted = sr2.melt(
    id_vars=['region','paper_count','dominant_stance'],
    value_vars=['risk_focused','opportunity_focused','balanced'],
    var_name='stance', value_name='proportion'
)
sr2_melted['proportion_pct'] = (sr2_melted['proportion'] * 100).round(1)
sr2_melted['stance_label'] = sr2_melted['stance'].str.replace('_', ' ').str.title()

v10 = alt.Chart(sr2_melted).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
    x=alt.X('stance_label:N',
            axis=alt.Axis(title='', labelAngle=-30, labelFontSize=10),
            sort=['Risk Focused', 'Balanced', 'Opportunity Focused']),
    y=alt.Y('proportion_pct:Q',
            axis=alt.Axis(title='% of Papers', grid=True, gridOpacity=0.3)),
    color=alt.Color('stance:N',
                    scale=alt.Scale(domain=STANCE_DOMAIN, range=STANCE_RANGE),
                    legend=None),
    tooltip=[
        alt.Tooltip('region:N', title='Region'),
        alt.Tooltip('stance_label:N', title='Stance'),
        alt.Tooltip('proportion_pct:Q', title='%', format='.1f'),
    ]
).facet(
    facet=alt.Facet('region:N', title=''),
    columns=3
).properties(
    title=alt.TitleParams(
        'AI Governance Stance by World Region',
        subtitle='Europe = most risk-focused | Africa & ME = most opportunity-focused',
        fontSize=15, subtitleFontSize=11
    )
)

v10
"""))

# =============================================================================
#  SECTION 3 — RO3: POLICY ALIGNMENT
# =============================================================================

cells.append(md("""---
## 📜 RO3 — Academic vs Policy Alignment

> *Where do academic governance discourse and national AI policy frameworks converge or diverge?*

**5 visualisations:**
1. Alignment Bubble Chart (academic papers vs policy chunks)
2. Connected Dot Plot — shared topic coverage gap
3. Policy Heatmap — country × topic
4. EU Vocabulary Dominance Chart
5. Topic Gap Bar — academic-only topics (absent from policy)
"""))

cells.append(md("### Visual 3.1 — Policy Alignment Bubble Chart\nx = academic papers, y = policy chunks, size = governance score. Shows where each corpus is stronger."))
cells.append(code("""
al = align[align['governance_score'] >= 0.5].copy()
al['short_label'] = al['topic_label'].str[:32]
al['alignment_label'] = al['alignment'].map({
    'shared': '🟦 Shared',
    'academic_only': '⬜ Academic Only',
    'policy_only': '🟧 Policy Only',
    'neither': '⬛ Neither'
})
al['policy_count_plot'] = al['policy_count'].fillna(0)
al['scopus_count_plot'] = al['scopus_count'].fillna(0)

# Diagonal line (equal coverage)
diag_data = pd.DataFrame({'x': [0, 1000], 'y': [0, 1000]})
diag_line2 = alt.Chart(diag_data).mark_line(
    strokeDash=[5,4], color='#aaa', opacity=0.5
).encode(x='x:Q', y='y:Q')

v11 = alt.Chart(al).mark_circle(opacity=0.85, stroke='white', strokeWidth=1).encode(
    x=alt.X('scopus_count_plot:Q',
            scale=alt.Scale(type='log', domain=[1, 2000]),
            axis=alt.Axis(title='Academic Papers (log scale)')),
    y=alt.Y('policy_count_plot:Q',
            scale=alt.Scale(type='log', domain=[1, 600]),
            axis=alt.Axis(title='Policy Document Chunks (log scale)')),
    size=alt.Size('governance_score:Q',
                  scale=alt.Scale(range=[60, 1400]),
                  legend=alt.Legend(title='Gov. Score')),
    color=alt.Color('alignment_label:N',
                    scale=alt.Scale(
                        domain=['🟦 Shared','⬜ Academic Only','🟧 Policy Only'],
                        range=[MID, '#AAAAAA', AMBER]
                    ),
                    legend=alt.Legend(title='Alignment')),
    tooltip=[
        alt.Tooltip('topic_label:N', title='Topic'),
        alt.Tooltip('scopus_count_plot:Q', title='Academic Papers'),
        alt.Tooltip('policy_count_plot:Q', title='Policy Chunks'),
        alt.Tooltip('governance_score:Q', title='Gov. Score', format='.2f'),
        alt.Tooltip('alignment_label:N', title='Alignment'),
    ]
).properties(width=600, height=480)

text_v11 = alt.Chart(al[al['scopus_count_plot'] > 50]).mark_text(
    fontSize=8, dy=-12, color='#333'
).encode(
    x=alt.X('scopus_count_plot:Q', scale=alt.Scale(type='log', domain=[1,2000])),
    y=alt.Y('policy_count_plot:Q', scale=alt.Scale(type='log', domain=[1,600])),
    text='short_label:N'
)

# Region annotations
above_label = alt.Chart(pd.DataFrame({'x':[15], 'y':[300], 'text':['Policy leads →']})).mark_text(
    color=AMBER, fontSize=10, fontStyle='italic', fontWeight='bold'
).encode(x='x:Q', y='y:Q', text='text:N')

below_label = alt.Chart(pd.DataFrame({'x':[400], 'y':[1.5], 'text':['← Academia leads']})).mark_text(
    color=MID, fontSize=10, fontStyle='italic', fontWeight='bold'
).encode(x='x:Q', y='y:Q', text='text:N')

(diag_line2 + v11 + text_v11 + above_label + below_label).properties(
    title=alt.TitleParams(
        'Academic vs Policy Coverage of Governance Topics',
        subtitle='Above diagonal = policy more detailed | Below = academia more detailed | Log scales',
        fontSize=15, subtitleFontSize=11
    )
).configure_view(strokeWidth=1).configure_axis(labelFontSize=10)
"""))

cells.append(md("### Visual 3.2 — Connected Dot Plot: Shared Topic Coverage Gap"))
cells.append(code("""
shared_al = align[(align['alignment'] == 'shared') & (align['governance_score'] >= 0.5)].copy()
shared_al['short'] = shared_al['topic_label'].str[:38]
shared_al = shared_al.sort_values('scopus_count', ascending=True)

# Melt to long format for dumbell
dum = shared_al.melt(
    id_vars=['topic_label','short','governance_score'],
    value_vars=['scopus_count','policy_count'],
    var_name='corpus', value_name='count'
)
dum['corpus_label'] = dum['corpus'].map({
    'scopus_count': 'Academic Papers',
    'policy_count': 'Policy Chunks'
})

# Lines connecting the two points
v12_lines = alt.Chart(shared_al).mark_rule(strokeWidth=2, color='#ccc').encode(
    y=alt.Y('short:N', sort=shared_al['short'].tolist()),
    x=alt.X('scopus_count:Q', axis=alt.Axis(title='Count')),
    x2='policy_count:Q'
)

v12_dots = alt.Chart(dum).mark_circle(size=120, opacity=0.9).encode(
    y=alt.Y('short:N',
            sort=shared_al['short'].tolist(),
            axis=alt.Axis(title='', labelFontSize=10)),
    x=alt.X('count:Q',
            axis=alt.Axis(title='Coverage (papers / chunks)', grid=True, gridOpacity=0.3)),
    color=alt.Color('corpus_label:N',
                    scale=alt.Scale(
                        domain=['Academic Papers', 'Policy Chunks'],
                        range=[MID, AMBER]
                    ),
                    legend=alt.Legend(title='Corpus', orient='top-right')),
    tooltip=[
        alt.Tooltip('topic_label:N', title='Topic'),
        alt.Tooltip('corpus_label:N', title='Corpus'),
        alt.Tooltip('count:Q', title='Count'),
    ]
)

v12 = (v12_lines + v12_dots).properties(
    width=580, height=380,
    title=alt.TitleParams(
        'Coverage Gap: Academic Papers vs Policy Chunks (Shared Topics)',
        subtitle='Wider gaps = greater divergence in depth of engagement',
        fontSize=15, subtitleFontSize=11
    )
).configure_view(strokeWidth=1)

v12
"""))

cells.append(md("### Visual 3.3 — Policy Document Heatmap: Country × Topic"))
cells.append(code("""
if not pol_docs.empty:
    pol = pol_docs.copy()
    pol['proportion_pct'] = pd.to_numeric(pol['proportion'], errors='coerce') * 100
    pol['short_topic'] = pol['topic_label_finetuned'].astype(str).str[:28]
    pol_agg = pol.groupby(['country','short_topic'])['proportion_pct'].max().reset_index()

    v13 = alt.Chart(pol_agg).mark_rect(stroke='white', strokeWidth=0.5).encode(
        x=alt.X('country:N',
                axis=alt.Axis(title='', labelAngle=-45, labelFontSize=9)),
        y=alt.Y('short_topic:N',
                sort=alt.SortField('proportion_pct', order='descending'),
                axis=alt.Axis(title='', labelFontSize=9)),
        color=alt.Color('proportion_pct:Q',
                        scale=alt.Scale(scheme='oranges'),
                        legend=alt.Legend(title='% Chunks')),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('short_topic:N', title='Topic'),
            alt.Tooltip('proportion_pct:Q', title='% of doc chunks', format='.1f'),
        ]
    ).properties(
        width=640, height=300,
        title=alt.TitleParams(
            'National AI Policy Topic Coverage',
            subtitle='Darker = more of that country\\'s policy document covers that governance topic',
            fontSize=15, subtitleFontSize=11
        )
    )

    text_v13 = alt.Chart(pol_agg[pol_agg['proportion_pct'] > 30]).mark_text(
        fontSize=7, color='white', fontWeight='bold'
    ).encode(
        x='country:N',
        y=alt.Y('short_topic:N', sort=alt.SortField('proportion_pct', order='descending')),
        text=alt.Text('proportion_pct:Q', format='.0f')
    )

    (v13 + text_v13).configure_view(step=18).configure_axis(labelFontSize=9)
else:
    print('policy_document_topics_v2.csv not found — skipping visual 3.3')
"""))

cells.append(md("### Visual 3.4 — EU Vocabulary Dominance in Global South\nWhich countries mirror EU AI Act language in their policy frameworks?"))
cells.append(code("""
if not pol_docs.empty:
    eu_topic = pol_docs[
        pol_docs['topic_label_finetuned'].astype(str).str.contains('EU AI Regulation', na=False)
    ].copy()
    eu_topic['proportion_pct'] = pd.to_numeric(eu_topic['proportion'], errors='coerce') * 100
    eu_topic = eu_topic.sort_values('proportion_pct', ascending=True)

    v14 = alt.Chart(eu_topic).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('proportion_pct:Q',
                axis=alt.Axis(title='% of Policy Document Using EU Regulatory Language')),
        y=alt.Y('country:N',
                sort=alt.SortField('proportion_pct', order='ascending'),
                axis=alt.Axis(title='')),
        color=alt.Color('region:N',
                        scale=alt.Scale(domain=REGION_DOMAIN, range=REGION_RANGE),
                        legend=alt.Legend(title='Region')),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('region:N', title='Region'),
            alt.Tooltip('proportion_pct:Q', title='% of document', format='.1f'),
        ]
    ).properties(
        width=560, height=380,
        title=alt.TitleParams(
            'EU Regulatory Vocabulary in National AI Policy Frameworks',
            subtitle='100% = entire document mapped to EU AI Regulation topic | Global South mirrors EU language',
            fontSize=15, subtitleFontSize=11
        )
    )

    ref = alt.Chart(pd.DataFrame({'x': [50]})).mark_rule(
        strokeDash=[4,3], color=RED, opacity=0.5
    ).encode(x='x:Q')

    (v14 + ref).configure_view(strokeWidth=1).configure_axis(labelFontSize=10)
"""))

cells.append(md("### Visual 3.5 — Academic-Only Topics (Policy Gaps)\nGovernance themes well-studied in academia but completely absent from national AI policy frameworks."))
cells.append(code("""
acad_only = align[
    (align['alignment'] == 'academic_only') & (align['governance_score'] >= 0.5)
].copy().sort_values('scopus_count', ascending=True)
acad_only['short'] = acad_only['topic_label'].str[:42]

v15 = alt.Chart(acad_only).mark_bar(
    cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=MID
).encode(
    x=alt.X('scopus_count:Q',
            axis=alt.Axis(title='Academic Papers (no policy equivalent)')),
    y=alt.Y('short:N',
            sort=alt.SortField('scopus_count', order='ascending'),
            axis=alt.Axis(title='', labelFontSize=10)),
    opacity=alt.Opacity('governance_score:Q',
                         scale=alt.Scale(range=[0.4, 1.0]),
                         legend=alt.Legend(title='Gov. Score')),
    tooltip=[
        alt.Tooltip('topic_label:N', title='Topic'),
        alt.Tooltip('scopus_count:Q', title='Academic Papers'),
        alt.Tooltip('governance_score:Q', title='Gov. Score', format='.2f'),
        alt.Tooltip('top_words:N', title='Keywords'),
    ]
)

text_v15 = alt.Chart(acad_only).mark_text(
    align='left', dx=4, fontSize=9, color='#333'
).encode(
    x='scopus_count:Q',
    y=alt.Y('short:N', sort=alt.SortField('scopus_count', order='ascending')),
    text=alt.Text('scopus_count:Q', format='d')
)

(v15 + text_v15).properties(
    width=560, height=360,
    title=alt.TitleParams(
        'Policy Gaps: Governance Topics in Academia But Absent From Policy',
        subtitle='These are areas where governments have not yet formalised AI governance priorities',
        fontSize=15, subtitleFontSize=11
    )
).configure_view(strokeWidth=1).configure_axis(labelFontSize=10)
"""))

# =============================================================================
#  SECTION 4 — STANCE ANALYSIS
# =============================================================================

cells.append(md("""---
## 💬 Stance Analysis — Risk vs Opportunity Framing

> *Do academics frame AI governance as a risk to be managed or an opportunity to be harnessed?*

**5 visualisations:**
1. Diverging Bar — Risk vs Opportunity per topic
2. Slope Chart — Stance shift pre vs post ChatGPT per topic
3. Scatter — Individual paper stance scores
4. Period Comparison — Stacked bar pre vs post
5. Topic × Region Stance Heatmap
"""))

cells.append(md("### Visual 4.1 — Diverging Bar: Risk vs Opportunity per Topic\nTopics sorted by risk score. Red = risk-dominated, green = opportunity-dominated."))
cells.append(code("""
st2 = s_topic.copy().sort_values('avg_risk_score', ascending=False)
st2['short'] = st2['topic_label_finetuned'].str[:40]
# Encode opportunity as negative for diverging layout
st2['opp_neg'] = -st2['opportunity_focused']

risk_bar = alt.Chart(st2).mark_bar(color=RED, opacity=0.85).encode(
    y=alt.Y('short:N',
            sort=st2['short'].tolist(),
            axis=alt.Axis(title='', labelFontSize=9)),
    x=alt.X('risk_focused:Q',
            scale=alt.Scale(domain=[-0.8, 0.8]),
            axis=alt.Axis(title='← Opportunity-focused  |  Risk-focused →',
                           values=[-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8],
                           format='.0%')),
    tooltip=[alt.Tooltip('topic_label_finetuned:N'), alt.Tooltip('risk_focused:Q', format='.1%')]
)
opp_bar = alt.Chart(st2).mark_bar(color=GREEN, opacity=0.85).encode(
    y=alt.Y('short:N', sort=st2['short'].tolist()),
    x=alt.X('opp_neg:Q'),
    tooltip=[alt.Tooltip('topic_label_finetuned:N'), alt.Tooltip('opportunity_focused:Q', format='.1%')]
)
zero_rule = alt.Chart(pd.DataFrame({'x':[0]})).mark_rule(color='#333', strokeWidth=1.5).encode(x='x:Q')

v16 = (risk_bar + opp_bar + zero_rule).properties(
    width=560, height=500,
    title=alt.TitleParams(
        'Risk vs Opportunity Framing by Governance Topic',
        subtitle='Red = risk-focused papers | Green = opportunity-focused papers | Length = proportion',
        fontSize=15, subtitleFontSize=11
    )
).configure_view(strokeWidth=1).configure_axis(labelFontSize=9)

v16
"""))

cells.append(md("### Visual 4.2 — Slope Chart: Pre vs Post ChatGPT Stance Shift"))
cells.append(code("""
# Build topic-level stance by period from papers
stance_period_topic = papers.groupby(['period','topic_label_finetuned','stance']).size().reset_index(name='count')
spt_total = papers.groupby(['period','topic_label_finetuned']).size().reset_index(name='total')
stance_period_topic = stance_period_topic.merge(spt_total, on=['period','topic_label_finetuned'])
stance_period_topic['proportion'] = stance_period_topic['count'] / stance_period_topic['total']

# Focus on risk_focused stance
risk_shift = stance_period_topic[stance_period_topic['stance'] == 'risk_focused'].copy()
risk_shift['period_label'] = risk_shift['period'].map({
    'pre_chatgpt': 'Pre-ChatGPT', 'post_chatgpt': 'Post-ChatGPT'
})
risk_shift['proportion_pct'] = (risk_shift['proportion'] * 100).round(1)
risk_shift['short'] = risk_shift['topic_label_finetuned'].str[:30]

# Compute direction for colouring
pre_  = risk_shift[risk_shift['period'] == 'pre_chatgpt'][['topic_label_finetuned','proportion']].rename(columns={'proportion':'pre'})
post_ = risk_shift[risk_shift['period'] == 'post_chatgpt'][['topic_label_finetuned','proportion']].rename(columns={'proportion':'post'})
direction = pre_.merge(post_, on='topic_label_finetuned')
direction['direction'] = (direction['post'] - direction['pre']).apply(
    lambda x: 'More risk post-ChatGPT' if x > 0.02 else ('Less risk post-ChatGPT' if x < -0.02 else 'Stable')
)
risk_shift = risk_shift.merge(direction[['topic_label_finetuned','direction']], on='topic_label_finetuned', how='left')

lines_v17 = alt.Chart(risk_shift).mark_line(point=True, strokeWidth=2).encode(
    x=alt.X('period_label:O',
            sort=['Pre-ChatGPT','Post-ChatGPT'],
            axis=alt.Axis(title='', labelFontSize=12, labelFontWeight='bold')),
    y=alt.Y('proportion_pct:Q',
            axis=alt.Axis(title='% Risk-Focused Papers', grid=True, gridOpacity=0.3)),
    detail='topic_label_finetuned:N',
    color=alt.Color('direction:N',
                    scale=alt.Scale(
                        domain=['More risk post-ChatGPT','Less risk post-ChatGPT','Stable'],
                        range=[RED, GREEN, '#BBBBBB']
                    ),
                    legend=alt.Legend(title='Direction', orient='top-left')),
    opacity=alt.Opacity('direction:N',
                         scale=alt.Scale(
                             domain=['More risk post-ChatGPT','Less risk post-ChatGPT','Stable'],
                             range=[1, 1, 0.4]
                         ), legend=None),
    tooltip=[alt.Tooltip('topic_label_finetuned:N'), alt.Tooltip('proportion_pct:Q', format='.1f'), alt.Tooltip('direction:N')]
)

labels_v17 = alt.Chart(risk_shift[risk_shift['period'] == 'post_chatgpt']).mark_text(
    align='left', dx=6, fontSize=8
).encode(
    x=alt.X('period_label:O', sort=['Pre-ChatGPT','Post-ChatGPT']),
    y='proportion_pct:Q',
    text='short:N',
    color=alt.Color('direction:N',
                    scale=alt.Scale(
                        domain=['More risk post-ChatGPT','Less risk post-ChatGPT','Stable'],
                        range=[RED, GREEN, '#BBBBBB']
                    ), legend=None),
)

v17 = (lines_v17 + labels_v17).properties(
    width=360, height=560,
    title=alt.TitleParams(
        'Risk Framing Shift: Pre vs Post ChatGPT (by Topic)',
        subtitle='Red = became more risk-focused | Green = became less risk-focused after ChatGPT',
        fontSize=15, subtitleFontSize=11
    )
).configure_view(strokeWidth=0)

v17
"""))

cells.append(md("### Visual 4.3 — Scatter: Individual Paper Stance Scores\nEach dot = one governance paper. Position = (risk score, opportunity score). Colour = classified stance."))
cells.append(code("""
# Sample for performance if needed
sample = papers.sample(min(2000, len(papers)), random_state=42).copy()
sample['score_risk'] = pd.to_numeric(sample['score_risk'], errors='coerce')
sample['score_opportunity'] = pd.to_numeric(sample['score_opportunity'], errors='coerce')
sample = sample.dropna(subset=['score_risk','score_opportunity'])
sample['short_topic'] = sample['topic_label_finetuned'].str[:25]

selection = alt.selection_point(fields=['stance'], bind='legend')

v18 = alt.Chart(sample).mark_circle(size=25, opacity=0.5).encode(
    x=alt.X('score_risk:Q',
            scale=alt.Scale(domain=[0, 0.95]),
            axis=alt.Axis(title='Risk Score', grid=True, gridOpacity=0.2)),
    y=alt.Y('score_opportunity:Q',
            scale=alt.Scale(domain=[0, 0.95]),
            axis=alt.Axis(title='Opportunity Score', grid=True, gridOpacity=0.2)),
    color=alt.Color('stance:N',
                    scale=alt.Scale(domain=STANCE_DOMAIN, range=STANCE_RANGE),
                    legend=alt.Legend(title='Stance')),
    opacity=alt.condition(selection, alt.value(0.6), alt.value(0.05)),
    tooltip=[
        alt.Tooltip('title:N', title='Paper Title'),
        alt.Tooltip('stance:N', title='Stance'),
        alt.Tooltip('short_topic:N', title='Topic'),
        alt.Tooltip('country:N', title='Country'),
        alt.Tooltip('score_risk:Q', title='Risk Score', format='.3f'),
        alt.Tooltip('score_opportunity:Q', title='Opp Score', format='.3f'),
    ]
).add_params(selection).properties(
    width=580, height=500,
    title=alt.TitleParams(
        'Individual Paper Stance Scores (click legend to filter)',
        subtitle=f'n={len(sample):,} governance papers sampled | Click a stance in the legend to highlight',
        fontSize=15, subtitleFontSize=11
    )
)

diag_v18 = alt.Chart(pd.DataFrame({'x':[0,0.95],'y':[0,0.95]})).mark_line(
    strokeDash=[4,4], color='#999', opacity=0.4
).encode(x='x:Q', y='y:Q')

(diag_v18 + v18).configure_view(strokeWidth=1).configure_axis(labelFontSize=10)
"""))

cells.append(md("### Visual 4.4 — Stacked Bar: Stance Distribution Pre vs Post ChatGPT"))
cells.append(code("""
sp2 = s_period.copy()
sp2['period_label'] = sp2['period'].map({
    'pre_chatgpt': 'Pre-ChatGPT\\n(2015–2021)',
    'post_chatgpt': 'Post-ChatGPT\\n(2022–2025)'
})
sp2_melted = sp2.melt(
    id_vars=['period','period_label','paper_count'],
    value_vars=['risk_focused','opportunity_focused','balanced'],
    var_name='stance', value_name='proportion'
)
sp2_melted['proportion_pct'] = (sp2_melted['proportion'] * 100).round(1)
sp2_melted['stance_label'] = sp2_melted['stance'].str.replace('_',' ').str.title()

v19 = alt.Chart(sp2_melted).mark_bar(
    cornerRadiusTopLeft=3, cornerRadiusTopRight=3
).encode(
    x=alt.X('period_label:O',
            axis=alt.Axis(title='', labelFontSize=13, labelFontWeight='bold'),
            sort=['Pre-ChatGPT\\n(2015–2021)', 'Post-ChatGPT\\n(2022–2025)']),
    y=alt.Y('proportion_pct:Q',
            stack='zero',
            axis=alt.Axis(title='% of Papers')),
    color=alt.Color('stance:N',
                    scale=alt.Scale(domain=STANCE_DOMAIN, range=STANCE_RANGE),
                    legend=alt.Legend(title='Stance')),
    order=alt.Order('stance:N', sort='ascending'),
    tooltip=[alt.Tooltip('stance_label:N'), alt.Tooltip('proportion_pct:Q', format='.1f')]
)

text_v19 = alt.Chart(sp2_melted[sp2_melted['proportion_pct'] > 8]).mark_text(
    color='white', fontWeight='bold', fontSize=13
).encode(
    x=alt.X('period_label:O', sort=['Pre-ChatGPT\\n(2015–2021)','Post-ChatGPT\\n(2022–2025)']),
    y=alt.Y('proportion_pct:Q', stack='zero', bandPosition=0.5),
    text=alt.Text('proportion_pct:Q', format='.1f'),
    order=alt.Order('stance:N', sort='ascending'),
)

v19_chart = (v19 + text_v19).properties(
    width=380, height=420,
    title=alt.TitleParams(
        'Governance Research Stance: Pre vs Post ChatGPT',
        subtitle='Risk framing declined from 34.9% → 29.4% after ChatGPT | Opportunity framing surged',
        fontSize=15, subtitleFontSize=11
    )
)

v19_chart
"""))

cells.append(md("### Visual 4.5 — Topic × Region Stance Heatmap\nWhich topics are framed as risks vs opportunities in each region?"))
cells.append(code("""
# Build topic × region avg risk score
topic_region = papers.groupby(['topic_label_finetuned','region']).agg(
    avg_risk=('score_risk','mean'),
    papers=('scopus_id','count')
).reset_index()
topic_region = topic_region[topic_region['region'] != 'Other / Unclassified']
topic_region['short_topic'] = topic_region['topic_label_finetuned'].str[:32]
topic_region['avg_risk'] = pd.to_numeric(topic_region['avg_risk'], errors='coerce')

v20 = alt.Chart(topic_region).mark_rect().encode(
    x=alt.X('region:N',
            axis=alt.Axis(title='', labelAngle=-30, labelFontSize=11)),
    y=alt.Y('short_topic:N',
            sort=alt.SortField('avg_risk', order='descending'),
            axis=alt.Axis(title='', labelFontSize=9)),
    color=alt.Color('avg_risk:Q',
                    scale=alt.Scale(scheme='redblue', reverse=True, domain=[0.15, 0.65]),
                    legend=alt.Legend(title='Avg Risk Score')),
    tooltip=[
        alt.Tooltip('topic_label_finetuned:N', title='Topic'),
        alt.Tooltip('region:N', title='Region'),
        alt.Tooltip('avg_risk:Q', title='Avg Risk Score', format='.3f'),
        alt.Tooltip('papers:Q', title='Papers'),
    ]
).properties(
    width=360, height=460,
    title=alt.TitleParams(
        'Average Risk Framing by Topic and Region',
        subtitle='Red = risk-dominated framing | Blue = opportunity-dominated framing',
        fontSize=15, subtitleFontSize=11
    )
)

text_v20 = alt.Chart(topic_region[topic_region['avg_risk'] > 0.50]).mark_text(
    fontSize=8, color='white', fontWeight='bold'
).encode(
    x='region:N',
    y=alt.Y('short_topic:N', sort=alt.SortField('avg_risk', order='descending')),
    text=alt.Text('avg_risk:Q', format='.2f')
)

(v20 + text_v20).configure_view(step=22).configure_axis(labelFontSize=10)
"""))

# ── EXPORT ────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 💾 Export Charts for Paper

Run the cell below to save all visuals as SVG files in `results/figures/`.
"""))
cells.append(code("""
import os

FIG_DIR = RESULTS / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# To save an Altair chart as SVG:
# chart.save(str(FIG_DIR / 'v1_bubble.svg'))
# chart.save(str(FIG_DIR / 'v1_bubble.png'), scale_factor=2.0)

# Example:
# v1.save(str(FIG_DIR / 'ro1_bubble_chart.svg'))
# v6.save(str(FIG_DIR / 'ro2_world_map.svg'))
# v11.save(str(FIG_DIR / 'ro3_alignment_bubble.svg'))
# v16.save(str(FIG_DIR / 'stance_diverging_bar.svg'))

print(f'Save charts to: {FIG_DIR}')
print('Use: chart.save(str(FIG_DIR / \"filename.svg\"))')
print('Or:  chart.save(str(FIG_DIR / \"filename.png\"), scale_factor=2.0)')
"""))

# =============================================================================
#  WRITE NOTEBOOK
# =============================================================================

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": cells
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook created: {OUT_PATH}")
print(f"   Cells: {len(cells)}")
print()
print("Next steps:")
print("  1. pip install altair vega_datasets pycountry networkx nbformat")
print("  2. Open 6_exploratory_visuals.ipynb in PyCharm")
print("  3. Run cells one by one — each produces one interactive Altair chart")
print()
print("Visuals included:")
print("  RO1: Bubble | Heatmap | Slope | Stacked Area | Governance Score scatter")
print("  RO2: World Choropleth | Co-authorship Network | Heatmap | Dot Map | Radar")
print("  RO3: Alignment Bubble | Connected Dot | Policy Heatmap | EU Dominance | Gaps")
print("  Stance: Diverging Bar | Slope | Paper Scatter | Period Stacked | Topic×Region")