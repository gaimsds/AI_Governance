# =============================================================================
#  NATIONAL AI FRAMEWORKS COLLECTION SCRIPT — VERSION 2
#  Project: Mapping Global AI Governance Narratives
#  Authors: Tambudzai Gundani & Joshua Gray
#  Date:    March 2026
# =============================================================================
#
#  WHAT THIS SCRIPT DOES:
#  ----------------------
#  Collects national AI strategy documents and international governance
#  frameworks from official government and intergovernmental sources.
#
#  PRIMARY REFERENCE SOURCE:
#  --------------------------
#  OECD AI Policy Observatory (https://oecd.ai/en/dashboards/policy-initiatives)
#  A live database maintained by OECD tracking 70+ national AI policies with
#  direct PDF links. Used to verify currency of all documents in this registry.
#  Cite in methodology as:
#  "OECD AI Policy Observatory (oecd.ai) was used to identify and verify
#  national AI policy documents as of March 2026."
#
#  DOCUMENT COUNT: 29 documents across 6 regions + international bodies
#
#  CHANGES FROM VERSION 1:
#  ------------------------
#  + Council of Europe AI Treaty CETS 225 (2024) — first binding AI treaty
#  + UK AI Opportunities Action Plan (2025) — replaces 2021 strategy
#  + China Generative AI Interim Measures (2023) — more current than 2017
#  + South Korea AI Basic Act (2025) — second comprehensive AI law globally
#  + ASEAN AI Governance and Ethics Guide (2024) — covers SE Asia
#  + ASEAN Expanded Guide on Generative AI (2025)
#  + US OMB M-25-21 (2025) — current US federal AI governance directive
#
#  OUTPUT FILES (saved to data_raw/policy_frameworks/):
#  -----------------------------------------------------
#  - [country]_[doc_name].pdf   — downloaded PDF files
#  - policy_manifest.csv        — metadata for all documents
#  - download_report.txt        — what succeeded / what needs manual download
# =============================================================================

import os
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

POLICY_DIR = Path(__file__).parent.parent / "data_raw" / "policy_frameworks"
POLICY_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(POLICY_DIR / "collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
#  DOCUMENT REGISTRY
#  All URLs verified against official sources March 2026.
#  OECD AI Policy Observatory (oecd.ai) used as primary reference.
# =============================================================================

DOCUMENTS = [

    # ── NORTH AMERICA ─────────────────────────────────────────────────────────

    {
        "region": "North America", "country": "United States",
        "issuer": "National Science and Technology Council",
        "doc_name": "National AI R&D Strategic Plan 2023 Update",
        "doc_type": "National Strategy", "year": 2023,
        "url": "https://www.nitrd.gov/pubs/National-Artificial-Intelligence-Research-and-Development-Strategic-Plan-2023-Update.pdf",
        "filename": "USA_National_AI_RD_Strategic_Plan_2023.pdf",
        "notes": "8th update to US federal AI R&D strategy; reaffirms 8 strategies, adds international collaboration pillar",
    },
    {
        "region": "North America", "country": "United States",
        "issuer": "NIST",
        "doc_name": "AI Risk Management Framework 1.0",
        "doc_type": "Framework", "year": 2023,
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "filename": "USA_NIST_AI_RMF_1_0_2023.pdf",
        "notes": "Voluntary risk framework; widely referenced internationally",
    },
    {
        # NEW — most current US federal AI governance directive as of 2025
        "region": "North America", "country": "United States",
        "issuer": "Office of Management and Budget",
        "doc_name": "OMB M-25-21 Accelerating Federal Use of AI 2025",
        "doc_type": "Federal Directive", "year": 2025,
        "url": "https://www.whitehouse.gov/wp-content/uploads/2025/04/M-25-21-Accelerating-Federal-Use-of-AI-through-Innovation-Governance-and-Public-Trust.pdf",
        "filename": "USA_OMB_M2521_Federal_AI_Governance_2025.pdf",
        "notes": "Most current US federal AI governance directive April 2025; innovation, governance, public trust",
        "manual": True,
        "manual_note": "If URL fails search 'OMB M-25-21' on whitehouse.gov",
    },
    {
        "region": "North America", "country": "Canada",
        "issuer": "Government of Canada",
        "doc_name": "Directive on Automated Decision-Making",
        "doc_type": "Regulation", "year": 2023,
        "url": "https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32592",
        "filename": "Canada_Directive_Automated_Decision_Making_2023.pdf",
        "notes": "Mandatory for federal agencies; requires algorithmic impact assessment",
        "manual": True,
        "manual_note": "HTML page — save as PDF from browser at the URL above",
    },

    # ── EUROPE ────────────────────────────────────────────────────────────────

    {
        "region": "Europe", "country": "European Union",
        "issuer": "European Parliament and Council",
        "doc_name": "EU Artificial Intelligence Act Regulation 2024/1689",
        "doc_type": "Binding Regulation", "year": 2024,
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
        "filename": "EU_AI_Act_Regulation_2024_1689.pdf",
        "notes": "First comprehensive binding AI regulation globally; entered force August 2024; most provisions apply August 2026",
    },
    {
        "region": "Europe", "country": "United Kingdom",
        "issuer": "UK Government Office for AI",
        "doc_name": "UK National AI Strategy 2021",
        "doc_type": "National Strategy", "year": 2021,
        "url": "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1020402/National_AI_Strategy_-_PDF_version.pdf",
        "filename": "UK_National_AI_Strategy_2021.pdf",
        "notes": "10-year plan for UK AI leadership; baseline document",
    },
    {
        # NEW — most current UK AI policy document January 2025
        "region": "Europe", "country": "United Kingdom",
        "issuer": "UK Department for Science Innovation and Technology",
        "doc_name": "UK AI Opportunities Action Plan 2025",
        "doc_type": "Action Plan", "year": 2025,
        "url": "https://assets.publishing.service.gov.uk/media/67851771f0528401055d2329/ai_opportunities_action_plan.pdf",
        "filename": "UK_AI_Opportunities_Action_Plan_2025.pdf",
        "notes": "50 recommendations; published January 2025; current UK AI policy position",
    },
    {
        "region": "Europe", "country": "Germany",
        "issuer": "German Federal Government",
        "doc_name": "Germany AI Strategy Update 2023",
        "doc_type": "National Strategy", "year": 2023,
        "url": "https://www.ki-strategie-deutschland.de/files/downloads/2023_AI_Update_barrierefrei.pdf",
        "filename": "Germany_AI_Strategy_Update_2023.pdf",
        "notes": "Update to 2018 strategy; emphasises trustworthy AI and European sovereignty",
    },
    {
        "region": "Europe", "country": "France",
        "issuer": "French Government",
        "doc_name": "France AI for Humanity Strategy 2021",
        "doc_type": "National Strategy", "year": 2021,
        "url": "https://www.gouvernement.fr/upload/media/default/0001/01/2021_10_ia-rapport-bericht-complet.pdf",
        "filename": "France_AI_Strategy_2021.pdf",
        "notes": "Covers research, talent, industrial uptake, ethics and sovereignty",
    },

    # ── ASIA-PACIFIC ──────────────────────────────────────────────────────────

    {
        "region": "Asia-Pacific", "country": "Singapore",
        "issuer": "IMDA / PDPC",
        "doc_name": "Model AI Governance Framework Second Edition 2020",
        "doc_type": "Framework", "year": 2020,
        "url": "https://www.imda.gov.sg/-/media/imda/files/infocomm-media-landscape/sg-digital/tech-pillars/artificial-intelligence/second-edition-of-the-model-ai-governance-framework.pdf",
        "filename": "Singapore_Model_AI_Governance_Framework_2020.pdf",
        "notes": "Voluntary framework for private sector; internationally recognised",
    },
    {
        "region": "Asia-Pacific", "country": "Singapore",
        "issuer": "AI Verify Foundation",
        "doc_name": "Model AI Governance Framework for Generative AI 2024",
        "doc_type": "Framework", "year": 2024,
        "url": "https://aiverifyfoundation.sg/wp-content/uploads/2024/05/Model-AI-Governance-Framework-for-Generative-AI-May-2024-1-1.pdf",
        "filename": "Singapore_Model_AI_Governance_GenAI_2024.pdf",
        "notes": "Extends 2020 framework to cover generative AI; released May 2024",
    },
    {
        "region": "Asia-Pacific", "country": "India",
        "issuer": "NITI Aayog",
        "doc_name": "National Strategy for Artificial Intelligence 2021",
        "doc_type": "National Strategy", "year": 2021,
        "url": "https://www.niti.gov.in/sites/default/files/2023-03/National-Strategy-for-Artificial-Intelligence.pdf",
        "filename": "India_National_AI_Strategy_2021.pdf",
        "notes": "AI for All vision; healthcare, agriculture, education, smart cities focus",
    },
    {
        "region": "Asia-Pacific", "country": "Australia",
        "issuer": "Department of Industry Science and Resources",
        "doc_name": "Australia AI Ethics Framework 2019",
        "doc_type": "Framework", "year": 2019,
        "url": "https://www.industry.gov.au/sites/default/files/2019-11/australias-artificial-intelligence-ethics-framework.pdf",
        "filename": "Australia_AI_Ethics_Framework_2019.pdf",
        "notes": "8 voluntary AI ethics principles for Australian organisations",
    },
    {
        "region": "Asia-Pacific", "country": "Japan",
        "issuer": "Cabinet Office Japan",
        "doc_name": "Japan AI Strategy 2022",
        "doc_type": "National Strategy", "year": 2022,
        "url": "https://www8.cao.go.jp/cstp/ai/aistratagy2022en.pdf",
        "filename": "Japan_AI_Strategy_2022.pdf",
        "notes": "Human-centred AI; industrial transformation and security focus",
    },
    {
        "region": "Asia-Pacific", "country": "China",
        "issuer": "State Council of China",
        "doc_name": "New Generation AI Development Plan 2017",
        "doc_type": "National Strategy", "year": 2017,
        "url": "https://cset.georgetown.edu/wp-content/uploads/t0112_next_gen_ai_development_plan_EN.pdf",
        "filename": "China_New_Generation_AI_Development_Plan_2017.pdf",
        "notes": "Foundational Chinese AI strategy translated by CSET Georgetown; sets 2030 vision",
    },
    {
        # NEW — China's first binding generative AI regulation
        "region": "Asia-Pacific", "country": "China",
        "issuer": "Cyberspace Administration of China",
        "doc_name": "China Interim Measures Generative AI Services 2023",
        "doc_type": "Binding Regulation", "year": 2023,
        "url": "https://www.airuniversity.af.edu/Portals/10/CASI/documents/Translations/2023-08-07%20ITOW%20Interim%20Measures%20for%20the%20Management%20of%20Generative%20Artificial%20Intelligence%20Services.pdf",
        "filename": "China_Interim_Measures_Generative_AI_2023.pdf",
        "notes": "First binding Chinese regulation on generative AI; effective August 2023; translated by US Air Force CASI",
    },
    {
        # NEW — South Korea's comprehensive AI law
        "region": "Asia-Pacific", "country": "South Korea",
        "issuer": "National Assembly of Korea",
        "doc_name": "South Korea AI Basic Act Framework Act 2025",
        "doc_type": "National Law", "year": 2025,
        "url": "https://cset.georgetown.edu/wp-content/uploads/t0625_south_korea_ai_law_EN.pdf",
        "filename": "SouthKorea_AI_Basic_Act_2025.pdf",
        "notes": "Second comprehensive national AI law globally after EU AI Act; consolidates 19 bills; effective January 2026; translated by CSET Georgetown",
    },

    # ── LATIN AMERICA ─────────────────────────────────────────────────────────

    {
        "region": "Latin America", "country": "Brazil",
        "issuer": "Ministry of Science Technology and Innovation",
        "doc_name": "Brazilian AI Strategy EBIA 2021",
        "doc_type": "National Strategy", "year": 2021,
        "url": "https://www.gov.br/mcti/pt-br/acompanhe-o-mcti/transformacaodigital/arquivosinteligenciaartificial/ebia-ingles_final.pdf",
        "filename": "Brazil_AI_Strategy_EBIA_2021.pdf",
        "notes": "National strategy covering investment, ethics, governance and inclusion",
    },
    {
        "region": "Latin America", "country": "Chile",
        "issuer": "Government of Chile",
        "doc_name": "Chile National AI Policy 2021",
        "doc_type": "National Strategy", "year": 2021,
        "url": "https://minciencia.gob.cl/uploads/filer_public/7d/e3/7de3f3a4-f43d-4568-b7de-a037269e234b/politica_nacional_de_inteligencia_artificial_version_en.pdf",
        "filename": "Chile_National_AI_Policy_2021.pdf",
        "notes": "First Latin American country with a national AI strategy",
    },
    {
        "region": "Latin America", "country": "Colombia",
        "issuer": "Departamento Nacional de Planeacion",
        "doc_name": "Colombia National AI Policy CONPES 4144 2025",
        "doc_type": "National Policy", "year": 2025,
        "url": "https://colaboracion.dnp.gov.co/CDT/Conpes/Econ%C3%B3micos/4144.pdf",
        "filename": "Colombia_National_AI_Policy_CONPES4144_2025.pdf",
        "notes": "Approved February 2025; 106 actions to 2030; ethics, data, R&D, talent, risk mitigation — written in Spanish",
    },
    {
        "region": "Latin America", "country": "Mexico",
        "issuer": "Government of Mexico",
        "doc_name": "Mexico Towards an AI National Agenda 2018",
        "doc_type": "Policy Agenda", "year": 2018,
        "url": "https://ia2030.mx/docs/IA-report-mexico.pdf",
        "filename": "Mexico_AI_National_Agenda_2018.pdf",
        "notes": "Consultative policy document; early Latin American AI governance thinking",
    },

    # ── AFRICA & MIDDLE EAST ──────────────────────────────────────────────────

    {
        "region": "Africa & Middle East", "country": "Rwanda",
        "issuer": "Ministry of ICT and Innovation",
        "doc_name": "National AI Policy Rwanda 2023",
        "doc_type": "National Strategy", "year": 2023,
        "url": "https://www.minict.gov.rw/index.php?eID=dumpFile&t=f&f=67550&token=6195a53203e197efa47592f40ff4aaf24579640e",
        "filename": "Rwanda_National_AI_Policy_2023.pdf",
        "notes": "Cabinet approved April 2023; first comprehensive AI policy in East Africa",
    },
    {
        "region": "Africa & Middle East", "country": "Kenya",
        "issuer": "Ministry of ICT",
        "doc_name": "Kenya National AI Strategy Draft 2025",
        "doc_type": "National Strategy", "year": 2025,
        "url": "https://ict.go.ke/sites/default/files/2025-01/Kenya%20National%20AI%20Strategy%20(Draft)%20for%20Public%20Validation%20%20%5B14-01-2025%5D.pdf",
        "filename": "Kenya_National_AI_Strategy_Draft_2025.pdf",
        "notes": "Draft for public validation January 2025; covers readiness, governance, ethics",
    },
    {
        "region": "Africa & Middle East", "country": "Egypt",
        "issuer": "Ministry of Communications and IT",
        "doc_name": "Egypt National AI Strategy 2021",
        "doc_type": "National Strategy", "year": 2021,
        "url": "https://mcit.gov.eg/Upcont/Documents/Publications_672021000_Egypt-National-AI-Strategy-English.pdf",
        "filename": "Egypt_National_AI_Strategy_2021.pdf",
        "notes": "4 pillars: AI for government, development, capacity, regional cooperation",
    },
    {
        "region": "Africa & Middle East", "country": "UAE",
        "issuer": "UAE Government",
        "doc_name": "UAE National AI Strategy 2031",
        "doc_type": "National Strategy", "year": 2022,
        "url": "https://ai.gov.ae/wp-content/uploads/2022/07/UAE-National-Strategy-for-Artificial-Intelligence-2031.pdf",
        "filename": "UAE_National_AI_Strategy_2031.pdf",
        "notes": "Aims to make UAE a global AI hub; education, economy, government services",
    },
    {
        "region": "Africa & Middle East", "country": "South Africa",
        "issuer": "Department of Communications and Digital Technologies",
        "doc_name": "South Africa National AI Policy Framework 2023",
        "doc_type": "Policy Framework", "year": 2023,
        "url": "https://www.dcdt.gov.za/images/phocadownload/2024/DCDT%20AI%20Policy%20Framework%20August%202023.pdf",
        "filename": "SouthAfrica_National_AI_Policy_Framework_2023.pdf",
        "notes": "Inclusive AI framework; addresses digital divide and developmental priorities",
    },
    {
        "region": "Africa & Middle East", "country": "Zimbabwe",
        "issuer": "Government of Zimbabwe",
        "doc_name": "Zimbabwe National AI Strategy",
        "doc_type": "National Strategy", "year": 2024,
        "url": "https://veritaszim.net/sites/veritas_d/files/Zimbabwe%20National%20Artificial%20Intelligence%20Strategy.pdf",
        "filename": "Zimbabwe_National_AI_Strategy.pdf",
        "notes": "National AI strategy covering economic development, ethics, governance and digital transformation",
    },
    {
        "region": "Africa & Middle East", "country": "Nigeria",
        "issuer": "NCAIR / NITDA",
        "doc_name": "Nigeria National AI Strategy Draft 2024",
        "doc_type": "National Strategy", "year": 2024,
        "url": "https://ncair.nitda.gov.ng/wp-content/uploads/2024/08/National-AI-Strategy_01082024-copy.pdf",
        "filename": "Nigeria_National_AI_Strategy_2024.pdf",
        "notes": "Draft released August 2024; most populous African nation; 5 strategic pillars covering innovation, ethics, infrastructure",
    },

    # ── INTERNATIONAL BODIES ──────────────────────────────────────────────────

    {
        "region": "International", "country": "UNESCO",
        "issuer": "UNESCO",
        "doc_name": "UNESCO Recommendation on the Ethics of AI 2021",
        "doc_type": "International Recommendation", "year": 2021,
        "url": "https://unesdoc.unesco.org/ark:/48223/pf0000381137",
        "filename": "UNESCO_Recommendation_Ethics_AI_2021.pdf",
        "notes": "Adopted by all 193 UNESCO member states; first global normative AI instrument",
        "manual": True,
        "manual_note": "Download PDF from UNESCO digital library at the URL above",
    },
    {
        "region": "International", "country": "OECD",
        "issuer": "OECD",
        "doc_name": "OECD AI Principles 2023",
        "doc_type": "International Principles", "year": 2023,
        "url": "https://legalinstruments.oecd.org/api/download/?uri=/instruments/OECD-LEGAL-0449.pdf",
        "filename": "OECD_AI_Principles_2023.pdf",
        "notes": "Adopted by 44 countries; foundation for many national strategies. Full policy database: oecd.ai",
    },
    {
        "region": "International", "country": "G7",
        "issuer": "G7",
        "doc_name": "G7 Hiroshima AI Process Guiding Principles 2023",
        "doc_type": "International Principles", "year": 2023,
        "url": "https://www.meti.go.jp/press/2023/10/20231030002/20231030002-1.pdf",
        "filename": "G7_Hiroshima_AI_Process_Guiding_Principles_2023.pdf",
        "notes": "11 principles for advanced AI developers; G7 Hiroshima Summit October 2023",
    },
    {
        # NEW — first legally binding international AI treaty; entered force November 2025
        "region": "International", "country": "Council of Europe",
        "issuer": "Council of Europe",
        "doc_name": "Framework Convention on AI Human Rights Democracy Rule of Law CETS 225",
        "doc_type": "Binding International Treaty", "year": 2024,
        "url": "https://rm.coe.int/1680afae3c",
        "filename": "CouncilOfEurope_AI_Framework_Convention_CETS225_2024.pdf",
        "notes": "First legally binding international AI treaty; signed by EU US UK 41 countries; entered force November 2025",
    },
    {
        "region": "International", "country": "African Union",
        "issuer": "African Union",
        "doc_name": "AU Continental AI Strategy for Africa 2024",
        "doc_type": "Continental Strategy", "year": 2024,
        "url": "https://au.int/sites/default/files/documents/43626-doc-AU_AI_Continental_Strategy_For_Africa_draft_April2024.pdf",
        "filename": "AfricanUnion_Continental_AI_Strategy_2024.pdf",
        "notes": "First continental AI strategy for Africa; covers all 55 AU member states",
    },
    {
        # NEW — ASEAN regional AI governance guide covering 10 SE Asian nations
        "region": "International", "country": "ASEAN",
        "issuer": "ASEAN Secretariat",
        "doc_name": "ASEAN Guide on AI Governance and Ethics 2024",
        "doc_type": "Regional Framework", "year": 2024,
        "url": "https://asean.org/wp-content/uploads/2024/02/ASEAN-Guide-on-AI-Governance-and-Ethics_beautified_201223_v2.pdf",
        "filename": "ASEAN_Guide_AI_Governance_Ethics_2024.pdf",
        "notes": "Covers all 10 ASEAN member states; voluntary; 7 core principles; released February 2024",
    },
    {
        # NEW — ASEAN expanded guide for generative AI January 2025
        "region": "International", "country": "ASEAN",
        "issuer": "ASEAN Secretariat",
        "doc_name": "ASEAN Expanded Guide AI Governance Generative AI 2025",
        "doc_type": "Regional Framework", "year": 2025,
        "url": "https://asean.org/wp-content/uploads/2025/01/Expanded-ASEAN-Guide-on-AI-Governance-and-Ethics-Generative-AI.pdf",
        "filename": "ASEAN_Expanded_Guide_AI_Governance_GenAI_2025.pdf",
        "notes": "Extends 2024 ASEAN guide specifically for generative AI risks; January 2025",
    },
]


# =============================================================================
#  DOWNLOAD FUNCTION
# =============================================================================

def download_pdf(url: str, filepath: Path, doc_name: str) -> bool:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 GWU-AI-Governance-Research/2.0",
        "Accept": "application/pdf,*/*",
    }
    try:
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            if filepath.stat().st_size < 1000:
                filepath.unlink()
                return False
            return True
        log.warning(f"   HTTP {response.status_code} for {doc_name}")
        return False
    except Exception as e:
        log.warning(f"   Download error for {doc_name}: {e}")
        return False


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_collection_pipeline():

    log.info("=" * 65)
    log.info("  NATIONAL AI FRAMEWORKS COLLECTION PIPELINE v2")
    log.info(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"  Documents to collect: {len(DOCUMENTS)}")
    log.info("  Reference: OECD AI Policy Observatory (oecd.ai)")
    log.info("=" * 65)

    results = []
    successful = 0
    manual_needed = 0
    failed = 0

    for i, doc in enumerate(DOCUMENTS, 1):
        filepath  = POLICY_DIR / doc["filename"]
        is_manual = doc.get("manual", False)

        log.info(f"")
        log.info(f"[{i:02d}/{len(DOCUMENTS):02d}] {doc['country']} — {doc['doc_name']}")

        if filepath.exists() and filepath.stat().st_size > 1000:
            log.info(f"   ✅ Already downloaded — skipping")
            results.append({**doc, "status": "already_exists", "filepath": str(filepath)})
            successful += 1
            continue

        if is_manual:
            log.info(f"   📋 MANUAL DOWNLOAD REQUIRED")
            log.info(f"   URL: {doc['url']}")
            log.info(f"   Save as: {doc['filename']}")
            if doc.get("manual_note"):
                log.info(f"   Note: {doc['manual_note']}")
            results.append({**doc, "status": "manual_required", "filepath": str(filepath)})
            manual_needed += 1
            continue

        log.info(f"   ⬇️  Downloading...")
        success = download_pdf(doc["url"], filepath, doc["doc_name"])

        if success:
            size_kb = filepath.stat().st_size // 1024
            log.info(f"   ✅ Downloaded ({size_kb:,} KB) → {doc['filename']}")
            results.append({**doc, "status": "downloaded", "filepath": str(filepath)})
            successful += 1
        else:
            log.warning(f"   ❌ Auto-download failed — manual download required")
            log.warning(f"   URL: {doc['url']}")
            results.append({**doc, "status": "failed_try_manual", "filepath": str(filepath)})
            failed += 1

        time.sleep(2)

    # ── Save manifest ──────────────────────────────────────────────────────────
    manifest_df = pd.DataFrame(results)
    manifest_df = manifest_df[[
        "region", "country", "issuer", "doc_name", "doc_type",
        "year", "filename", "status", "notes", "url"
    ]]
    manifest_path = POLICY_DIR / "policy_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")
    log.info(f"\n✅ Manifest saved: {manifest_path.name}")

    # ── Save download report ───────────────────────────────────────────────────
    report_path = POLICY_DIR / "download_report.txt"
    with open(report_path, "w") as f:
        f.write("NATIONAL AI FRAMEWORKS — DOWNLOAD REPORT v2\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("Reference: OECD AI Policy Observatory (oecd.ai)\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Total documents:          {len(DOCUMENTS)}\n")
        f.write(f"Successfully downloaded:  {successful}\n")
        f.write(f"Manual download required: {manual_needed + failed}\n\n")

        manual_docs = [r for r in results if r["status"] in ("manual_required", "failed_try_manual")]
        if manual_docs:
            f.write("DOCUMENTS REQUIRING MANUAL DOWNLOAD:\n")
            f.write("-" * 65 + "\n")
            for doc in manual_docs:
                f.write(f"\nCountry:  {doc['country']}\n")
                f.write(f"Document: {doc['doc_name']}\n")
                f.write(f"Save as:  {doc['filename']}\n")
                f.write(f"URL:      {doc['url']}\n")
                if doc.get("manual_note"):
                    f.write(f"Note:     {doc['manual_note']}\n")

        f.write("\n\nMETHODOLOGY NOTE:\n")
        f.write("All document URLs were verified using the OECD AI Policy Observatory\n")
        f.write("(https://oecd.ai/en/dashboards/policy-initiatives) as primary reference\n")
        f.write("source in March 2026. Currency was confirmed for each document.\n")

    log.info(f"✅ Report saved:   {report_path.name}")

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info("  COLLECTION SUMMARY")
    log.info("=" * 65)
    log.info(f"  Total documents:          {len(DOCUMENTS)}")
    log.info(f"  Successfully downloaded:  {successful}")
    log.info(f"  Manual download needed:   {manual_needed + failed}")
    log.info("")

    region_counts = {}
    for r in results:
        region = r["region"]
        region_counts[region] = region_counts.get(region, 0) + 1
    for region, count in sorted(region_counts.items()):
        log.info(f"    {region:<25} {count} documents")

    log.info("")
    log.info(f"  📁 Files saved to: {POLICY_DIR}")
    log.info(f"  📋 Check download_report.txt for manual download instructions")
    log.info(f"  🌐 Additional documents: https://oecd.ai/en/dashboards/policy-initiatives")
    log.info("=" * 65)
    log.info("")


if __name__ == "__main__":
    run_collection_pipeline()