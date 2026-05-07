"""
========================================================================================================================
OPENALEX INSTITUTION GEOCODING (EXPLORATORY — NOT USED IN FINAL ANALYSIS)
Project: Uneven Science–Policy Translation Shapes Global AI Governance
Authors: Tambudzai G. Charumbira & Joshua Gray
Institution: George Washington University, CCAS | M.S. Data Science | Spring 2026
Date: February 2026

DESCRIPTION:
This script geocoded institutions from the exploratory OpenAlex dataset by querying the OpenAlex Institutions API.
It was part of the early data evaluation phase and was not used in the final analysis. The final geographic
attribution used Nominatim geocoding on the Scopus corpus (see 2b_institution_geocoding_scopus.py).

The output (openalex_institution_geo.parquet) was used only for initial comparison of data source quality
between OpenAlex and Scopus, which informed the decision to proceed with Scopus exclusively.

NOTE: This script is not required to reproduce the final results.
========================================================================================================================
"""
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import requests
import time
#%%
load_dotenv()
EMAIL = os.getenv("EMAIL_J")

df = pd.read_parquet("../data_raw/open_alex_abs_data.parquet")

institution_counts = (
    df["institutions"]
    .str.split(";")
    .explode()
    .str.strip()
    .value_counts()
    .reset_index()
)
institution_counts.columns = ["institution", "count"]
print(institution_counts)

institution_df = pd.DataFrame(columns=[
    "institution",
    "count",
    "city",
    "state",
    "country",
    "latitude",
    "longitude"
])

rows = []
for _, row in institution_counts.iterrows():
    institution = row["institution"]
    count = row["count"]

    if not institution:
        continue

    url = (
        f"https://api.openalex.org/institutions"
        f"?filter=display_name.search:{requests.utils.quote(institution)}"
        f"&mailto={EMAIL}"
    )

    response = requests.get(url)
    city, state, country, latitude, longitude = "", "", "", None, None

    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            geo = results[0].get("geo", {})
            city = geo.get("city", "")
            state = geo.get("region", "")
            country = geo.get("country", "")
            latitude = geo.get("latitude", None)
            longitude = geo.get("longitude", None)

    rows.append({
        "institution": institution,
        "count": count,
        "city": city,
        "state": state,
        "country": country,
        "latitude": latitude,
        "longitude": longitude
    })

    print(f"{institution} queried. {len(rows)} rows saved out of {len(institution_counts)}.")
    time.sleep(1)

institution_df = pd.DataFrame(rows, columns=institution_df.columns)
institution_df.to_parquet("../data_raw/openalex_institution_geo.parquet", index=False)
print(f"Done. {len(institution_df)} unique institutions saved.")

#%%
df = pd.read_parquet("data_raw/openalex_institution_geo.parquet")
df.replace('', np.nan, inplace=True)
print(df.info())