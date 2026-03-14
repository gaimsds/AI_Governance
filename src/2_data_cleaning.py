import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import requests
import time
#%%
load_dotenv()

EMAIL = os.getenv("EMAIL_J")

# Load your dataset
df = pd.read_parquet("../data_raw/open_alex_abs_data.parquet")

# Build institution counts
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

# Initialize results dataframe
institution_df = pd.DataFrame(columns=[
    "institution",
    "count",
    "city",
    "state",
    "country",
    "latitude",
    "longitude"
])

# Query OpenAlex for each unique institution
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

# Populate dataframe and save
institution_df = pd.DataFrame(rows, columns=institution_df.columns)
institution_df.to_parquet("../data_raw/institution_geo.parquet", index=False)
print(f"Done. {len(institution_df)} unique institutions saved.")

#%%
df = pd.read_parquet("data_raw/institution_geo.parquet")
df.replace('', np.nan, inplace=True)
print(df.info())