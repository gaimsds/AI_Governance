import requests
import json

# ======================================================================================================================
# Configuration
# ======================================================================================================================
API_KEY = "8bbae266b21f6a115f07c05b085c0371b9e7f53b"
BASE_URL = "https://api.clarivate.com/apis/wos-starter/v1"

headers = {
    "X-ApiKey": API_KEY,
    "Accept": "application/json"
}

params = {
    "q": 'TS="artificial intelligence"',
    "page": 1
}

# ======================================================================================================================
# Making Request
# ======================================================================================================================
response = requests.get(BASE_URL, headers=headers, params=params)

# Raise error if request failed
response.raise_for_status()
data = response.json()

# ======================================================================================================================
# Inspecting Metadata
# ======================================================================================================================
print("Total Hits:", data.get("metadata", {}).get("total"))

records = data.get("hits", [])

print(f"\nReturned {len(records)} records.\n")

# ======================================================================================================================
# Printing Key Fields
# ======================================================================================================================
for i, record in enumerate(records, start=1):
    print(f"--- Document {i} ---")

    print("Title:", record.get("title"))
    print("Authors:", record.get("names"))
    print("Journal:", record.get("source", {}).get("title"))
    print("ISSN:", record.get("source", {}).get("issn"))
    print("Publication Year:", record.get("publicationYear"))
    print("DOI:", record.get("doi"))
    print("Document Type:", record.get("documentType"))
    print()

# ======================================================================================================================
# Saving Full Raw JSON
# ======================================================================================================================
with open("wos_sample_response.json", "w") as f:
    json.dump(data, f, indent=2)

print("Full response saved to wos_sample_response.json")