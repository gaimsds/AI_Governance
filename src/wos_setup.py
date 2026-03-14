
import os
import time
import clarivate.wos_starter.client
from clarivate.wos_starter.client.rest import ApiException
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()
# Defining the host is optional and defaults to http://api.clarivate.com/apis/wos-starter/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = clarivate.wos_starter.client.Configuration(
    host = "https://api.clarivate.com/apis/wos-starter/v2"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: ClarivateApiKeyAuth
api_key = os.getenv("WOS_API_KEY")

if not api_key:
    raise ValueError("WOS_API_KEY not found in environment variables.")

configuration.api_key['ClarivateApiKeyAuth'] = api_key

print("Loaded key:", api_key[:6], "...")

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
#configuration.api_key_prefix['ClarivateApiKeyAuth'] = 'Bearer'


# Enter a context with an instance of the API client
with clarivate.wos_starter.client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = clarivate.wos_starter.client.DocumentsApi(api_client)
    q = 'WC=("Artificial Intelligence") AND TS=("artificial intelligence")'
    db = 'WOS'
    limit = 10 # int | set the limit of records on the page (1-50) (optional) (default to 10)
    page = 1 # int | set the result page (optional) (default to 1)
    sort_field = 'LD+D'
    modified_time_span = None
    tc_modified_time_span = None
    detail = None # str | it will returns the full data by default, if detail=short it returns the limited data (optional)

    try:
        # Query Web of Science documents
        api_response = api_instance.documents_get(q, db=db, limit=limit, page=page,
                                                  sort_field=sort_field, modified_time_span=modified_time_span,
                                                  tc_modified_time_span=tc_modified_time_span, detail=detail)
        print("The response of DocumentsApi->documents_get:\n")
        # Convert response object to full dictionary
        response_dict = api_response.to_dict()

        print("\nTop-level keys in response:")
        print(response_dict.keys())

        # Save full structured data
        with open("wos_full_response.json", "w") as f:
            import json
            json.dump(response_dict, f, indent=2)

        print("\nFull response saved to wos_full_response.json")

        # Store documents in a separate structure
        documents = response_dict.get("hits", [])

        print(f"\nNumber of documents returned: {len(documents)}")

        # Example: Inspect metadata keys for first document
        if documents:
            print("\nKeys available in first document:")
            print(documents[0].keys())

            # Save just documents list separately
            with open("wos_documents_only.json", "w") as f:
                json.dump(documents, f, indent=2)

            print("Document list saved to wos_documents_only.json")
    except ApiException as e:
        print("Exception when calling DocumentsApi->documents_get: %s\n" % e)
        pprint(e.body)
