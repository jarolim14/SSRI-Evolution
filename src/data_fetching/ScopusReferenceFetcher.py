import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any


class ScopusReferenceFetcher:
    """
    Class to fetch references from Scopus.
    Set up before using in a loop with the api key.
    """

    def __init__(self, api_key_info: Dict[str, Any]):
        """Initialize the class with the api key info.
        Args:
            api_key_info (Dict[str, Any]): Dictionary containing API key information including the key and rate limit.
        """
        self.headers = {
            "Accept": "application/json",
            "X-ELS-APIKey": api_key_info["key"],
        }
        self.rate_limit = api_key_info["rate_limit"]
        self.description = api_key_info["description"]
        self.key_name = api_key_info["key_name"]

    def setup_session(self) -> requests.Session:
        """Set up a session with retries.
        Retries only if the status code is 500, 502, 503 or 504 (server errors)
        Returns:
            session: Session with retries."""
        session = requests.Session()
        retries = Retry(
            total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def process_response(self, response: requests.Response) -> Tuple[List[Dict], int]:
        """
        Process the response from the API.
        Retrieve the references and the total number of references.
        Args:
            response (Response): Response from the API.
        Returns:
            tuple: (List of references, Total number of references)
        """
        if response.status_code == 429:
            raise requests.exceptions.HTTPError(
                f"Rate limit exceeded for API key {self.key_name}"
            )

        response.raise_for_status()
        results = response.json()
        # some eids dont have references, so we need to check for that.
        abstract_retrieval_response = results.get("abstracts-retrieval-response")

        if not abstract_retrieval_response:
            return [], 0

        ref_list = abstract_retrieval_response.get("references", {}).get(
            "reference", []
        )
        total_reference_count = int(
            abstract_retrieval_response.get("references", {}).get(
                "@total-references", 0
            )
        )
        return ref_list, total_reference_count

    def request_eid(
        self, eid: str, id_type: str = "eid", view: str = "REF", batch_size: int = 40
    ) -> List[Dict]:
        """
        Request the references for a given eid.
        Args:
            eid (str): EID of the article.
            view (str): View of the references. Defaults to "REF".
            batch_size (int): Batch size of the references. Defaults to 40.
        Returns:
            list: List of references.
        """
        session = self.setup_session()
        url = f"https://api.elsevier.com/content/abstract/{id_type}/{eid}"
        params = {"view": view, "startref": 0, "count": batch_size}

        try:
            response = session.get(url, params=params, headers=self.headers)
            ref_list, total_reference_count = self.process_response(response)

            if total_reference_count <= len(ref_list):
                return ref_list

            all_refs = [ref_list]
            start_ref = len(ref_list)

            while start_ref < total_reference_count:
                params["startref"] = start_ref
                response = session.get(url, params=params, headers=self.headers)
                ref_list, _ = self.process_response(response)
                all_refs.append(ref_list)
                start_ref += len(ref_list)

            # Flatten list
            return [item for sublist in all_refs for item in sublist]

        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                raise requests.exceptions.HTTPError(
                    f"Rate limit exceeded for API key {self.key_name}"
                )
            raise
