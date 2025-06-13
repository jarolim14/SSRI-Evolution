# Fetch Scopus Data
"""
Resouces:

- [Scopus Search API](https://dev.elsevier.com/documentation/ScopusSearchAPI.wadl)
- [Scopus Retrieval of more than 5,000 artilcles](https://dev.elsevier.com/support.html)(Q: How can I obtain more than 5,000 / 6,000 results through Scopus / ScienceDirect APIs?)
- [Interactive Scopus API](https://dev.elsevier.com/scopus.html)
- [API Settings (rate limits)](https://dev.elsevier.com/api_key_settings.html)
- Remember Logging In to Cisco VPN!!!
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


class ScopusArticleFetcher:
    """
    A class to search and retrieve data from the Scopus API.

    Attributes:
        api_keys (List[str]): List of API keys for Scopus API.
        base_url (str): Base URL for the Scopus API.
        output_dir (str): Directory to save results.
        logger (logging.Logger): Logger instance for tracking errors and events.
    """

    def __init__(self, api_keys: List[str], output_dir: str = "../data/01-raw/scopus"):
        """
        Initializes ScopusSearcher with API keys and output directory.

        Args:
            api_keys (List[str]): List of API keys for Scopus API.
            output_dir (str): Directory to save results.
        """
        self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]
        self.current_api_key_index = 0
        self.base_url = "https://api.elsevier.com/content/search/scopus"
        self.headers = {"Accept": "application/json"}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """Sets up logging configuration for the fetcher."""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.output_dir) / "logs"
        log_dir.mkdir(exist_ok=True)

        # Create a unique log file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"scopus_fetch_{timestamp}.log"

        # Configure logging
        self.logger = logging.getLogger(f"ScopusFetcher_{timestamp}")
        self.logger.setLevel(logging.INFO)

        # File handler for detailed logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler for important messages only (warnings and errors)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only show warnings and errors

        # Create formatters and add them to the handlers
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Log initialization
        self.logger.info(
            f"Initialized ScopusArticleFetcher with {len(self.api_keys)} API keys"
        )
        self.logger.info(f"Output directory: {self.output_dir}")

    def build_params(self, cursor: str, query_params: Optional[Dict] = None) -> Dict:
        """
        Builds the parameters for the API request.

        Args:
            cursor (str): The cursor parameter for pagination in API request.
            query_params (Dict, optional): Custom query parameters to override defaults.

        Returns:
            dict: The parameters for the API request.
        """
        default_params = {
            "cursor": cursor,
            "count": 25,
            "query": "TITLE-ABS('selective serotonin reuptake inhibitor') OR TITLE-ABS(ssri) OR TITLE-ABS(zimeldine) OR TITLE-ABS(fluoxetine) OR TITLE-ABS(citalopram) OR TITLE-ABS(paroxetine) OR TITLE-ABS(sertraline) OR TITLE-ABS(fluvoxamine) OR TITLE-ABS(escitalopram)",
            "apiKey": self.api_keys[self.current_api_key_index],
            "httpAccept": "application/json",
            "view": "COMPLETE",
            "date": "1982-2030",
            "sort": "+pubyear",
            "facets": "language(include=English)",
        }

        if query_params:
            default_params.update(query_params)
            self.logger.info(
                f"Using custom query parameters: {json.dumps(query_params, indent=2)}"
            )

        return default_params

    def request_page(self, params: Dict) -> Optional[Dict]:
        """
        Makes a request to the Scopus API with retry logic.

        Args:
            params (Dict): Request parameters.

        Returns:
            Optional[Dict]: API response or None if request fails.
        """
        session = requests.Session()
        retries = Retry(
            total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))

        try:
            self.logger.debug(
                f"Making request with API key {self.current_api_key_index + 1}"
            )
            response = session.get(self.base_url, params=params, headers=self.headers)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit exceeded
                self.logger.warning(
                    f"Rate limit exceeded for API key {self.current_api_key_index + 1}. "
                    f"Response: {response.text}"
                )
                if self.current_api_key_index < len(self.api_keys) - 1:
                    self.current_api_key_index += 1
                    self.logger.info(
                        f"Switching to API key {self.current_api_key_index + 1}"
                    )
                    return self.request_page(params)
                else:
                    self.logger.error("All API keys have reached their rate limit")
                    return None
            else:
                self.logger.error(
                    f"Failed to fetch data: HTTP Status Code {response.status_code}\n"
                    f"Response: {response.text}"
                )
                return None
        except Exception as e:
            self.logger.error(f"Error making request: {str(e)}", exc_info=True)
            return None

    def processing_results(self, results: Dict) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Processes API results into a DataFrame and extracts the next cursor.

        Args:
            results (Dict): API response results.

        Returns:
            Tuple[pd.DataFrame, Optional[str]]: Processed DataFrame and next cursor.
        """
        if not results or "search-results" not in results:
            self.logger.warning("No valid results found in API response")
            return pd.DataFrame(), None

        df = pd.DataFrame(results["search-results"].get("entry", []))
        cursor = results["search-results"]["cursor"].get("@next")

        if len(df) == 0:
            self.logger.warning("No entries found in search results")
        else:
            self.logger.debug(f"Processed {len(df)} entries")

        return df, cursor

    def fetch_results(
        self, saving_interval: int = 10, query_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Fetches all results from the Scopus API with pagination.

        Args:
            saving_interval (int): Number of iterations between saves.
            query_params (Dict, optional): Custom query parameters.

        Returns:
            pd.DataFrame: Complete results DataFrame.
        """
        self.logger.info("Starting to fetch results")
        params = self.build_params(cursor="*", query_params=query_params)
        results = self.request_page(params)

        if not results:
            self.logger.error("Failed to fetch initial results")
            return pd.DataFrame()

        total_results = int(results["search-results"]["opensearch:totalResults"])
        print(f"\nTotal results found: {total_results}")  # Use print for initial info

        iterations = (total_results + 24) // 25  # Ceiling division
        self.logger.info(f"Expected number of iterations: {iterations}")

        df, cursor = self.processing_results(results)
        full_df = df

        seen_cursors = set()
        iteration_counter = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize progress bar with total results
        with tqdm(
            total=total_results, desc="Fetching results", unit="articles"
        ) as pbar:
            # Update progress bar with initial results
            pbar.update(len(df))

            while cursor:
                iteration_counter += 1

                params = self.build_params(cursor=cursor, query_params=query_params)
                results = self.request_page(params)

                if not results:
                    self.logger.error(
                        f"Failed to fetch results at iteration {iteration_counter}"
                    )
                    break

                df, cursor = self.processing_results(results)

                if cursor in seen_cursors or not cursor:
                    self.logger.info(
                        "Reached end of results or encountered duplicate cursor"
                    )
                    break
                seen_cursors.add(cursor)

                full_df = pd.concat([full_df, df], ignore_index=True)
                # Update progress bar with number of new articles
                pbar.update(len(df))

                if iteration_counter % saving_interval == 0:
                    filename = os.path.join(
                        self.output_dir,
                        f"scopus_results_{timestamp}_iteration_{iteration_counter}.csv",
                    )
                    full_df.to_csv(filename, index=False)
                    self.logger.info(f"Saved intermediate results to {filename}")

        final_filename = os.path.join(
            self.output_dir, f"final_scopus_results_{timestamp}.csv"
        )
        full_df.to_csv(final_filename, index=False)
        print(f"\nSaved final results to {final_filename}")  # Use print for final info
        print(f"Total articles fetched: {len(full_df)}")  # Use print for final info

        return full_df


if __name__ == "__main__":
    # Example usage
    API_KEYS = ["your_api_key_1", "your_api_key_2", "your_api_key_3"]

    # Custom query parameters (optional)
    custom_params = {
        "date": "2020-2024",
        "sort": "-citedby-count",
        "query": "TITLE-ABS('machine learning') AND PUBYEAR > 2020",
    }

    # Initialize fetcher with multiple API keys
    fetcher = ScopusArticleFetcher(
        api_keys=API_KEYS, output_dir="../data/01-raw/scopus"
    )

    # Fetch results with custom parameters
    results_df = fetcher.fetch_results(saving_interval=10, query_params=custom_params)

    print(f"Total articles fetched: {len(results_df)}")
    print(f"Total articles fetched: {len(results_df)}")
    print(f"Total articles fetched: {len(results_df)}")
