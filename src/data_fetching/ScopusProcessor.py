import requests
import datetime
import json
import logging
import os
import sys
import time
from math import ceil
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.data_fetching.ScopusReferenceFetcher import ScopusReferenceFetcher
from src.data_fetching.ScopusApiKeyLoader import ScopusApiKeyLoader


class ScopusRefFetcherPrep:
    """
    Class to prepare the data for fetching references from Scopus.
    """

    @staticmethod
    def load_fetched_reference_data(data_path: str) -> tuple[List[str], int]:
        """
        This is only necessary after the initial fetch of the reference data.
        It loads the data and returns a list of EIDs and the highest batch number.

        Args:
            data_path (str): Path to the directory containing reference data files

        Returns:
            tuple[List[str], int]: List of processed EIDs and the highest batch number
        """
        files = os.listdir(data_path)
        files = [file for file in files if file.endswith(".json")]
        # Extract batch numbers and find the maximum
        max_batch = max(int("".join(filter(str.isdigit, file))) for file in files)
        logging.info(f"Found {len(files)} files with batch numbers up to {max_batch}.")
        eids = []
        for file in files:
            with open(data_path + file, "r") as fp:
                data = json.load(fp)
                eids.extend(list(data.keys()))
        # remove data from memory
        del data
        return eids, max_batch

    @staticmethod
    def load_and_filter_articles(path: str, eids: List[str]) -> pd.DataFrame:
        """
        Load articles from CSV and filter out already processed ones.

        Args:
            path (str): Path to the CSV file containing articles
            eids (List[str]): List of already processed EIDs

        Returns:
            pd.DataFrame: Filtered DataFrame of articles to process
        """
        # read pkl file
        df = pd.read_pickle(path)
        df_filtered = df[~df["eid"].isin(eids)]
        df_filtered = df_filtered.reset_index(drop=True)
        logging.info(f"Number of articles to fetch: {len(df_filtered)}")
        return df_filtered


class ScopusRefFetcherProcessor:
    """
    Class to fetch references from Scopus.
    """

    @staticmethod
    def setup_logging(
        log_directory: str, log_level: int = logging.INFO, console_output: bool = False
    ) -> None:
        """
        Set up logging to write to a file with a timestamp in its name.
        Creates a new log for each run and configures a fresh logger.

        Args:
            log_directory (str): Directory where the log file will be saved.
            log_level (int): Logging level to capture. Default is logging.INFO.
            console_output (bool): Whether to output logs to console. Default is False.
        """
        # Create a directory for logs if it doesn't exist
        os.makedirs(log_directory, exist_ok=True)

        # Reset root logger to avoid duplicate handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure logging to write to a file with a timestamp in its name
        # Include milliseconds in timestamp to ensure uniqueness
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        log_filename = f"{log_directory}/scopus_fetcher_{current_time}.log"

        # Create file handler
        file_handler = logging.FileHandler(log_filename, mode="w")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Configure root logger
        logging.root.setLevel(log_level)
        logging.root.addHandler(file_handler)

        # Add console handler only if requested (disabled by default)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logging.root.addHandler(console_handler)

        # Add a first log message to verify the file is being written to
        logging.info(f"Logging started at {current_time}")
        logging.info(f"Log file: {os.path.abspath(log_filename)}")

        # Flush handlers to ensure log is written immediately
        for handler in logging.root.handlers:
            handler.flush()

    @staticmethod
    def process_scopus_batches(
        df_to_fetch: pd.DataFrame,
        data_path: str,
        last_processed_batch: int,
        batch_size: int = 500,
    ) -> None:
        """
        Process batches of data and fetch references using ScopusReferenceFetcher.
        Automatically rotates through API keys when rate limits are hit.

        Args:
            df_to_fetch (pd.DataFrame): DataFrame containing data to be processed.
            data_path (str): Path where the processed data will be saved.
            last_processed_batch (int): The last processed batch number for naming files.
            batch_size (int, optional): Number of records to process in each batch. Default is 500.
        """
        current_api_key_idx = 0
        api_keys = ScopusApiKeyLoader.get_api_keys()
        api_key_names = list(api_keys.keys())

        if not api_key_names:
            logging.error("No API keys available. Stopping.")
            return

        # Calculate total number of batches
        num_batches_to_fetch = ceil(len(df_to_fetch) / batch_size)
        total_nr_of_batches = last_processed_batch + num_batches_to_fetch

        logging.info(f"Fetching total of {num_batches_to_fetch} new batches")
        logging.info(f"Fetching {len(df_to_fetch)} articles")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Last processed batch: {last_processed_batch}")
        logging.info(f"Total number of batches when done: {total_nr_of_batches}")

        # Use tqdm for progress tracking
        for current_batch in tqdm(range(last_processed_batch, total_nr_of_batches)):
            # Calculate the correct slice indices for the current batch
            start_idx = (current_batch - last_processed_batch) * batch_size
            end_idx = min(start_idx + batch_size, len(df_to_fetch))

            if start_idx >= len(df_to_fetch):
                logging.info(
                    f"All articles processed. Stopping at batch {current_batch}."
                )
                break

            logging.info(
                f"Processing batch {current_batch + 1} ({end_idx - start_idx} articles)"
            )

            # Initialize empty data dictionary for this batch
            data_dict: Dict[str, List[Dict]] = {}
            batch_completed = False

            # Try with different API keys until batch is complete or all keys are exhausted
            while not batch_completed:
                current_api_key_info = api_keys[api_key_names[current_api_key_idx]]
                logging.info(
                    f"Using API key: {current_api_key_info['key_name']} ({current_api_key_info['description']})"
                )

                fetcher = ScopusReferenceFetcher(current_api_key_info)
                rate_limit_hit = False

                try:
                    # Process each article in the current batch slice
                    for _, row in df_to_fetch.iloc[start_idx:end_idx].iterrows():
                        eid = row["eid"]
                        if (
                            eid in data_dict
                        ):  # Skip if already processed with a previous API key
                            continue

                        # Replace the error handling section in process_scopus_batches with this improved version
                        try:
                            logging.info(f"Fetching references for EID: {eid}")
                            references = fetcher.request_eid(eid)
                            data_dict[eid] = references
                            logging.info(
                                f"Successfully fetched {len(references)} references for EID: {eid}"
                            )
                        except requests.exceptions.HTTPError as e:
                            # Check for either '429' status code OR 'Rate limit exceeded' in the error message
                            if "429" in str(e) or "Rate limit exceeded" in str(e):
                                logging.warning(
                                    f"Rate limit hit for API key: {current_api_key_info['key_name']}"
                                )
                                rate_limit_hit = True
                                break
                            else:
                                logging.error(f"HTTP error for EID {eid}: {e}")
                        except Exception as e:
                            logging.error(f"Error processing EID {eid}: {str(e)}")

                    # If we didn't hit a rate limit, the batch is complete
                    if not rate_limit_hit:
                        batch_completed = True
                    else:
                        # Switch to next API key
                        current_api_key_idx = (current_api_key_idx + 1) % len(
                            api_key_names
                        )
                        if current_api_key_idx == 0:  # We've tried all keys
                            logging.warning(
                                "All API keys have hit their rate limits. Saving partial batch and stopping."
                            )
                            batch_completed = (
                                True  # Force completion to save partial results
                            )

                except Exception as e:
                    logging.error(
                        f"Unexpected error in batch {current_batch + 1}: {str(e)}"
                    )
                    current_api_key_idx = (current_api_key_idx + 1) % len(api_key_names)
                    logging.info(
                        f"Switching to API key: {api_key_names[current_api_key_idx]}"
                    )
                    time.sleep(2)  # Small delay to ensure clean switch between keys

                    if current_api_key_idx == 0:  # We've tried all keys
                        logging.error(
                            "All API keys have encountered errors. Saving partial batch and stopping."
                        )
                        batch_completed = (
                            True  # Force completion to save partial results
                        )

            # Save batch data (complete or partial)
            if data_dict:  # Only save if we have data
                batch_file = os.path.join(data_path, f"batch_{current_batch + 1}.json")
                with open(batch_file, "w") as fp:
                    json.dump(data_dict, fp)
                logging.info(
                    f"Saved batch {current_batch + 1} with {len(data_dict)} articles."
                )
            else:
                logging.warning(
                    f"Batch {current_batch + 1} had no successful fetches. Skipping file creation."
                )

            # If we've cycled through all API keys and still hit rate limits, stop processing
            if rate_limit_hit and current_api_key_idx == 0:
                logging.error("All API keys exhausted. Stopping batch processing.")
                break
