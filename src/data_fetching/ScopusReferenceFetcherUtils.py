import json
import logging
import os
import sys
import datetime
from math import ceil
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from src.data_fetching.ScopusReferenceFetcher import ScopusReferenceFetcher


class ScopusRefFetcherProcessor:
    """
    Class to fetch references from Scopus.
    """

    @staticmethod
    def setup_logging(log_directory: str, log_level=logging.INFO) -> None:
        """
        Set up logging to write to a file with a timestamp in its name.

        Args:
            log_directory (str): Directory where the log file will be saved.
            log_level (logging.Level): Logging level to capture. Default is logging.INFO.
        """
        # Create a directory for logs if it doesn't exist
        os.makedirs(log_directory, exist_ok=True)

        # Configure logging to write to a file with a timestamp in its name
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{log_directory}/scopus_fetcher_{current_time}.log"
        logging.basicConfig(
            filename=log_filename,
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    @staticmethod
    def process_scopus_batches(
        api_keys: Dict[str, Dict[str, Any]],
        df_to_fetch: pd.DataFrame,
        data_path: str,
        last_processed_batch: int,
        batch_size: int = 500,
    ) -> None:
        """
        Process batches of data and fetch references using ScopusReferenceFetcher.
        Automatically rotates through API keys when rate limits are hit.

        Args:
            api_keys (Dict[str, Dict[str, Any]]): Dictionary of API key information for ScopusReferenceFetcher.
            df_to_fetch (DataFrame): DataFrame containing data to be processed.
            data_path (str): Path where the processed data will be saved.
            last_processed_batch (int): The last processed batch number for naming files.
            batch_size (int, optional): Number of records to process in each batch. Default is 500.
        """
        current_api_key_idx = 0
        api_key_names = list(api_keys.keys())
        current_batch = last_processed_batch
        num_batches = ceil(len(df_to_fetch) / batch_size)

        while current_batch < num_batches:
            current_api_key_info = api_keys[api_key_names[current_api_key_idx]]
            logging.info(
                f"Using API key: {api_key_names[current_api_key_idx]} ({current_api_key_info['description']})"
            )

            fetcher = ScopusReferenceFetcher(current_api_key_info)
            data_dict = {}

            try:
                for _, row in df_to_fetch.iloc[
                    current_batch * batch_size : (current_batch + 1) * batch_size
                ].iterrows():
                    try:
                        data_dict[row["eid"]] = fetcher.request_eid(row["eid"])
                    except Exception as e:
                        logging.error(f"Error processing EID: {row['eid']}. Error: {e}")
                        if "429" in str(e):
                            logging.warning(
                                f"Rate limit hit for API key: {api_key_names[current_api_key_idx]}"
                            )
                            # Save current progress
                            batch_file = os.path.join(
                                data_path, f"batch_{current_batch + 1}.json"
                            )
                            with open(batch_file, "w") as fp:
                                json.dump(data_dict, fp)

                            # Rotate to next API key
                            current_api_key_idx = (current_api_key_idx + 1) % len(
                                api_keys
                            )
                            if current_api_key_idx == 0:
                                logging.error(
                                    "All API keys have hit their rate limits. Stopping."
                                )
                                return
                            continue
                        else:
                            logging.error(f"Unexpected error for EID {row['eid']}: {e}")
                            continue

                # Save successful batch
                batch_file = os.path.join(data_path, f"batch_{current_batch + 1}.json")
                with open(batch_file, "w") as fp:
                    json.dump(data_dict, fp)
                logging.info(f"Data for batch {current_batch + 1} saved to file.")
                current_batch += 1

            except Exception as e:
                logging.error(f"Error processing batch {current_batch + 1}: {e}")
                if "429" in str(e):
                    logging.warning(
                        f"Rate limit hit for API key: {api_key_names[current_api_key_idx]}"
                    )
                    current_api_key_idx = (current_api_key_idx + 1) % len(api_keys)
                    if current_api_key_idx == 0:
                        logging.error(
                            "All API keys have hit their rate limits. Stopping."
                        )
                        return
                    continue
                else:
                    raise e
