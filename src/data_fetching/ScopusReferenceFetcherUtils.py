import json
import logging
import os
import sys
import datetime
from math import ceil
from dotenv import load_dotenv

import pandas as pd
from tqdm import tqdm

from src.data_fetching.ScopusReferenceFetcher import ScopusReferenceFetcher


class ScopusRefFetcherPrep:
    """
    Class to prepare the data for fetching references from Scopus.
    """

    @staticmethod
    def get_api_keys():
        """
        Get Scopus API keys from environment variables.

        Returns:
            dict: Dictionary of API keys with their names as keys
        """
        load_dotenv()

        api_keys = {
            "api_key_A": os.getenv("SCOPUS_API_KEY_A"),
            "api_key_B": os.getenv("SCOPUS_API_KEY_B"),
            "api_key_deb": os.getenv("SCOPUS_API_KEY_DEB"),
            "api_key_haoxin": os.getenv("SCOPUS_API_KEY_HAOXIN"),
        }

        # Remove any None values
        api_keys = {k: v for k, v in api_keys.items() if v is not None}

        if not api_keys:
            raise ValueError("No API keys found in environment variables")

        print("API Rate Limits:")
        print("- api_key_A: 40,000 requests per week")
        print("- Other keys: 10,000 requests per week")

        return api_keys

    @staticmethod
    def load_fetched_reference_data(data_path):
        """
        This is only necessary after the inital fetch of the reference data.
        It loads the data and returns a list of EIDs and the highest batch number.
        """
        files = os.listdir(data_path)
        files = [file for file in files if file.endswith(".json")]

        if not files:
            print("No previously processed files found.")
            return [], 0

        # Extract batch numbers and find the maximum
        max_batch = max(int(file.split("_")[3].split(".")[0]) for file in files)
        print(f"Found {len(files)} files with batch numbers up to {max_batch}.")

        eids = []
        for file in files:
            with open(data_path + file, "r") as fp:
                data = json.load(fp)
                eids.extend(list(data.keys()))
        # remove data from memory
        del data
        return eids, max_batch

    @staticmethod
    def load_and_filter_articles(path, eids):
        """
        Load articles from either CSV or pickle file and filter out already processed ones.

        Args:
            path (str): Path to the file (CSV or pickle)
            eids (list): List of already processed EIDs

        Returns:
            DataFrame: Filtered dataframe with unprocessed articles
        """
        # Try different encodings for CSV files
        encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

        if path.endswith(".csv"):
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    raise Exception(f"Error reading CSV file: {e}")
            else:
                raise UnicodeDecodeError(
                    f"Could not read file with any of the encodings: {encodings}"
                )
        elif path.endswith(".pkl"):
            try:
                df = pd.read_pickle(path)
            except Exception as e:
                raise Exception(f"Error reading pickle file: {e}")
        else:
            raise ValueError("File must be either CSV or pickle format")

        df_filtered = df[~df["eid"].isin(eids)]
        df_filtered = df_filtered.reset_index(drop=True)
        print(f"Number of articles to fetch: {len(df_filtered)}")
        return df_filtered


class ScopusRefFetcherProcessor:
    """
    Class to fetch references from Scopus.
    """

    @staticmethod
    def setup_logging(log_directory, log_level=logging.INFO):
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
        api_keys, df_to_fetch, data_path, last_processed_batch, batch_size=500
    ):
        """
        Process batches of data and fetch references using ScopusReferenceFetcher.
        Automatically rotates through API keys when rate limits are hit.

        Args:
        api_keys (dict): Dictionary of API keys for ScopusReferenceFetcher.
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
            current_api_key = api_keys[api_key_names[current_api_key_idx]]
            logging.info(f"Using API key: {api_key_names[current_api_key_idx]}")

            fetcher = ScopusReferenceFetcher(current_api_key)
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
                            with open(
                                f"{data_path}scopus_references_batch_{current_batch + 1}.json",
                                "w",
                            ) as fp:
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
                with open(
                    f"{data_path}scopus_references_batch_{current_batch + 1}.json",
                    "w",
                ) as fp:
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
