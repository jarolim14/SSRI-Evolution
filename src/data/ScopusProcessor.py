import datetime
import json
import logging
import os
import sys
from math import ceil

import pandas as pd
from tqdm import tqdm

from src.data.ScopusReferenceFetcher import ScopusReferenceFetcher


class ScopusRefFetcherPrep:
    """
    Class to prepare the data for fetching references from Scopus.
    """

    @staticmethod
    def get_api_keys(path="../notebooks/api_key_scopus.json"):
        api_keys = json.load(open(path))
        return api_keys

    @staticmethod
    def load_fetched_reference_data(data_path):
        """
        This is only necessary after the inital fetch of the reference data.
        It loads the data and returns a list of EIDs and the highest batch number.
        """
        files = os.listdir(data_path)
        files = [file for file in files if file.endswith(".json")]
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
        df = pd.read_csv(path)
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
        api_key, df_to_fetch, data_path, last_processed_batch, batch_size=500
    ):
        """
        Process batches of data and fetch references using ScopusReferenceFetcher.

        Args:
        api_key (str): API key for ScopusReferenceFetcher.
        df_to_fetch (DataFrame): DataFrame containing data to be processed.
        data_path (str): Path where the processed data will be saved.
        last_processed_batch (int): The last processed batch number for naming files.
        batch_size (int, optional): Number of records to process in each batch. Default is 500.
        """

        fetcher = ScopusReferenceFetcher(api_key)
        num_batches = ceil(len(df_to_fetch) / batch_size)

        for batch in tqdm(range(num_batches)):
            data_dict = {}

            for _, row in df_to_fetch.iloc[
                batch * batch_size : (batch + 1) * batch_size
            ].iterrows():
                try:
                    data_dict[row["eid"]] = fetcher.request_eid(row["eid"])
                except Exception as e:
                    logging.error(f"Error processing EID: {row['eid']}. Error: {e}")
                    if "429" in str(e):
                        logging.error("Too many requests, breaking the loop.")
                        with open(
                            f"{data_path}scopus_references_batch_{last_processed_batch + batch + 1}.json",
                            "w",
                        ) as fp:
                            json.dump(data_dict, fp)
                        logging.info(
                            f"Last possible batch {last_processed_batch + batch + 1} saved uncompleted."
                        )
                        sys.exit("Stopping script due to 429 Too Many Requests error.")

            with open(
                f"{data_path}scopus_references_batch_{last_processed_batch + batch + 1}.json",
                "w",
            ) as fp:
                json.dump(data_dict, fp)
            logging.info(
                f"Data for batch {last_processed_batch + batch + 1} saved to file."
            )
