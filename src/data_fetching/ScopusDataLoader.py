import json
import os

import pandas as pd


class ScopusDataLoader:
    """
    Class to load and filter Scopus data.
    """
    @staticmethod
    def load_fetched_reference_data(data_path):
        """
        This is only necessary after the initial fetch of the reference data.
        It loads all batch files with a progress bar, combines their data into a single JSON file,
        and returns a list of EIDs, the highest batch number, and the combined data.
        """
        from tqdm import tqdm

        files = os.listdir(data_path)
        files = [file for file in files if file.endswith(".json")]

        if not files:
            print("No previously processed files found.")
            return [], 0, {}

        # Extract batch numbers and find the maximum
        batch_files = [file for file in files if file.startswith("batch_")]
        if not batch_files:
            print("No batch files found.")
            return [], 0, {}

        try:
            max_batch = max(
                int(file.split("_")[1].split(".")[0]) for file in batch_files
            )
            print(
                f"Found {len(batch_files)} batch files with numbers up to {max_batch}."
            )
        except (ValueError, IndexError) as e:
            print(f"Error parsing batch numbers: {e}")
            return [], 0, {}

        # Combine all JSON data into a single dictionary
        combined_data = {}
        eids = []

        # Use tqdm to show progress while processing files
        for file in tqdm(batch_files, desc="Processing batch files"):
            with open(os.path.join(data_path, file), "r") as fp:
                data = json.load(fp)
                combined_data.update(data)
                eids.extend(list(data.keys()))
                # remove data from memory after adding to combined_data
                del data

        # Create a combined file with all the data
        combined_file_path = os.path.join(data_path, "combined_reference_data.json")
        with open(combined_file_path, "w") as fp:
            json.dump(combined_data, fp)

        print(f"Created combined reference data file at: {combined_file_path}")

        # Return the list of EIDs, max batch number, and the combined data dictionary
        return eids, max_batch, combined_data

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
