import pandas as pd
from tqdm import tqdm


class PaperConnector:
    def __init__(self, df=None, path=None):
        """
        Initializes the PaperConnector object with a DataFrame containing paper details.

        Parameters:
        - df (pandas.DataFrame, optional): DataFrame containing paper details. Must include
          columns like 'pid', 'year', 'doi', and 'merged_dois'.
        - path (str, optional): Path to a parquet file containing paper details. This is an alternative
          to providing a DataFrame directly.

        Either 'df' or 'path' must be specified. If both are provided, 'df' takes precedence.

        Raises:
        - ValueError: If neither 'df' nor 'path' is provided.
        """
        if df is None and path:
            self.df = pd.read_parquet(path)
        elif path is None and df is not None:
            self.df = df
        else:
            raise ValueError("Either df or path must be given.")

    def _process_papers(self):
        """
        Prepares the DataFrame for efficient matching by sorting it by year.

        This method sorts the DataFrame in descending order of the 'year' column, ensuring
        that newer papers are processed first in the matching algorithm. It's a preliminary
        step that optimizes the subsequent matching process.
        """
        self.df = self.df.sort_values(by="year", ascending=False).reset_index(drop=True)
        print("Papers sorted by year.")

    def _create_doi_lookup(self):
        """
        Creates a lookup dictionary mapping DOIs to paper IDs.

        This method iterates over each row in the DataFrame and constructs a dictionary where
        each DOI is a key, and the value is a list of paper IDs ('pid') that contain this DOI.
        This lookup facilitates efficient searching and matching of papers based on DOIs.

        Returns:
        - dict: A dictionary mapping DOIs to lists of paper IDs.
        """
        doi_lookup = {}
        for _, row in self.df.iterrows():
            for doi in row["merged_dois"]:
                doi_lookup.setdefault(doi, []).append(row["paper_id"])
        return doi_lookup

    def connect_papers(self):
        """
        Connects papers in the database based on matching DOIs.

        This method utilizes the DOI lookup dictionary created by `_create_doi_lookup` to find
        and record matches for each paper based on its DOI. The matches are stored in a new
        column 'paper_ids_matched_on_dois' in the DataFrame. This function significantly optimizes
        the matching process by eliminating redundant searches and using efficient data structures.
        """
        self._process_papers()
        print("Starting paper connection...")

        doi_lookup = self._create_doi_lookup()

        paper_ids_matched_on_dois = {}
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            doi = row["doi"]
            og_pid = row["paper_id"]

            if not doi:
                paper_ids_matched_on_dois[og_pid] = ["no doi given"]
            else:
                # Ensure matched_pids are stored as strings
                matched_pids = [
                    str(pid) for pid in doi_lookup.get(doi, ["no doi match"])
                ]
                paper_ids_matched_on_dois[og_pid] = matched_pids

        # Map the dictionary to the DataFrame
        self.df["paper_ids_matched_on_dois"] = self.df["paper_id"].apply(
            lambda x: paper_ids_matched_on_dois.get(x, ["no doi match"])
        )

        print("Paper connection finished.")
        print(
            "Average number of matches on DOIs:",
            self.df["paper_ids_matched_on_dois"].apply(len).mean(),
        )


if __name__ == "__main__":
    # Example usage of the PaperConnector class
    # Replace 'your_dataframe' and 'your_path' with actual DataFrame or path
    try:
        connector = PaperConnector(df=your_dataframe)  # Replace 'your_dataframe'
    except NameError:
        connector = PaperConnector(path="your_path")  # Replace 'your_path'

    # Connect papers based on DOIs
    connector.connect_papers()
