import glob
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd


class ReadAndCleanScopusCSV:
    def __init__(self, df: pd.DataFrame = None):
        """
        Initialize the DataCleaner instance.
        """
        self.df = df
        print("DataCleaner instance initialized")

    def read_scopus_data(
        self, directory: str, sample_size: Optional[int] = None
    ) -> None:
        """
        Read all CSV data from a specified directory, merge them, and perform initial processing.
        """
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(directory, "*.csv"))

        # Check if there are any CSV files in the directory
        if not csv_files:
            print("No CSV files found in the directory.")
            return

        # Read and append each CSV file into a list
        dataframes = []
        for file in csv_files:
            print(f"Reading data from {file}")
            dtype_dict = {
                8: str,
                20: str,
                28: str,
                30: str,
                31: str,
                32: str,
                33: str,
                34: str,
                35: str,
            }
            df = pd.read_csv(file, dtype=dtype_dict)
            dataframes.append(df)

        # Merge all dataframes
        self.df = pd.concat(dataframes, ignore_index=True)

        # Processing steps
        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])
        self.df = self.df.sort_values(by="Year").reset_index(drop=True)

        if sample_size is not None:
            self.df = (
                self.df.sample(sample_size, random_state=1887)
                .sort_values(by="Year")
                .reset_index(drop=True)
            )

        self.df.reset_index(inplace=True, drop=True)
        print(
            "Data from all files read and merged. Nr of Articles: {}".format(
                len(self.df)
            )
        )

    def clean_column_names(self) -> None:
        """
        Clean column names by replacing spaces and special characters.
        """
        self.df.columns = [
            re.sub(
                "[^0-9a-zA-Z_]+", "", col.replace(" ", "_").replace(".", "_").lower()
            )
            for col in self.df.columns
        ]
        self.df.columns = [
            ("_" + col) if col[0].isdigit() else col for col in self.df.columns
        ]
        self.df.rename(columns={"years": "year"}, inplace=True)
        self.df.rename(columns={"titles": "title"}, inplace=True)
        print("Column names cleaned")
        print("-" * 50)

    def english_only(self) -> None:
        """
        Make sure all articles are in English.
        """
        non_english_count = len(
            self.df[self.df["language_of_original_document"] != "English"]
        )
        print("Number of articles not in English removed: {}".format(non_english_count))
        self.df = self.df[self.df["language_of_original_document"] == "English"]

    def remove_columns(
        self, default: bool = False, columns_to_remove: Optional[List[str]] = None
    ) -> None:
        """
        Remove specified columns from the dataframe.
        """
        default_columns_to_remove = [
            "author_full_names",
            "authors_id",
            "source_title",
            "volume",
            "issue",
            "art__no_",
            "page_start",
            "page_end",
            "page_count",
            "affiliations",
            "authors_with_affiliations",
            "molecular_sequence_numbers",
            "chemicalscas",
            "tradenames",
            "manufacturers",
            "funding_details",
            "funding_texts",
            "correspondence_address",
            "editors",
            "publisher",
            "sponsors",
            "conference_name",
            "conference_date",
            "conference_location",
            "conference_code",
            "issn",
            "isbn",
            "coden",
            "abbreviated_source_title",
            "publication_stage",
            "open_access",
            "language_of_original_document",
        ]
        if default:
            columns_to_remove = default_columns_to_remove
        self.df = self.df.drop(columns=columns_to_remove)

    def remove_duplicates(self, column: str = "eid") -> None:
        """
        Remove duplicates based on a given column.
        """
        len_before = len(self.df)
        self.df.drop_duplicates(subset=column, inplace=True)
        len_after = len(self.df)
        print(f"Removed {len_before - len_after} duplicates based on {column}")

    def remove_document_types(
        self,
        document_types: List[str] = [
            "Book chapter",
            "Note",
            "Short survey",
            "Erratum",
            "Retracted",
            "Book",
            "Data paper",
            "Conference review",
            "Article in press",
        ],
    ) -> None:
        """
        Removes rows from the DataFrame based on the specified document types.

        Parameters:
        document_types (List[str]): A list of document types to be removed from the DataFrame.
        """
        len_before = len(self.df)
        self.df = self.df[~self.df["document_type"].isin(document_types)]
        len_after = len(self.df)
        print(f"Removed {len_before - len_after} articles based on document type")
        print(f"Types removed: {document_types}")

    def clean_title_abstract(self) -> None:
        """
        Change types of missing values in title and abstract columns.
        """
        self.df["title"] = self.df["title"].apply(
            lambda x: np.nan if x in ["", " ", "No Title", None] else x
        )
        self.df["abstract"] = self.df["abstract"].apply(
            lambda x: np.nan if x in ["", "[No abstract available]", None] else x
        )

    def clean_author_names(self) -> None:
        """
        Create list of authors where one item is a string of lastname then comma, then initials
        """

        def extract_names(name_string):
            if not isinstance(name_string, str):
                return np.nan

            # Split the string by semicolon to separate individual names
            names = name_string.split(";")

            # Regex pattern to match last names and initials with periods
            # The pattern is designed to capture initials which may include multiple uppercase letters each followed by a period
            pattern = r"([\w\-\s]*[\w\-])\s(([A-Z]\.)+)"

            extracted_names = []
            for name in names:
                # Trim whitespace and search with regex
                match = re.search(pattern, name.strip())
                if match:
                    # Extract and format name
                    last_name = match.group(1).strip()
                    initials = match.group(2).strip()  # Initials including periods
                    extracted_names.append(f"{last_name}; {initials}")

            return extracted_names

        self.df["authors"] = self.df["authors"].apply(extract_names)

    def parquet_saver(self, path: str, filename: str) -> None:
        """
        Save the dataframe as a parquet file.
        """
        if filename.endswith(".parquet"):
            filename = filename[:-8]
        self.df.reset_index(inplace=True, drop=True)
        self.df.to_parquet(path + filename + ".parquet")
        print("Dataframe saved here:\n", path + filename)


# Example usage
if __name__ == "__main__":
    # Initialize the DataCleaner instance
    cleaner = ReadAndCleanScopusCSV()

    # Read and merge all CSV files in the specified directory
    cleaner.read_scopus_data(directory="data/raw/scopus_data/")

    # Clean column names
    cleaner.clean_column_names()

    # Remove non-English articles
    cleaner.english_only()

    # Remove columns
    cleaner.remove_columns(default=True)

    # Remove duplicates
    cleaner.remove_duplicates()

    # Remove document types
    cleaner.remove_document_types()

    # change types of missing values
    clenaner.clean_title_abstract()

    # Save the dataframe as a parquet file
    cleaner.parquet_saver(path="data/processed/", filename="scopus_data.parquet")
    cleaner.parquet_saver(path="data/processed/", filename="scopus_data.parquet")
    cleaner.parquet_saver(path="data/processed/", filename="scopus_data.parquet")
