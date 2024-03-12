import json

import requests
from tqdm import tqdm


class RefsFetcher:
    """Fetch references using OpenCitations and Crossref for a given list of DOIs."""

    def __init__(self, dataframe):
        """
        Initialize RefsFetcher with a DataFrame containing DOIs.

        :param dataframe: A DataFrame containing DOIs in a column named "doi."
        """
        self.df = dataframe

    @staticmethod
    def handle_error(message):
        """Return a standardized error structure."""
        error_dict = [{"ERROR": message}]
        error_list = ["ERROR", message]
        return error_dict, error_list

    def fetch_references(self, url, extraction_func):
        """Fetch references from a given URL and apply an extraction function to get the details."""
        try:
            response = requests.get(url)
            if response.status_code == 200 and response.text != "":
                return extraction_func(response)
            else:
                return self.handle_error("NOT FOUND")
        except Exception as e:
            return self.handle_error(str(e))

    def open_citations_func(self, doi):
        """Use OpenCitations API to retrieve citation data for a given DOI."""
        url = f"https://w3id.org/oc/index/api/v1/references/{doi}"

        def extract_oc_data(response):
            oc_refs = json.loads(response.text)
            oc_dois = [ref["cited"] for ref in oc_refs]

            return oc_refs, oc_dois

        return self.fetch_references(url, extract_oc_data)

    def cross_ref_func(self, doi):
        """Use Crossref API to retrieve citation data for a given DOI."""
        url = f"https://api.crossref.org/v1/works/{doi}"

        def extract_cr_data(response):
            metadata = response.json()
            cr_refs = metadata.get("message", {}).get("reference", [])
            cr_dois = [ref.get("DOI", None) for ref in cr_refs]
            return cr_refs, cr_dois

        return self.fetch_references(url, extract_cr_data)

    def reference_fetcher(self):
        """Fetch references for a list of DOIs, process and merge them, and update the dataframe with results."""
        # Initialize the lists to store results
        oc_refs_list, oc_dois_list, cr_refs_list, cr_dois_list, merged_dois = (
            [],
            [],
            [],
            [],
            [],
        )

        print("Starting reference fetching...")
        for doi in tqdm(self.df["doi"]):
            try:
                if doi:
                    # Get references from both OpenCitations and Crossref
                    oc_ref, oc_doi = self.open_citations_func(doi)
                    cr_ref, cr_doi = self.cross_ref_func(doi)

                    # Append the results
                    oc_refs_list.append(oc_ref)
                    oc_dois_list.append(oc_doi)
                    cr_refs_list.append(cr_ref)
                    cr_dois_list.append(cr_doi)

                    # Merge the DOIs, removing duplicates and ignoring None values
                    merged_dois.append(
                        [x for x in set(oc_doi + cr_doi) if x is not None]
                    )
                else:
                    # Handle cases with no DOI
                    error_dict, error_list = self.handle_error("NO DOI")
                    oc_refs_list.append(error_dict)
                    oc_dois_list.append(error_list)
                    cr_refs_list.append(error_dict)
                    cr_dois_list.append(error_list)
                    merged_dois.append(error_list)
            except Exception as e:
                # Append error messages if any exception occurs
                error_dict, error_list = self.handle_error(str(e))
                oc_refs_list.append(error_dict)
                oc_dois_list.append(error_list)
                cr_refs_list.append(error_dict)
                cr_dois_list.append(error_list)
                merged_dois.append(error_list)

        # Update the dataframe with the results
        self.df.loc[:, "oc_refs"], self.df.loc[:, "oc_dois"] = (
            oc_refs_list,
            oc_dois_list,
        )
        self.df.loc[:, "cr_refs"], self.df.loc[:, "cr_dois"] = (
            cr_refs_list,
            cr_dois_list,
        )
        self.df.loc[:, "merged_dois"] = merged_dois

        # remove empty lists
        self.df["oc_dois"] = self.df["oc_dois"].apply(
            lambda x: ["ERROR", "Unknown"] if x == [] else x
        )
        self.df["cr_dois"] = self.df["cr_dois"].apply(
            lambda x: ["ERROR", "Unknown"] if x == [] else x
        )
        self.df["oc_refs"] = self.df["oc_refs"].apply(
            lambda x: [{"ERROR": "Unknown"}] if x == [] else x
        )
        self.df["cr_refs"] = self.df["cr_refs"].apply(
            lambda x: [{"ERROR": "Unknown"}] if x == [] else x
        )

        self.df["merged_dois"] = self.df["merged_dois"].apply(
            lambda x: ["ERROR", "Unknown"] if x == [] else x
        )

        print("All references fetched.")
        self.calculate_and_print_avg()

    def calculate_and_print_avg(self):
        """Calculate and print the average number of DOIs found."""
        self.df["oc_nr_dois"] = self.df["oc_dois"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        self.df["cr_nr_dois"] = self.df["cr_dois"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        self.df["merged_nr_dois"] = self.df["merged_dois"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        print(
            "Average number of DOIs found per paper and per method (cr and oc, and the joint of both sets):"
        )
        print(
            self.df.aggregate(
                {"oc_nr_dois": "mean", "cr_nr_dois": "mean", "merged_nr_dois": "mean"}
            )
        )

    def parquet_saver(self, path: str, filename: str) -> None:
        """
        Save the dataframe as a parquet file.
        """
        # Serialize the complex columns into JSON strings
        # complex_columns = ['oc_refs', 'oc_dois', 'cr_refs', 'cr_dois']
        # for col in complex_columns:
        #    self.df[col] = self.df[col].apply(json.dumps)
        self.df.reset_index(inplace=True, drop=True)
        if filename[-8:] == ".parquet":
            filename = filename[:-8]
        self.df.to_parquet(path + filename + ".parquet")
        print("Dataframe saved here:\n", path + filename)


if __name__ == "__main__":
    path = "/Users/jlq293/Projects/Study1 local/ClusterAndMap/Data/"
    filename = "TESTAntidepressants1986-2022_Processed.parquet"
    df = pd.read_parquet(path + filename).sample(frac=0.001, random_state=42)
    ref_fetcher = RefsFetcher(df)
    ref_fetcher.reference_fetcher()
    ref_fetcher.calculate_and_print_avg()
    ref_fetcher.parquet_saver(path, filename)
