import requests


class PMIDsFromQuery:
    """
    A class to fetch article data from PubMed using E-utilities.

    Attributes:
        email (str): Email address to be used for queries to the PubMed E-utilities API.
    """

    def __init__(self, email, query=None):
        """
        Initialize the LitDataFromMedline with the provided email address.

        Args:
            email (str): An email address to be used for queries to the PubMed E-utilities API.
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = email

    def fetch_pubmed_ids(self, edited_query):
        """
        Fetch PubMed IDs (PMIDs) that match a given query.

        Args:
            query (str): A query string used to search PubMed.

        Returns:
            tuple: A tuple containing a list of PMIDs and the total count of PMIDs that match the query.
        """
        params = {
            "db": "pubmed",
            "term": edited_query,
            "retmode": "json",
            "email": self.email,  # Use the email provided in the constructor
            "retmax": 100000,  # Adjust based on how many results you expect
        }
        fetch_url = f"{self.base_url}esearch.fcgi"
        response = requests.get(fetch_url, params=params)
        response_json = response.json()
        pubmed_ids = response_json["esearchresult"]["idlist"]
        pubmed_ids_count = response_json["esearchresult"]["count"]

        return pubmed_ids, pubmed_ids_count

    def get_pubmed_ids_from_query(
        self, query=None, start_year=1983, end_year=2022, step=5
    ):
        """
        Generate queries for different year ranges and fetch PubMed IDs for each range.

        Args:
            query (str, optional): Custom query string. If None, a default query is used.
            start_year (int, optional): The starting year for the queries. Defaults to 1983.
            end_year (int, optional): The ending year for the queries. Defaults to 2022.
            step (int, optional): The step for the year range. Defaults to 5.

        Returns:
            list: A list of unique PMIDs obtained from all the queries.
        """
        if not query:
            query = """('selective serotonin reuptake inhibitor'[Title/Abstract] OR ('ssri'[Title/Abstract] OR 'fluvoxamine'[Title/Abstract] OR 'fluoxetine'[Title/Abstract] OR 'citalopram'[Title/Abstract] OR 'paroxetine'[Title/Abstract] OR 'sertraline'[Title/Abstract] OR 'escitalopram'[Title/Abstract]) AND 1983/01/01:2022/12/31[Date - Publication] AND 'english'[Language] )"""
            print(f"Using default query:\n{query}")

        time_limit = f"{start_year}/01/01:{end_year}/12/31[Date - Publication]"

        year_steps = list(range(start_year, end_year, step)) + [end_year]
        year_steps = [str(x) for x in year_steps]
        pubmed_ids = []
        pubmed_ids_count = []
        for i, year in enumerate(year_steps):
            start = year
            end = str(int(year) + 4) if i < len(year_steps) - 1 else str(end_year)
            edited_query = query.replace(
                time_limit, f"{start}/01/01:{end}/12/31[Date - Publication]"
            )
            ids, count = self.fetch_pubmed_ids(edited_query)
            pubmed_ids.extend(ids)
            pubmed_ids_count.append(count)
            print(
                f"In the range {start} to {end}, we found {count} articles. We fetched {len(ids)} articles."
            )

        unique_pubmed_ids = list(set(pubmed_ids))
        print(f"In total, we found {len(unique_pubmed_ids)} unique articles.")
        return unique_pubmed_ids

    # example usage
    # email = "email"
    # fetcher = PMIDsFromQuery(email)
    # pubmed_ids = fetcher.get_pubmed_ids_from_query()
    # print(pubmed_ids)
