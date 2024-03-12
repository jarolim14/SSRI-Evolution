import logging
import xml.etree.ElementTree as ET
from typing import Optional

import requests
from pmidcite.icite.downloader import get_downloader


class DoiFetcherFromPMID:
    """
    A class for fetching Digital Object Identifiers (DOIs) for given PubMed IDs (PMIDs).

    This class provides methods to fetch DOIs using two different services:
    1. pmidcite: A service that provides citation information including DOIs for PubMed articles.
    2. NCBI's E-utilities: A suite of web services provided by the National Center for Biotechnology Information (NCBI) for querying and retrieving data from NCBI's databases, including PubMed.

    The class first attempts to fetch the DOI using pmidcite. If unsuccessful, it falls back to using NCBI's E-utilities.

    Methods:
    pmidcite(pmid: str) -> Optional[str]:
        Fetches the DOI using pmidcite for a given PMID. Returns the DOI if found, or None otherwise.

    e_utilities(pmid: str) -> Optional[str]:
        Fetches the DOI for a given PMID using NCBI's E-utilities. Returns the DOI if found, or None otherwise.

    fetch_doi(pmid: str) -> Optional[str]:
        Attempts to fetch a DOI for a given PMID using both pmidcite and E-utilities. Returns the DOI if found by either service, or None if both fail.

    Parameters:
    pmidcite_downloader: An instance of a downloader class used to interface with the pmidcite service. This is optional and a default downloader is used if not provided.

    Example Usage:
    doi_fetcher = DoiFetcherFromPMID()
    doi = doi_fetcher.fetch_doi("12345678")
    print(doi)  # Prints the DOI corresponding to the given PMID, if found.

    Note:
    This class requires the `requests` library for HTTP requests and `xml.etree.ElementTree` for XML parsing.
    It also utilizes a custom downloader for the pmidcite service, which should be provided during initialization.
    """

    def __init__(self, pmidcite_downloader=get_downloader()):
        """
        Initialize the DoiFetcherFromPMID.

        Parameters:
        pmidcite_downloader: The downloader instance to use for pmidcite requests.
        """
        self.pmidcite_downloader = pmidcite_downloader

    def pmidcite(self, pmid: str) -> Optional[str]:
        """
        Fetch DOI using pmidcite.

        Parameters:
        pmid (str): The PubMed ID for which to fetch the DOI.

        Returns:
        Optional[str]: The DOI if found, None otherwise.
        """
        try:
            nih_entry = self.pmidcite_downloader.get_icite(pmid)
            if nih_entry.dct["doi"]:
                return nih_entry.dct["doi"]
        except Exception as e:
            logging.error(f"Error fetching DOI from pmidcite for PMID {pmid}: {e}")
            return None

    def e_utilities(self, pmid: str) -> Optional[str]:
        """
        Fetch DOI for a given PubMed ID (PMID) using NCBI's E-utilities.
        """
        try:
            efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
            response = requests.get(efetch_url, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            for article in root.findall(".//ArticleIdList/ArticleId"):
                if article.get("IdType") == "doi":
                    return article.text

        except requests.RequestException as e:
            logging.error(
                f"HTTP error fetching DOI from E-utilities for PMID {pmid}: {e}"
            )
            return None
        except ET.ParseError as e:
            logging.error(
                f"XML parsing error fetching DOI from E-utilities for PMID {pmid}: {e}"
            )
            return None

    def fetch_doi(self, pmid: str) -> Optional[str]:
        """
        Fetch DOI for a given PubMed ID (PMID).
        """
        if not pmid:
            return None
        doi = self.pmidcite(pmid)
        if doi:
            return doi

        doi = self.e_utilities(pmid)
        if doi:
            return doi

        return None


if __name__ == "__main__":
    # Test the DoiFetcherFromPMID class
    doi_fetcher = DoiFetcherFromPMID()
    doi = doi_fetcher.fetch_doi("12345678")
    print(doi)
