import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import tqdm


class ArticleDetailsFromPMID:
    """
    A class for fetching details of articles corresponding to PubMed IDs.
    First, fetch_article_details() is called to fetch details of articles corresponding to a list of PubMed IDs.
    Then, parse_article_data() is called to parse the XML data returned by PubMed E-utilities.
    Then, fetch_details_and_create_dataframe() is called to fetch article details in batches and compile them into a pandas DataFrame.
    Then, clean_dataframe() is called to clean the DataFrame.
    """

    def __init__(self, email):
        """
        Initialize the ArticleDetailsFromPMID with the provided email address.

        Args:
            email (str): An email address to be used for queries to the PubMed E-utilities API.
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = email

    def fetch_article_details(self, pubmed_ids):
        """
        Fetch details of articles corresponding to a list of PubMed IDs.

        Args:
            pubmed_ids (list): A list of PubMed IDs.

        Returns:
            list: A list of dictionaries, each containing details of an article, and the raw XML response text.
        """
        fetch_url = f"{self.base_url}efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),
            "retmode": "xml",
            "email": self.email,
        }
        response = requests.get(fetch_url, params=params)
        try:
            articles, raw_xml = self.parse_article_data(response.text)
            return articles, raw_xml
        except ET.ParseError as e:
            print(f"XML Parsing Error: {e}")
            print(
                f"Response leading to error: {response.text[:500]}"
            )  # Print first 500 chars of the response
            return [], response.text

    @staticmethod
    def parse_article_data(xml_data):
        """
        Parse XML data from PubMed E-utilities to extract article details.

        Args:
            xml_data (str): XML data as a string.

        Returns:
            tuple: A tuple containing a list of dictionaries, each with details of an article, and the raw XML data.
        """
        root = ET.fromstring(xml_data)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            data = {
                "title": article.findtext(".//ArticleTitle"),
                "year": article.findtext(".//PubDate/Year"),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.findtext('.//PMID')}/",
                "abstract": article.findtext(".//Abstract/AbstractText"),
                "doi": article.findtext(".//ELocationID[@EIdType='doi']"),
                "journal": article.findtext(".//Journal/Title"),
                "authors": ArticleDetailsFromPMID.parse_authors(article),
                "keywords": [
                    keyword.text
                    for keyword in article.findall(".//KeywordList/Keyword")
                ],
                "pubmed_id": article.findtext(".//PMID"),
                "publication_type": article.findtext(".//PublicationType"),
            }
            articles.append(data)
        return articles, xml_data

    @staticmethod
    def parse_authors(article):
        """
        Parse author details from an article element.

        Args:
            article (xml.etree.ElementTree.Element): An article element.

        Returns:
            list: A list of author names in the format "LastName, ForeName".
        """
        authors = []
        for author in article.findall(".//Author"):
            last_name = author.findtext("LastName") or ""
            fore_name = author.findtext("ForeName") or ""
            author_name = (last_name + ", " + fore_name).strip(", ")
            if author_name:
                authors.append(author_name)
        return authors

    def fetch_details_and_create_dataframe(self, pubmed_ids, batch_size=100):
        """
        Fetch article details in batches and compile them into a pandas DataFrame.

        Args:
            pubmed_ids (list): A list of PubMed IDs for which to fetch details.
            batch_size (int, optional): The number of articles to fetch in each batch. Defaults to 100.

        Returns:
            DataFrame: A pandas DataFrame containing details of all fetched articles and the raw XML response.
        """
        full_data = []
        raw_xml_responses = []
        print(
            f"Fetching details for {len(pubmed_ids)} articles in batches of {batch_size}..."
        )
        for i in tqdm.tqdm(range(0, len(pubmed_ids), batch_size)):
            batch_ids = pubmed_ids[i : i + batch_size]
            articles, raw_xml = self.fetch_article_details(batch_ids)
            full_data.extend(articles)
            raw_xml_responses.append(raw_xml)

        # Create a DataFrame
        self.df = pd.DataFrame(full_data)
        self.df["raw_xml"] = pd.Series(raw_xml_responses * batch_size).iloc[
            : len(self.df)
        ]
        return self.df

    def clean_dataframe(self):
        print("Cleaning dataframe...")
        # prepare df for merging with scopus later
        # already exists in scopus
        self.df["source"] = "PubMed"

        # remove year 2023
        self.df_allyears = len(self.df)
        self.df = self.df[self.df["year"] != "2023"]

        print(f"Removed on year 2023: {self.df_allyears - len(self.df)}")

        # remove document types
        publication_types_to_remove = [
            "Bibliography",
            "Lecture",
            "News",
            "Newspaper Article",
            "Consensus Development Conference",
            "Congress",
            "Interview",
            "Published Erratum",
            "Biography",
            "Preprint",
            "Patient Education Handout",
        ]

        self.df_all = len(self.df)
        self.df = self.df[
            ~self.df["publication_type"].isin(publication_types_to_remove)
        ]
        self.df_removed = len(self.df)

        print(f"Document types removed: {publication_types_to_remove}")
        print(f"Removed on document types: {self.df_all - self.df_removed}")
        print(f"New length: {self.df_removed}")

        # rename columns
        self.df = self.df.rename(
            columns={
                "url": "link",
                "keywords": "author_keywords",
                "publication_type": "document_type",
            }
        )

        # set np.nan for all possible nan values
        self.df["title"] = self.df["title"].apply(
            lambda x: np.nan if x in ["", "No Title", None] else x
        )
        # set np.nan for all possible nan values
        self.df["abstract"] = self.df["abstract"].apply(
            lambda x: np.nan if x in ["", "No Abstract", None] else x
        )

    # example usage
    # email = "lukas.westphal@outlookcom"
    # pubmed_ids = [123, 456, 789]
    # article_details = ArticleDetailsFromPMID(email)
    # df = article_details.fetch_details_and_create_dataframe(pubmed_ids)
    # df.head()
