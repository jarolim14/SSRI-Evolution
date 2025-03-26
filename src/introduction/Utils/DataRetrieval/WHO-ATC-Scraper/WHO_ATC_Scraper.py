import io

import pandas as pd
import requests
from bs4 import BeautifulSoup


class ATCScraper:
    """
    A web scraper for retrieving data from WHOCC's ATC/DDD Index website.

    Attributes:
        base_url (str): The base URL of the website.
        url_params (dict): URL parameters for the specific query.
        relevant_links (list): List to store relevant links.
        dataframes_dict (dict): Dictionary to store DataFrames.

    Methods:
        fetch_data(): Fetches data from the website and extracts relevant links.
        scrape_and_store_data(): Scrapes data from the relevant links and stores it in DataFrames.
    """

    def __init__(
        self,
        base_url="https://www.whocc.no/atc_ddd_index/",
        code="N06A",
        show_description="no",
    ):
        """
        Initialize the ATCScraper with base URL and query parameters.

        Args:
            base_url (str, optional): The base URL of the website. Default is the WHOCC's ATC/DDD Index URL.
            code (str, optional): ATC code for the query. Default is 'N06A'.
            show_description (str, optional): Show description option. Default is 'no'.
        """
        self.base_url = base_url
        self.url_params = {"code": code, "showdescription": show_description}
        self.relevant_links = []
        self.dataframes_dict = {}

    def fetch_data(self):
        """
        Fetch data from the website and extract relevant links.

        Returns:
            list: List of relevant links found on the page.
        """
        response = requests.get(self.base_url, params=self.url_params)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all links that contain ATC codes
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and "code=" in href:
                self.relevant_links.append(href)
        
        return self.relevant_links

    def scrape_and_store_data(self):
        """
        Scrape data from the relevant links and store it in DataFrames.

        Returns:
            dict: Dictionary containing DataFrames for each ATC category.
        """
        for link in self.relevant_links:
            response = requests.get(link)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract ATC code from the URL
            atc_code = link.split("code=")[1].split("&")[0]
            
            # Create DataFrame from the table
            table = soup.find("table")
            if table:
                dataframe = pd.read_html(str(table))[0]
                self.dataframes_dict[atc_code] = dataframe
        
        return self.dataframes_dict


if __name__ == "__main__":
    atc_scraper = ATCScraper()
    atc_scraper.fetch_data()
    atc_scraper.scrape_and_store_data()

    # Now, 'atc_scraper.dataframes_dict' contains DataFrames for each linked page
    # You can access them using the href text as keys
    for key, value in atc_scraper.dataframes_dict.items():
        print(f"Key: {key}")
        print(value)  # Print the DataFrame
