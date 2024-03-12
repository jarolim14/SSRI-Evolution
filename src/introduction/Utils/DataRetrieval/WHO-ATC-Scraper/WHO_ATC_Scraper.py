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
        Fetches data from the website and extracts relevant links.
        """
        # Construct the URL with query parameters
        url = f'{self.base_url}?{"&".join([f"{key}={value}" for key, value in self.url_params.items()])}'

        # Send an HTTP GET request to fetch the HTML content
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all <a> elements
            all_hrefs = soup.find_all("a")

            # Initialize a flag to check if 'ANTIDEPRESSANTS' has been found
            found_antidepressants = False

            # Iterate through all <a> elements
            for ref in all_hrefs:
                if found_antidepressants:
                    # Process the links after 'ANTIDEPRESSANTS'
                    self.relevant_links.append(ref)

                elif ref.text == "ANTIDEPRESSANTS":
                    # Set the flag when 'ANTIDEPRESSANTS' is found
                    found_antidepressants = True
        else:
            print(
                f"Failed to retrieve the main page. Status code: {response.status_code}"
            )

    def scrape_and_store_data(self):
        """
        Scrapes data from the relevant links and stores it in DataFrames.
        """
        # Iterate through the relevant links
        for reference in self.relevant_links:
            href = reference["href"]
            text = reference.get_text()

            # Construct the full URL of the linked page
            linked_url = f"{self.base_url}/{href[2:]}"
            linked_response = requests.get(linked_url)

            # Check if the request for the linked page was successful (status code 200)
            if linked_response.status_code == 200:
                linked_soup = BeautifulSoup(linked_response.text, "html.parser")

                # Find all tables on the linked page
                tables = linked_soup.find_all("table")

                # Check if there are any tables
                if tables:
                    # Convert the first table to a Pandas DataFrame
                    html_string = str(tables[0])

                    # Wrap the HTML string in a 'StringIO' object
                    html_io = io.StringIO(html_string)

                    # Read the table data into a Pandas DataFrame
                    df = pd.read_html(html_io)[0]
                else:
                    # Handle the case where no tables were found
                    df = None  # Or any other appropriate action
                # Assign the first row as column names
                df.columns = df.iloc[0]

                # Drop the first row
                df = df.iloc[1:]

                # Fill NaN values using the previous value in the column
                df = df.ffill()

                # Add the DataFrame to the dictionary with the href text as the key
                self.dataframes_dict[text] = df

            else:
                print(f"Failed to retrieve linked page: {linked_url}")


if __name__ == "__main__":
    atc_scraper = ATCScraper()
    atc_scraper.fetch_data()
    atc_scraper.scrape_and_store_data()

    # Now, 'atc_scraper.dataframes_dict' contains DataFrames for each linked page
    # You can access them using the href text as keys
    for key, value in atc_scraper.dataframes_dict.items():
        print(f"Key: {key}")
        print(value)  # Print the DataFrame
