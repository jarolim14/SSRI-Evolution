import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


class MedStatScraperVolumeSoldADs:
    def __init__(self, url):
        self.url = url
        self.df = None

    def scrape_data(self):
        # Send an HTTP GET request to the URL
        response = requests.get(self.url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the table with the class "statistical-data-table"
            table = soup.find("table", class_="statistical-data-table")

            if table:
                # Initialize a list to store the text content of each <tr> element
                tr_texts = []

                # Find all <tr> elements within the table
                tr_elements = table.find_all("tr")

                # Loop through each <tr> element, extract its text, and store it in the list
                for tr in tr_elements:
                    tr_text = [
                        i.get_text(strip=True) for i in tr if i.get_text().strip() != ""
                    ]  # Get the text content, removing extra spaces
                    tr_texts.append(tr_text)

                # Now tr_texts contains the text content of each <tr> element
                return tr_texts
            else:
                print("Table not found on the page.")
        else:
            print("Failed to retrieve the page. Status code:", response.status_code)

    def create_dataframe(self, tr_texts):
        if not tr_texts:
            return None

        # Define colnames
        years = tr_texts[0]
        colnames = ["ATC-code", "Unit"] + years

        # Every list after the second as a row
        rows = tr_texts[2:]

        # Create DataFrame
        df = pd.DataFrame(rows, columns=colnames)

        # Get drug names from ATC-code column
        df["Drug"] = (
            df["ATC-code"]
            .str.split(" ")
            .apply(lambda x: " ".join(x[1:]).replace("(", "").replace(")", ""))
        )

        df["ATC-code"] = df["ATC-code"].apply(lambda x: x.split(" ")[0])

        # Split ATC-code column to get only the code

        # Move the Drug column to the front
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        # Specify '-' as NaN and convert numerical columns to integers
        df.iloc[:, 3:] = df.iloc[:, 3:].applymap(
            lambda cell_value: int(cell_value.replace(",", ""))
            if cell_value != "-"
            else np.nan
        )

        self.df = df  # Store the DataFrame in the class attribute

    def get_dataframe(self):
        return self.df


if __name__ == "__main__":
    url = "https://medstat.dk/en/viewDataTables/medicineAndMedicalGroups/%7B%22year%22:[%222022%22,%222021%22,%222020%22,%222019%22,%222018%22,%222017%22,%222016%22,%222015%22,%222014%22,%222013%22,%222012%22,%222011%22,%222010%22,%222009%22,%222008%22,%222007%22,%222006%22,%222005%22,%222004%22,%222003%22,%222002%22,%222001%22,%222000%22,%221999%22,%221998%22,%221997%22,%221996%22],%22region%22:[%220%22],%22gender%22:[%22A%22],%22ageGroup%22:[%22A%22],%22searchVariable%22:[%22sold_volume%22],%22errorMessages%22:[],%22atcCode%22:[%22N06A%22,%22N06AA%22,%22N06AA02%22,%22N06AA03%22,%22N06AA04%22,%22N06AA05%22,%22N06AA06%22,%22N06AA07%22,%22N06AA09%22,%22N06AA10%22,%22N06AA11%22,%22N06AA12%22,%22N06AA16%22,%22N06AA17%22,%22N06AA21%22,%22N06AB%22,%22N06AB03%22,%22N06AB04%22,%22N06AB05%22,%22N06AB06%22,%22N06AB08%22,%22N06AB10%22,%22N06AF%22,%22N06AF01%22,%22N06AG%22,%22N06AG02%22,%22N06AX%22,%22N06AX03%22,%22N06AX06%22,%22N06AX11%22,%22N06AX12%22,%22N06AX16%22,%22N06AX18%22,%22N06AX21%22,%22N06AX22%22,%22N06AX26%22,%22N06AX27%22],%22sector%22:[%222%22]%7D"
    scraper = MedStatScraper(url)
    tr_texts = scraper.scrape_data()
    scraper.create_dataframe(tr_texts)


class MedStatScraperNrUsersSSRI:
    def __init__(self, url):
        self.url = url
        self.html = None
        self.table_data = None
        self.dataframe = None

    def fetch_html(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            self.html = response.text
        else:
            raise Exception(f"Failed to fetch URL: {self.url}")

    def parse_table(self):
        if self.html is None:
            raise Exception("HTML content not fetched. Call fetch_html() first.")

        soup = BeautifulSoup(self.html, "html.parser")
        table = soup.find("table", class_="statistical-data-table")
        if not table:
            raise Exception("No table found in the HTML content.")

        tr_elements = table.find_all("tr")
        self.table_data = [
            [i.get_text(strip=True) for i in tr if i.get_text().strip() != ""]
            for tr in tr_elements
        ]

    def format_dataframe(self):
        if self.table_data is None:
            raise Exception("Table data not parsed. Call parse_table() first.")

        years = self.table_data[0]
        colnames = ["ATC-code"] + years
        rows = self.table_data[2:]
        df = pd.DataFrame(rows, columns=colnames)

        # Extract and format Drug names
        df["Drug"] = (
            df["ATC-code"]
            .str.split(" ")
            .apply(lambda x: " ".join(x[1:]).replace("(", "").replace(")", ""))
        )
        df["ATC-code"] = df["ATC-code"].apply(lambda x: x.split(" ")[0])

        # Move the Drug column to the front
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        # Handle missing values and convert data to numeric
        df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

        # Delete columns where all values are NaN
        df.dropna(axis=1, how="all", inplace=True)

        self.dataframe = df

    def get_dataframe(self):
        if self.dataframe is None:
            raise Exception("DataFrame not created. Call format_dataframe() first.")
        return self.dataframe


# example
# url = 'https://medstat.dk/en/viewDataTables/medicineAndMedicalGroups/%7B%22year%22:[%222022%22,%222021%22,%222020%22,%222019%22,%222018%22,%222017%22,%222016%22,%222015%22,%222014%22,%222013%22,%222012%22,%222011%22,%222010%22,%222009%22,%222008%22,%222007%22,%222006%22,%222005%22,%222004%22,%222003%22,%222002%22,%222001%22,%222000%22,%221999%22,%221998%22,%221997%22,%221996%22],%22region%22:[%220%22],%22gender%22:[%22A%22],%22ageGroup%22:[%22A%22],%22searchVariable%22:[%22people_count_1000%22],%22errorMessages%22:[],%22atcCode%22:[%22N06AB%22,%22N06AB03%22,%22N06AB04%22,%22N06AB05%22,%22N06AB06%22,%22N06AB08%22,%22N06AB10%22],%22sector%22:[%220%22]%7D'
# MedStatScraperNrUsers = MedStatScraperNrUsers(url)
# MedStatScraperNrUsers.fetch_html()
# MedStatScraperNrUsers.parse_table()
# MedStatScraperNrUsers.format_dataframe()
# df = MedStatScraperNrUsers.get_dataframe()
