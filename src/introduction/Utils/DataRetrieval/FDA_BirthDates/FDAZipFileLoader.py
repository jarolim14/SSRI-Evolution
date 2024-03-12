import zipfile
from io import BytesIO

import pandas as pd
import requests


class FDAZipFileLoader:
    """
    A class for downloading, extracting, and reading data from a zip file.

    Attributes:
        zip_url (str): The URL of the zip file to download.
        desired_file_name (str): The name of the file to extract and read.
        df (pd.DataFrame): A Pandas DataFrame to store the extracted data.

    Methods:
        load_data(): Downloads the zip file, extracts the desired file, and reads it into a Pandas DataFrame.
    """

    def __init__(self, zip_url, desired_file_name):
        """
        Initialize the FDAZipFileLoader with the zip file URL and desired file name.

        Args:
            zip_url (str): The URL of the zip file to download.
            desired_file_name (str): The name of the file to extract and read.
        """
        self.zip_url = zip_url
        self.desired_file_name = desired_file_name
        self.df = None

    def load_data(self):
        """
        Downloads the zip file, extracts the desired file, and reads it into a Pandas DataFrame.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the data from the desired file.
        """
        # Send an HTTP GET request to download the zip file
        response = requests.get(self.zip_url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the zip file from the response content
            with zipfile.ZipFile(BytesIO(response.content), "r") as zip_archive:
                # Check if the desired file exists within the archive
                if self.desired_file_name in zip_archive.namelist():
                    # Read the contents of the desired file into a Pandas DataFrame
                    with zip_archive.open(self.desired_file_name) as desired_file:
                        self.df = pd.read_csv(
                            desired_file, delimiter="~", encoding="utf-8"
                        )

                        # Convert the 'Approval_Date' column to datetime
                        self.df["Approval_Date"] = pd.to_datetime(
                            self.df["Approval_Date"],
                            format="%b %d, %Y",
                            errors="coerce",
                        )

                        return self.df
                else:
                    print(
                        f"The file '{self.desired_file_name}' does not exist in the zip archive."
                    )
        else:
            print(
                f"Failed to download the zip file. Status code: {response.status_code}"
            )


if __name__ == "__main__":
    # URL of the zip file
    zip_url = "https://www.fda.gov/media/76860/download?attachment"

    # Define the name of the file you want to extract
    desired_file_name = "products.txt"

    # Create an instance of the FDAZipFileLoader
    loader = FDAZipFileLoader(zip_url, desired_file_name)

    # Load the data
    df = loader.load_data()

    if df is not None:
        # Now 'df' contains your data in a DataFrame
        print("Data loaded successfully!")
        print(df.head())
