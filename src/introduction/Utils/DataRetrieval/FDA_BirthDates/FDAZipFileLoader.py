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
        dataframe (pd.DataFrame): A Pandas DataFrame to store the extracted data.

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
        self.dataframe = None

    def load_data(self):
        """
        Download the zip file, extract the desired file, and read it into a DataFrame.

        Returns:
            pd.DataFrame: The extracted data as a Pandas DataFrame.

        Raises:
            Exception: If the zip file cannot be downloaded or the desired file is not found.
        """
        try:
            # Download the zip file
            response = requests.get(self.zip_url)
            response.raise_for_status()

            # Create a BytesIO object from the content
            zip_content = BytesIO(response.content)

            # Open the zip file
            with zipfile.ZipFile(zip_content) as zip_file:
                # Check if the desired file exists in the zip
                if self.desired_file_name not in zip_file.namelist():
                    raise FileNotFoundError(
                        f"File '{self.desired_file_name}' not found in the zip file."
                    )

                # Read the desired file into a DataFrame
                with zip_file.open(self.desired_file_name) as file:
                    self.dataframe = pd.read_csv(file)

            return self.dataframe

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download zip file: {str(e)}")
        except zipfile.BadZipFile:
            raise Exception("Invalid zip file.")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")


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
