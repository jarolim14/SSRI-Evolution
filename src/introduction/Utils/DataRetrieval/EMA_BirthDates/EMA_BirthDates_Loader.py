import logging
from io import BytesIO

import pandas as pd
import requests
import urllib3


class EMA_BirthDates_Loader:
    def __init__(self, MedStatScraper, ATCScraper, verbose=True, url=None, logger=None):
        self.verbose = verbose
        self.url = (
            url
            if url
            else "https://www.ema.europa.eu/documents/other/list-european-union-reference-dates-frequency-submission-periodic-safety-update-reports-psurs_.xlsx"
        )
        self.logger = logger if logger else logging.getLogger(__name__)
        self.MedStatScraper = MedStatScraper
        self.ATCScraper = ATCScraper
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _load_data_from_url(self, url):
        response = requests.get(url, verify=False)
        if response.status_code != 200:
            self.logger.error(
                f"Failed to retrieve the Excel file. Status code: {response.status_code}"
            )
            return None
        return pd.read_excel(BytesIO(response.content), skiprows=17)

    def load_ema_data(self):
        self.df = self._load_data_from_url(self.url)
        if self.df is not None:
            self.logger.info("Data loaded successfully")

    def load_medstat_data(self, url=None):
        url = (
            url
            if url
            else "https://medstat.dk/en/viewDataTables/medicineAndMedicalGroups/%7B%22year%22:[%222022%22,%222021%22,%222020%22,%222019%22,%222018%22,%222017%22,%222016%22,%222015%22,%222014%22,%222013%22,%222012%22,%222011%22,%222010%22,%222009%22,%222008%22,%222007%22,%222006%22,%222005%22,%222004%22,%222003%22,%222002%22,%222001%22,%222000%22,%221999%22,%221998%22,%221997%22,%221996%22],%22region%22:[%220%22],%22gender%22:[%22A%22],%22ageGroup%22:[%22A%22],%22searchVariable%22:[%22sold_volume%22],%22errorMessages%22:[],%22atcCode%22:[%22N06A%22,%22N06AA%22,%22N06AA02%22,%22N06AA03%22,%22N06AA04%22,%22N06AA05%22,%22N06AA06%22,%22N06AA07%22,%22N06AA09%22,%22N06AA10%22,%22N06AA11%22,%22N06AA12%22,%22N06AA16%22,%22N06AA17%22,%22N06AA21%22,%22N06AB%22,%22N06AB03%22,%22N06AB04%22,%22N06AB05%22,%22N06AB06%22,%22N06AB08%22,%22N06AB10%22,%22N06AF%22,%22N06AF01%22,%22N06AG%22,%22N06AG02%22,%22N06AX%22,%22N06AX03%22,%22N06AX06%22,%22N06AX11%22,%22N06AX12%22,%22N06AX16%22,%22N06AX18%22,%22N06AX21%22,%22N06AX22%22,%22N06AX26%22,%22N06AX27%22],%22sector%22:[%222%22]%7D"
        )
        scraper = self.MedStatScraper(url)
        tr_texts = scraper.scrape_data()
        scraper.create_dataframe(tr_texts)
        self.df_dk_drugs = scraper.df
        self.dk_drugs = [
            drug.lower()
            for drug in scraper.df.Drug.unique().tolist()
            if " " not in drug
        ]
        self.logger.info("MedStat data loaded successfully")

    def load_all_atc_drugs(self):
        atc_scraper = self.ATCScraper()
        atc_scraper.fetch_data()
        atc_scraper.scrape_and_store_data()
        all_drugs = [
            drug.lower().strip()
            for sublist in atc_scraper.dataframes_dict.values()
            for drug in sublist["Name"].tolist()
        ]
        self.all_drugs = list(set(all_drugs))

    def preprocess_data(self, all_atc_drugs=False, only_dk_drugs=False):
        if self.df is None:
            self.logger.error("Data not loaded. Cannot preprocess.")
            return

        self.df = self.df.iloc[:, 0:3]
        self.df.rename(columns={self.df.columns[1]: "DrugName"}, inplace=True)

        if only_dk_drugs:
            self.load_medstat_data()
            self.df = self.df[self.df["DrugName"].isin(self.dk_drugs)]

        elif all_atc_drugs:
            self.load_all_atc_drugs()
            self.df = self.df[self.df["DrugName"].isin(self.all_drugs)]

        self.df["BirthDate"] = pd.to_datetime(
            self.df.iloc[:, 2], format="%d/%m/%Y", errors="coerce"
        )
        self.df.drop(columns=[self.df.columns[2]], inplace=True)
        self.df.dropna(subset=["BirthDate"], inplace=True)
        self.df["BirthYear"] = self.df["BirthDate"].dt.year
        self.df.sort_values(by=["BirthYear"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.logger.info("Data preprocessed successfully")
