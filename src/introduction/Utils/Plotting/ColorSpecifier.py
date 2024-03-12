import sys

import pandas as pd
import seaborn as sns

sys.path.append(
    "02_AllThingsData/Utils/DataRetrieval/WHO-ATC-Scraper/WHO_ATC_Scraper.py"
)

from WHO_ATC_Scraper import ATCScraper

# Scrape WHO ATC codes

atc_scraper = ATCScraper()
atc_scraper.fetch_data()
atc_scraper.scrape_and_store_data()

# Process Dataframes, assign colors, and save


colors = [
    "Blues",
    "Greens",
    "Reds",
    "Oranges",
    "Purples",
]  # Removed extra spaces

for i, (key, value) in enumerate(atc_scraper.dataframes_dict.items()):
    # Remove duplicates based on ATC code
    value.drop_duplicates(subset=["ATC code"], inplace=True)
    # Add key to DataFrame as a column
    value["DrugClass"] = key
    # Get the color palette
    color_palette = sns.color_palette(colors[i], n_colors=len(value))
    # Create a "Color" column and assign colors from the palette
    value["Color"] = color_palette
    value["ColorPalette"] = colors[i]
    # drop columns DDD, U, Adm.Rm Note
    # value.drop(columns=["DDD", "U", "Adm.R", "Note"], inplace=True)
    # Print the DataFrame
    # print("")
    # print(f"Key: {key}")
    # print("")
    # print(value)


# concat
df = pd.concat(atc_scraper.dataframes_dict.values(), ignore_index=True)
df.rename(columns={"ATC code": "ATC_code", "Name": "DrugName"}, inplace=True)
df.to_csv("02_AllThingsData/Utils/Plotting/Data/ColorPalettes_PerATC.csv")

df.head(3)

for key, value in atc_scraper.dataframes_dict.items():
    # Print the DataFrame
    print("")
    print(f"Key: {key}")
    print("")
    print(value)
