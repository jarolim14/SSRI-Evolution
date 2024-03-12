import json
import re

import pandas as pd


class TextProcessor:
    """
    Class for processing text data.
    """

    def __init__(self, df):
        self.df = df

    def save_na_dict_to_json(
        self,
        columns=["title", "abstract"],
        file_path="./output/removal_log/na_log_text_cols.json",
    ):
        na_dict = self.df[columns].isna().sum().to_dict()
        json_data = json.dumps(na_dict)
        with open(file_path, "w") as json_file:
            json_file.write(json_data)
        print(f"NA dict saved to {file_path}")

    def merge_title_abstract(self, title, abstract):
        """
        Merge title and abstract into one text.
        """
        # Check if title is not a string or is None
        if not isinstance(title, str) or title is None:
            title = ""

        # Check if abstract is not a string or is None
        if not isinstance(abstract, str) or abstract is None:
            abstract = ""

        # Merge title and abstract
        text = title + ". " + abstract

        return text

    def clean_text(self, title_abstract):
        # Remove specific punctuation and extra brackets content
        regex_pattern = r"\[.*?\]|[^\w\s();:.!?-]"
        title_abstract = re.sub(regex_pattern, "", title_abstract)

        # Condense multiple punctuations and reduce excess whitespaces
        title_abstract = re.sub(r"([;:.!?-])\s*[;:.!?-]+\s*", r"\1 ", title_abstract)
        title_abstract = re.sub(r"\s{2,}", " ", title_abstract)

        # strip leading and trailing whitespaces
        title_abstract = title_abstract.strip()

        return title_abstract

    def remove_starting_phrases(self, text):
        """
        Remove starter phrases from the beginning of the text.
        Starter phrases are identified as any word ending with a colon at the beginning of the text.
        In practice, it is usually "Introduction:" or "Background:".
        """

        # Check if text is not a string or is None
        if not isinstance(text, str) or text is None:
            return ""

        # Regular expression to match a starting word ending with a colon
        starter_phrase_pattern = r"^\s*\w+:\s*"

        # Remove the starter phrase if found
        cleaned_text = re.sub(starter_phrase_pattern, "", text, count=1)

        # Make it case insensitive and remove only if it's the first word
        cleaned_text = cleaned_text.replace("Abstract", "", 1).replace("Title", "", 1)

        return cleaned_text

    def remove_ending_statements(self, abstract):
        """
        Removes parts of the abstract after the second-to-last or last sentence
        if certain conditions are met. Specifically, if the sentence starts with
        'copyright', starts with 4 consecutive digits, or ends with 4 consecutive digits.

        Parameters:
        abstract (str): The abstract to be processed.

        Returns:
        str: Modified abstract with specific end sentences removed.
        """
        # Split abstract into sentences and filter out very short fragments
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", abstract)
            if len(sentence.strip()) > 3
        ]

        # Define a regex pattern for matching the criteria
        pattern_start_with_copy_or_digits = re.compile(
            r"(?i)^(copyright|\d{4}|\(c\)|all rights reserved|©)"
        )
        pattern_end_with_digits = re.compile(r"\d{4}$")

        # Function to check the conditions on a sentence
        def check_conditions(sentence):
            return pattern_start_with_copy_or_digits.match(
                sentence
            ) or pattern_end_with_digits.search(sentence)

        # Check the second-to-last sentence first
        if len(sentences) >= 2 and check_conditions(sentences[-2]):
            return " ".join(sentences[:-2])

        # If conditions not met in second-to-last, check the last sentence
        elif len(sentences) >= 1 and check_conditions(sentences[-1]):
            return " ".join(sentences[:-1])

        # If none of the conditions are met, return the original abstract
        return abstract

    def clean_text_and_remove_start_and_ending_statements(self):
        """
        use both functions above to clean text
        """

        clean_title = self.df["title"].apply(self.remove_starting_phrases)
        clean_abstract = self.df["abstract"].apply(self.remove_starting_phrases)
        clean_abstract = [
            self.remove_ending_statements(abstract) for abstract in clean_abstract
        ]
        title_abstract = [
            self.merge_title_abstract(title, abstract)
            for title, abstract in zip(clean_title, clean_abstract)
        ]

        self.df["title_abstract"] = [self.clean_text(t) for t in title_abstract]
        return self.df


df = pd.read_pickle("data/03-connected/scopus_cleaned_connected.pkl")


# Usage:
cols = ["abstract", "title"]
file_path = "./output/removal_log/na_log_text_cols.json"

tp = TextProcessor(df)
tp.save_na_dict_to_json(cols, file_path)
df = tp.clean_text_and_remove_start_and_ending_statements()
