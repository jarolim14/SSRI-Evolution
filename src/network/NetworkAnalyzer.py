import re
import string
from collections import Counter

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class TextAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess_text(
        self, text_corpus: list, use_stemming: bool = False, stemmer=None
    ) -> list:
        """
        Preprocesses a corpus of text documents by converting to lowercase, replacing numbers
        and punctuation with whitespace, and optionally applying stemming.

        Parameters:
        - text_corpus (list): A list of text documents to be preprocessed.
        - use_stemming (bool): Flag to enable or disable stemming. Default is False.
        - stemmer: A stemmer object with a .stem(word) method. Required if use_stemming is True.

        Returns:
        - list: A list of preprocessed text documents.
        """
        if use_stemming and stemmer is None:
            raise ValueError("Stemmer must be provided if use_stemming is True.")

        # Compile the regular expression for removing numbers
        number_re = re.compile(r"\b\d+\b")

        # Define a function for punctuation replacement
        replace_punct = lambda text: text.translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )

        text_corpus_preprocessed = []
        for doc in text_corpus:
            try:
                # Convert to lowercase
                doc = doc.lower()
                # Replace numbers with whitespace
                doc = number_re.sub(" ", doc)
                # Replace punctuation with whitespace
                doc = replace_punct(doc)
                # Optionally apply stemming
                if use_stemming:
                    doc = " ".join([stemmer.stem(word) for word in doc.split()])

                text_corpus_preprocessed.append(doc)
            except Exception as e:
                # Optionally, log the error or handle it as deemed appropriate
                print(f"Error processing document: {e}")
                continue

        return text_corpus_preprocessed

    def count_word_quantities(self, text_corpus):
        text_corpus_preprocessed = self.preprocess_text(text_corpus)
        text_corpus_filtered = [
            word
            for doc in text_corpus_preprocessed
            for word in word_tokenize(doc)
            if word not in self.stop_words
        ]
        word_quantities = Counter(text_corpus_filtered)
        return pd.DataFrame(
            list(word_quantities.items()), columns=["Word_freq", "Frequency"]
        )

    def tfidf_word_values(self, text_corpus):
        text_corpus_preprocessed = self.preprocess_text(text_corpus)
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
        X = vectorizer.fit_transform(text_corpus_preprocessed)
        feature_names = vectorizer.get_feature_names_out()

        tfidf_df = pd.DataFrame(X.todense(), columns=feature_names)
        df_tfidf = (
            tfidf_df.sum().sort_values(ascending=False).head(25).to_frame(name="TF-IDF")
        )
        df_tfidf["Word_tfidf"] = df_tfidf.index
        df_tfidf.reset_index(drop=True, inplace=True)
        return df_tfidf[["Word_tfidf", "TF-IDF"]]

    def merged_df(self, word_quantities, df_tfidf):
        df_wq = word_quantities.sort_values(
            by="Frequency", ascending=False
        ).reset_index(drop=True)
        df_tfidf = df_tfidf.sort_values(by="TF-IDF", ascending=False).reset_index(
            drop=True
        )
        df_merged = pd.concat([df_wq, df_tfidf], axis=1)
        return df_merged


class CommunityExplorer:
    def __init__(self, df, cluster_column):
        self.df = df
        self.cluster_column = cluster_column
        self.text_analyzer = TextAnalyzer()

    def create_cluster_sheets(
        self, sort=True, n=15, sort_column="eigenvector_cluster_centrality"
    ):
        sheets = []
        top_words_per_cluster = []
        for cluster in self.df[self.cluster_column].unique():
            paper_count = self.df[self.df[self.cluster_column] == cluster].shape[0]
            cols_to_show = [
                "title",
                "year",
                # "title_abstract",
                "abstract",
                "doi",
                # "specter2_embeddings",
            ]
            if paper_count < n:
                n = paper_count
            if sort:
                sheet = (
                    self.df[self.df[self.cluster_column] == cluster]
                    .sort_values(by=sort_column, ascending=False)
                    .head(n)
                )
                cols_to_show.append(sort_column)
            else:
                sheet = self.df[self.df[self.cluster_column] == cluster].sample(
                    n, random_state=1887
                )
            sheets.append(sheet[cols_to_show])
            text_corpus = sheet["title_abstract"].values
            word_quantities = self.text_analyzer.count_word_quantities(text_corpus)
            df_tfidf = self.text_analyzer.tfidf_word_values(text_corpus)
            top_words_per_cluster.append(
                self.text_analyzer.merged_df(word_quantities, df_tfidf).head(20)
            )
        self.sheets = sheets
        self.top_words_per_cluster = top_words_per_cluster

    def create_summary_sheet(self):
        # Calculate the full value counts per cluster
        summary_sheet = self.df[self.cluster_column].value_counts().reset_index()
        summary_sheet.columns = ["Cluster", "Nr of Pubs"]
        # add column with mean and sd of year
        summary_sheet["Mean Year"] = (
            self.df.groupby(self.cluster_column)["year"].mean().round(1)
        )
        summary_sheet["SD Year"] = (
            self.df.groupby(self.cluster_column)["year"].std().round(1)
        )
        summary_sheet["Given Label"] = ""
        # add top words
        rows = [
            self.top_words_per_cluster[cluster]["Word_tfidf"].values
            for cluster in range(len(summary_sheet))
        ]
        word_df = pd.DataFrame(rows)

        word_df.columns = [f"Word_{i}" for i in range(20)]

        self.df_summary = pd.concat([summary_sheet, word_df], axis=1)

    # save to excel
    def save_to_excel(self, file_name):
        # merge all sheets into one excel file
        with pd.ExcelWriter(file_name) as writer:
            self.df_summary.to_excel(writer, sheet_name="summary")
            for i, sheet in enumerate(self.sheets):
                sheet.to_excel(writer, sheet_name=f"cluster_{i}")

    def full_return(self):
        return self.df_summary, self.sheets


"""
example usage
df = pd.read_csv("data.csv")
ce = CommunityExplorer(df, "cluster")
ce.create_cluster_sheets()
ce.create_summary_sheet()
ce.save_to_excel("output.xlsx")
df_summary, sheets = ce.full_return()
"""


import matplotlib.pyplot as plt


class ClusterPlotter:
    def __init__(self, df, cluster_column):
        self.df = df
        self.cluster_column = cluster_column

    def plot_rolling_cluster_distribution(self, cluster_number_label_dict, window=2):
        df_sub = self.df[
            self.df[self.cluster_column].isin(cluster_number_label_dict.keys())
        ]

        # Set the size of the figure
        plt.figure(figsize=(15, 10))

        # Perform data manipulation
        cluster_distribution = (
            df_sub.groupby("year")[self.cluster_column]
            .value_counts(normalize=True)
            .unstack()
        )
        rolling_cluster_distribution = cluster_distribution.rolling(window).mean()

        # Plot the data
        plt.plot(rolling_cluster_distribution)

        # Legend with both cluster number and label
        plt.legend(
            labels=[
                f"{i}: {cluster_number_label_dict[i]}"
                for i in rolling_cluster_distribution.columns
            ],
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        plt.title("Cluster Distribution Over Time (Rolling Mean)")
        plt.xlabel("Year")
        plt.ylabel("Normalized Cluster Count")

        # Display the plot
        plt.show()


"""
# Assuming df_0025 is your DataFrame and 'cluster' is the name of the cluster column
# Initialize the class with your DataFrame and the column name
cluster_analysis = ClusterListPlotter(df_0025, "cluster")

# Plot the rolling cluster distribution for specified cluster numbers
cluster_analysis.plot_rolling_cluster_distribution(
    cluster_numbers_list=[0, 1, 2, 3, 4, 5, 6], window=2
)
"""
