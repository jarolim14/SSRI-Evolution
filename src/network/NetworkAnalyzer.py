import sys

import pandas as pd

sys.path.append("/Users/jlq293/Projects/Study-1-Bibliometrics/src/network")
from NetworkAnalyzerUtils import TextAnalyzer


class CommunityExplorer:
    def __init__(
        self,
        df,
        cluster_column,
        sort_column="eigenvector_cluster_centrality",
        nr_tiles=15,
        nr_words=20,
    ):
        self.df = df
        self.cluster_column = cluster_column
        self.text_analyzer = TextAnalyzer()
        self.sort_column = sort_column
        self.nr_tiles = nr_tiles
        self.nr_words = nr_words

    def create_summary_dict(self):
        summary_cols = [
            "Cluster",
            "Nr of Pubs",
            "Mean Year",
            "SD Year",
            "Given Label",
        ]
        summary_cols.extend([f"Word_{i}" for i in range(self.nr_words)])

        summary_dict = {col: [] for col in summary_cols}
        return summary_dict

    def cluster_titles_sheets_creator(self, df_cluster, nr_tiles_adapted):
        cols_to_show = [
            "title",
            "year",
            "abstract",
            "eid",
            self.sort_column,
        ]
        # Get the sorted and random titles
        sorted_titles = (
            df_cluster.sort_values(by=self.sort_column, ascending=False).head(
                nr_tiles_adapted
            )[cols_to_show]
        ).reset_index(drop=True)
        sorted_titles.columns = [f"{col}_sorted" for col in sorted_titles.columns]
        random_titles = df_cluster.sample(nr_tiles_adapted, random_state=1887)[
            cols_to_show
        ].reset_index(drop=True)
        random_titles.columns = [f"{col}_random" for col in random_titles.columns]

        # put dfs next to each other
        cluster_title_sheet = pd.concat([sorted_titles, random_titles], axis=1)
        return cluster_title_sheet

    def analyze_text(self, df_cluster):
        self.text_analyzer = TextAnalyzer()
        text_corpus = df_cluster["title_abstract"].values
        # word_quantities = self.text_analyzer.count_word_quantities(text_corpus)
        df_tfidf = self.text_analyzer.tfidf_word_values(text_corpus)
        # df_both = self.text_analyzer.merged_df(word_quantities, df_tfidf).head(
        #    self.nr_words
        # )
        tfids_words = df_tfidf["Word_tfidf"].values
        return tfids_words

    def create_full_explorer(self):
        cluster_titles_sheets_dict = {}
        summary_dict = self.create_summary_dict()

        for cluster in self.df[self.cluster_column].unique():
            df_cluster = self.df[self.df[self.cluster_column] == cluster]

            paper_count = df_cluster.shape[0]
            nr_tiles_adapted = min(self.nr_tiles, paper_count)

            # Get the sorted and random titles
            cluster_title_sheet = self.cluster_titles_sheets_creator(
                df_cluster, nr_tiles_adapted
            )
            # save it to dict.
            cluster_titles_sheets_dict[cluster] = cluster_title_sheet

            # summary dict
            ## Calculate the full value counts per cluster
            summary_dict["Cluster"].append(cluster)
            summary_dict["Nr of Pubs"].append(df_cluster.shape[0])
            summary_dict["Mean Year"].append(round(df_cluster["year"].mean(), 1))
            summary_dict["SD Year"].append(round(df_cluster["year"].std(), 1))
            summary_dict["Given Label"].append("")

            ## text analyser
            tfids_words = self.analyze_text(df_cluster)

            # add to summary dict
            for i, word in enumerate(tfids_words):
                summary_dict[f"Word_{i}"].append(word)

        self.cluster_titles_sheets_dict = cluster_titles_sheets_dict

        self.df_summary = (
            pd.DataFrame(summary_dict)
            .sort_values(by="Nr of Pubs", ascending=False)
            .reset_index(drop=True)
        )

    def full_return(self):
        return self.df_summary, self.cluster_titles_sheets_dict


"""
example usage
# Initialize the class with your DataFrame, cluster column, and other parameters
community_explorer = CommunityExplorer(
    df, "cluster", sort_column="eigenvector_cluster_centrality", nr_tiles=15, nr_words=20)

# Create the full explorer
community_explorer.create_full_explorer()

# Get the summary and cluster titles sheets dictionary
summary, cluster_titles_sheets_dict = community_explorer.full_return()

"""


class FullExplorer:
    def __init__(self, df, cluster_column, params, resolution):
        self.df = df
        self.cluster_column = cluster_column
        self.params = params
        self.resolution = resolution
        self.nr_clusters = df[cluster_column].nunique()
        self.nr_words = 15

    def summary_dict_creator(self):
        summary_dict = {
            "KNN-Params": self.params,
            "Resolution": self.resolution,
            "Nr of Clusters": self.nr_clusters,
        }
        return summary_dict

    def params_dict_creator(self):
        params_cols = [
            "Cluster",
            "Nr of Pubs",
            "Mean Year",
            "SD Year",
        ]
        params_cols.extend([f"Word_{i}" for i in range(self.nr_words)])
        params_dict = {col: [] for col in params_cols}
        return params_dict

    def explorer_sheets_creator(self):
        params_dict = self.params_dict_creator()
        for cluster in range(self.nr_clusters):
            df_cluster = self.df[self.df[self.cluster_column] == cluster]
            params_dict["Cluster"].append(cluster)
            params_dict["Nr of Pubs"].append(df_cluster.shape[0])
            params_dict["Mean Year"].append(round(df_cluster["year"].mean(), 1))
            params_dict["SD Year"].append(round(df_cluster["year"].std(), 1))

            # Now add words
            text_analyzer = TextAnalyzer()  # Make sure TextAnalyzer is defined/imported
            text_corpus = df_cluster["title_abstract"].values
            df_word_quantities = text_analyzer.count_word_quantities(text_corpus)
            #            df_tfidf = text_analyzer.tfidf_word_values(text_corpus)
            words = df_word_quantities["Word_freq"].values[
                : self.nr_words
            ]  # Limit to the first 15 words

            for i in range(15):  # Always iterate 15 times.
                word = (
                    words[i] if i < len(words) else None
                )  # Use None or a placeholder if there are fewer than 15 words.
                params_dict[f"Word_{i}"].append(word)

        params_df = pd.DataFrame(params_dict)
        return params_df
