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

        def replace_punct(text):
            return text.translate(
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

        tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)
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
