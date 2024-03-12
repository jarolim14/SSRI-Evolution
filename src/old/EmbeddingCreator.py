import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class EmbeddingCreator:
    def __init__(self, df, modelpath, batch_size=16):
        """
        Initializes the TextProcessor with a DataFrame, model path, and batch size.

        Parameters:
        df (pd.DataFrame): DataFrame containing the text data to be processed.
        modelpath (str): Path to the pretrained model.
        batch_size (int): Size of each batch for processing. Default is 16.
        """
        self.df = df.copy()
        # sort by year
        self.df.sort_values("year", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.modelpath = modelpath
        self.model_short = self.extract_model_short_name(modelpath)
        self.tokenizer, self.model = self.initialize_tokenizer_and_model(modelpath)
        self.batch_size = batch_size
        print(f"Model: {self.modelpath}")
        print(f"Df Shape: {self.df.shape}")

    def extract_model_short_name(self, modelpath):
        """
        Extracts a short model name from the provided model path.

        Parameters:
        modelpath (str): Full path to the model.

        Returns:
        str: Extracted short model name.
        """
        try:
            return re.search(r"/([a-zA-Z]*)[^a-zA-Z]", modelpath).group(1)
        except:
            return modelpath.split("/")[-1]

    def initialize_tokenizer_and_model(self, modelpath):
        """
        Initializes the tokenizer and model from the given path.

        Parameters:
        modelpath (str): Path to the pretrained model.

        Returns:
        tuple: A tuple containing the tokenizer and the model.
        """
        print("Using autotokenizer and automodel")
        tokenizer = AutoTokenizer.from_pretrained(modelpath)
        model = AutoModel.from_pretrained(modelpath)
        return tokenizer, model

    def model_output_in_batches(self, encoded_inputs):
        """
        Processes the text in batches and returns the model outputs and attention masks.

        Parameters:
        encoded_inputs (dict): Encoded inputs for the model.

        Returns:
        tuple: A tuple containing the model output and attention mask.
        """
        input_ids_batches = torch.split(encoded_inputs["input_ids"], self.batch_size)
        attention_mask_batches = torch.split(
            encoded_inputs["attention_mask"], self.batch_size
        )
        model_outputs, attention_masks = [], []

        for input_ids_batch, attention_mask_batch in tqdm(
            zip(input_ids_batches, attention_mask_batches),
            total=len(input_ids_batches),
            desc="Processing batches",
        ):
            input_ids_batch = input_ids_batch.to(self.model.device)
            attention_mask_batch = attention_mask_batch.to(self.model.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids_batch, attention_mask=attention_mask_batch
                )
            model_outputs.append(outputs.last_hidden_state)
            attention_masks.append(attention_mask_batch)

        model_output = torch.cat(model_outputs)
        attention_mask = torch.cat(attention_masks)
        return model_output, attention_mask

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Averages the token embeddings, weighted by the attention mask.

        Parameters:
        model_output (torch.Tensor): The output from the model.
        attention_mask (torch.Tensor): The attention mask.

        Returns:
        torch.Tensor: Mean pooled embeddings.
        """
        token_embeddings = model_output
        attention_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        weighted_sum = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
        attention_sum = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        mean_embeddings = weighted_sum / attention_sum
        return mean_embeddings

    def split_dataframe(self, chunk_size=5000):
        """
        Splits the DataFrame into smaller chunks.

        Parameters:
        chunk_size (int): Size of each chunk. Default is 5000.

        Returns:
        list: List of DataFrame chunks.
        """
        num_chunks = len(self.df) // chunk_size + (len(self.df) % chunk_size > 0)
        print(
            f"Dataframe sorted by year and split into {num_chunks} chunks of size {chunk_size}"
        )
        return [
            self.df[i * chunk_size : (i + 1) * chunk_size].copy()
            for i in tqdm(range(num_chunks))
        ]

    def create_embeddings(
        self,
        text_column_name,
        embeddings_column_name,
        save_directory,
        return_df=False,
        start_chunk=0,
        chunk_size=2500,
        max_length=512,
    ):
        """
        Processes the DataFrame in chunks, tokenizes the text, computes embeddings,
        and saves the processed chunks as pickle files.

        Parameters:
        column_name (str): The column name which contains the text to be processed.
        save_directory (str): Directory to save the processed chunks.
        start_chunk (int): The chunk number to start processing from. Default is 0.
        chunk_size (int): Size of each chunk. Default is 200.
        max_length (int): Maximum length of tokens for the tokenizer. Default is 512.
        """
        chunks = self.split_dataframe(chunk_size=chunk_size)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        embeddings_chunks = []

        for i, chunk in enumerate(chunks[start_chunk:], start=start_chunk):
            text_list = (
                chunk[text_column_name].apply(lambda x: "" if x is None else x).tolist()
            )
            encoded_input = self.tokenizer(
                text_list,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            model_output, attention_mask = self.model_output_in_batches(encoded_input)
            mean_pooled_embeddings = self.mean_pooling(model_output, attention_mask)
            # Use .loc for assignment to avoid SettingWithCopyWarning
            chunk.loc[:, embeddings_column_name] = mean_pooled_embeddings.tolist()

            # Create a filename based on chunk number and year range
            year_range = f"{chunk['year'].iloc[0]}-{chunk['year'].iloc[-1]}"
            filename = os.path.join(save_directory, f"{i}_{year_range}.pkl")
            chunk.to_pickle(filename)
            print(f"Saved chunk {i} to pickle: {filename}")
            embeddings_chunks.append(chunk)

        if return_df:
            return pd.concat(embeddings_chunks)

    if __name__ == "__main__":
        print("This is a module. Please import it.")
        pass
