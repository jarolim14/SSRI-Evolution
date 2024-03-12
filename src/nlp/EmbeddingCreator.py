import gc
import os
import warnings

import torch
from adapters import AutoAdapterModel
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.simplefilter(action="ignore", category=FutureWarning)


class PaperEmbeddingProcessor:
    def __init__(
        self,
        df,
        model_name,
        adapter_name,
        save_dir,
        title_col="clean_title",
        abstract_col="clean_abstract",
        batch_size=32,
        chunk_size=100,  # Manageable chunk size for memory
    ):
        self.df = df
        self.title_col = title_col
        self.abstract_col = abstract_col
        self.chunk_size = chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.model.load_adapter(
            "allenai/specter2",
            source="hf",
            load_as=adapter_name,
            set_active=True,
        )
        self.save_dir = save_dir
        self.batch_size = batch_size
        os.makedirs(save_dir, exist_ok=True)

    def process_batch(self, batch):
        text_batch = [
            d[self.title_col]
            + self.tokenizer.sep_token
            + (d.get(self.abstract_col) or "")
            for d in batch
        ]
        inputs = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        with torch.no_grad():
            output = self.model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :]
        return embeddings.cpu()

    def process_papers(self):
        all_embeddings = []
        total_records = len(self.df)

        for start_idx in tqdm(range(0, total_records, self.chunk_size)):
            end_idx = min(start_idx + self.chunk_size, total_records)
            batch_embeddings = []

            for batch_start in range(start_idx, end_idx, self.batch_size):
                batch_end = min(batch_start + self.batch_size, end_idx)
                batch = self.df.iloc[batch_start:batch_end].to_dict(orient="records")

                if batch:  # Check if the batch is not empty
                    embeddings = self.process_batch(batch)
                    batch_embeddings.append(embeddings)

                    # Clear memory
                    del embeddings
                    gc.collect()

            # Save and accumulate batch embeddings
            if batch_embeddings:  # Check if there are embeddings to concatenate
                batch_embeddings = torch.cat(batch_embeddings, dim=0)
                batch_file = os.path.join(
                    self.save_dir, f"embeddings_chunk_{start_idx//self.chunk_size}.pt"
                )
                torch.save(batch_embeddings, batch_file)
                all_embeddings.append(batch_embeddings)

        total_embeddings = (
            torch.cat(all_embeddings, dim=0) if all_embeddings else torch.tensor([])
        )
        torch.save(total_embeddings, os.path.join(self.save_dir, "total_embeddings.pt"))
        return total_embeddings

    def save_embeddings_with_data(
        self, embeddings, file_name="df_with_specter2_embeddings.pkl"
    ):
        self.df["specter2_embeddings"] = list(embeddings.numpy())
        self.df.to_pickle(os.path.join(self.save_dir, file_name))
