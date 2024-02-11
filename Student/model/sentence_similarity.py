import torch
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer, util


class SentenceBERTModel(pl.LightningModule):
    def __init__(self, data_path_sentences="./data/openbook.txt"):
        super().__init__()
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        prefix = "SentenceBERT"

        with open(data_path_sentences, "r") as file:
            self.sentences = [line.strip().strip('"') for line in file]

        print(f"{prefix} Loaded {len(self.sentences)} sentences from {data_path_sentences}")
        print(f"{prefix} Example sentences: {self.sentences[:3]}")

    def forward(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)

    def get_top_similar_sentences(self, compared_sentence, top_n=3):
        # Compute embeddings
        compared_embedding = self.forward([compared_sentence])
        embeddings = self.forward(self.sentences)

        # Compute cosine-similarities between the compared sentence and all other sentences
        cosine_scores = util.cos_sim(compared_embedding, embeddings)
        similarity_scores = cosine_scores.reshape(-1)

        # Sort the scores
        sorted_scores = torch.argsort(similarity_scores, descending=True)

        # Get the top-n most similar sentences
        return [self.sentences[idx] for idx in sorted_scores[:top_n]]


# Example usage
model = SentenceBERTModel()
compared_embedding = "A dog is an animal"

similar_sentences = model.get_top_similar_sentences(compared_embedding, top_n=3)
print(f"Most similar sentences to '{compared_embedding}': {similar_sentences}")
