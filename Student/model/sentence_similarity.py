import torch
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer, util


class SentenceBERTModel(pl.LightningModule):
    def __init__(self, data_path_sentences):
        super().__init__()
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        with open(data_path_sentences, 'r') as file:
            self.sentences = [line.strip().strip('"') for line in file]

        print(f"Loaded {len(self.sentences)} sentences from {data_path_sentences}")
        print(f"Example sentences: {self.sentences[:3]}")

    def forward(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)

    def get_top_similar_sentences(self, compared_sentence, top_n=3):
        # Compute embeddings
        compared_embedding = self.forward([compared_sentence])
        embeddings = self.forward(self.sentences)

        # Compute cosine-similarities between the compared sentence and all other sentences
        cosine_scores = util.cos_sim(compared_embedding, embeddings)

        # Sort the scores
        sorted_indices = torch.argsort(cosine_scores, descending=True)

        # Get the top-n most similar sentences
        return [self.sentences[idx] for idx in sorted_indices[:top_n]]


# Example usage
model = SentenceBERTModel()
sentences = [
    "This is the first sentence.",
    "This is the second sentence.",
    "This is the third sentence.",
    "Another unrelated sentence.",
]
compared_embedding = 'This is the inital sentence'

similar_sentences = model.get_top_similar_sentences(compared_embedding, top_n=3)
print(f"Most similar sentences to '{compared_embedding}': {similar_sentences}")