import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import util
from model.sentence_similarity import SentenceBERTModel

class RobertaPromptForMultipleChoice(pl.LightningModule):
    def __init__(self, sbert_model):
        super().__init__()
        self.sbert_model = sbert_model
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained("roberta-base")

    def forward(self, prompt, choices):
        # Tokenize prompt and choices
        prompt_input = self.roberta_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        choice_inputs = self.roberta_tokenizer(choices, return_tensors="pt", padding=True, truncation=True)

        # Forward pass through Roberta model
        prompt_output = self.roberta_model(**prompt_input)
        choice_outputs = self.roberta_model(**choice_inputs)

        # Compute embeddings for prompt and choices
        prompt_embedding = prompt_output.last_hidden_state[:, 0, :]
        choice_embeddings = choice_outputs.last_hidden_state[:, 0, :]

        # Compute similarity between prompt and choices
        similarity_scores = util.cos_sim(prompt_embedding, choice_embeddings)

        return similarity_scores

# Example usage
sbert_model = SentenceBERTModel()
model = RobertaPromptForMultipleChoice(sbert_model)

prompt = "Which of the following animals is a mammal?"
choices = [
    "a) Dog",
    "b) Parrot",
    "c) Snake",
    "d) Dolphin",
]

similarity_scores = model(prompt, choices)
print(f"Similarity scores for prompt='{prompt}' and choices={choices}: {similarity_scores}")
