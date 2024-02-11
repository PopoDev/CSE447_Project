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

        return choice_outputs[0][:, 1]

# Example usage
sbert_model = SentenceBERTModel()
model = RobertaPromptForMultipleChoice(sbert_model)

prompt = "Which animal is considered a predator?"
choices = [
    "a) ant",
    "b) snake",
    "c) elephant",
    "d) giraphe",
]

similarity_scores = model(prompt, choices)
print(f"Similarity scores for prompt='{prompt}' and choices={choices}: {similarity_scores}")
