import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, RobertaForMultipleChoice
from model.sentence_similarity import SentenceBERTModel

class RobertaPromptForMultipleChoice(pl.LightningModule):
    def __init__(self, sbert_model):
        super().__init__()
        self.sbert = sbert_model
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForMultipleChoice.from_pretrained("roberta-base")

    def forward(self, question, choices, labels):
        prefix = self.sbert.get_top_similar_sentences(question + ', '.join(choices), top_n=3)
        prompt_prefix = "Given the following facts: " + ', '.join(prefix) + " answer the question. "
        prompt = prompt_prefix + question
        print("Prompt:", prompt)

        encoding = self.tokenizer([prompt]*len(choices), choices, return_tensors="pt", padding=True)
        outputs = self.model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)

        loss = outputs.loss
        logits = outputs.logits
        
        probabilities = torch.softmax(logits, dim=1)
        answer_index = torch.argmax(probabilities).item()
        print('Probabilities:', probabilities)
        print('Answer index:', answer_index)

        return choices[answer_index]

# Example usage
sbert_model = SentenceBERTModel()
model = RobertaPromptForMultipleChoice(sbert_model)

question = "Which animal is considered a predator?"
choices = [
    "a) ant",
    "b) snake",
    "c) elephant",
    "d) giraphe",
]
labels = torch.tensor(1).unsqueeze(0)  # choice b is the correct answer

answer = model(question, choices, labels)
print(f"Answer for question='{question}': {answer}")