import torch
from transformers import AutoTokenizer, RobertaForMultipleChoice

class RobertaPromptForMultipleChoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForMultipleChoice.from_pretrained("roberta-base")

    def forward(self, inputs, labels):
        outputs = self.model(**inputs, labels=labels)
        return outputs
    