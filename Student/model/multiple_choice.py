import torch
from transformers import RobertaForMultipleChoice

class RobertaPromptForMultipleChoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RobertaForMultipleChoice.from_pretrained("roberta-base")

    def forward(self, input_ids, attention_mask, labels):
        batch_size = input_ids.shape[0]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels.view(batch_size, 1))
        return outputs
    