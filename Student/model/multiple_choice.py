from transformers import RobertaForMultipleChoice, RobertaPreTrainedModel

class RobertaPromptForMultipleChoice(RobertaPreTrainedModel):
    def __init__(self):
        super().__init__()
        self.model = RobertaForMultipleChoice.from_pretrained("roberta-base")

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    