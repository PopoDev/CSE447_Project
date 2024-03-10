from transformers import AutoModelForMultipleChoice, PreTrainedModel

class ModelForMultipleChoice(PreTrainedModel):
    def __init__(self, model, config):
        super().__init__(config)
        self.model = AutoModelForMultipleChoice.from_pretrained(model)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    