import torch
from torch.utils.data import Dataset

class OBQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, sbert_model=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sbert = sbert_model

    def get_label_tensor(self, label: str):
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        return torch.tensor(mapping[label])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question_stem']
        choices = item['choices']['text']
        label = self.get_label_tensor(item['answerKey'])

        prompt_prefix = ""
        if self.sbert:
            clues = self.sbert.get_top_similar_sentences(question + ', '.join(choices), top_n=3)
            prompt_prefix = "Given the following facts: " + ', '.join(clues) + ". "

        prompt = prompt_prefix + question

        encoding = self.tokenizer([prompt]*len(choices), choices, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        encoding['labels'] = label

        return encoding
    