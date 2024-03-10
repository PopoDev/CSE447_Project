import torch
from torch.utils.data import Dataset

class OBQADataset(Dataset):
    def __init__(self, data, tokenizer, use_book=True, n_facts=3, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_facts = n_facts
        self.use_book = use_book

        print(f"Loaded {len(self.data)} samples from OpenBookQA dataset {'with' if use_book else 'without'} facts from the book")

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
        if self.use_book:
            facts = item['facts'][:self.n_facts]
            prompt_prefix = ', '.join(facts) + ". "

        prompt = prompt_prefix + question

        encoding = self.tokenizer([prompt]*len(choices), choices, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        encoding['labels'] = label

        return encoding
    