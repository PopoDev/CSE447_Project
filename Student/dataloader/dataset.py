from datasets import load_dataset
from dataloader.obqa_dataset import OBQADataset

def load_openbookqa_data():
    dataset = load_dataset("openbookqa", "main")
    train_data, val_data, test_data = dataset["train"], dataset["validation"], dataset["test"]
    
    print(f"Train data sample: {train_data[0]}")

    return train_data, val_data, test_data

def get_openbookqa_dataset(tokenizer, sbert_model=None):
    train_dataset, val_dataset, test_dataset = map(lambda data: OBQADataset(data, tokenizer, sbert_model), load_openbookqa_data())
    print(f"Train dataset sample: {train_dataset[0]}")

    return train_dataset, val_dataset, test_dataset
