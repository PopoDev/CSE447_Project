from datasets import load_dataset
from dataloader.obqa_dataset import OBQADataset

def load_openbookqa_data(use_book=True):
    if use_book:
        path = "./data/facts/"
        dataset = load_dataset("json", data_files={"train": path + "train.jsonl", 
                                                   "validation": path + "dev.jsonl", 
                                                   "test": path + "test.jsonl"})
    else:
        dataset = load_dataset("openbookqa", "main")
    
    train_data, val_data, test_data = dataset["train"], dataset["validation"], dataset["test"]
    print(f"Train data sample: {train_data[0]}")

    return train_data, val_data, test_data

def get_openbookqa_dataset(tokenizer, use_book=True, n_facts=3):
    train_dataset, val_dataset, test_dataset = map(lambda data: OBQADataset(data, tokenizer, use_book=use_book, n_facts=n_facts), load_openbookqa_data(use_book))
    print(f"Train dataset sample: {train_dataset[0]}")

    return train_dataset, val_dataset, test_dataset
