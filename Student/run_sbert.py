import json
from tqdm import tqdm
from dataloader.dataset import load_openbookqa_data
from model.sentence_similarity import SentenceBERTModel

def add_openbook_facts(data):
    sbert = SentenceBERTModel(path='./data/openbook.txt')

    for i, item in enumerate(tqdm(data)):
        question = item['question_stem']
        choices = item['choices']['text']
        sentence = question + ": " + ", ".join(choices)
        facts = sbert.get_top_similar_sentences(sentence, top_n=3)
        item['facts'] = facts

if __name__ == "__main__":
    train_data, val_data, test_data = load_openbookqa_data()

    with open('./data/train_facts.jsonl', 'w') as file:
        add_openbook_facts(train_data)
        json.dump(train_data, file)

    with open('./data/dev_facts.jsonl', 'w') as file:
        add_openbook_facts(val_data)
        json.dump(val_data, file)

    with open('./data/test_facts.jsonl', 'w') as file:
        add_openbook_facts(test_data)
        json.dump(test_data, file)