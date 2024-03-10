import json
import argparse
from tqdm import tqdm
from dataloader.dataset import load_openbookqa_data
from model.sentence_similarity import SentenceBERTModel

def add_openbook_facts(data, debug=False):
    sbert = SentenceBERTModel(path='./data/openbook.txt')

    max_iters = 10 if debug else len(data)

    for item in tqdm(data[:max_iters]):
        question = item['question_stem']
        choices = item['choices']['text']
        sentence = question + ": " + ", ".join(choices)
        facts = sbert.get_top_similar_sentences(sentence, top_n=3)
        item['facts'] = facts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False)
    args = parser.parse_args()

    train_data, val_data, test_data = load_openbookqa_data()

    with open('./data/train.facts.jsonl', 'w') as file:
        add_openbook_facts(train_data, debug=args.debug)
        json.dump(train_data, file)

    with open('./data/dev.facts.jsonl', 'w') as file:
        add_openbook_facts(val_data, debug=args.debug)
        json.dump(val_data, file)

    with open('./data/test.facts.jsonl', 'w') as file:
        add_openbook_facts(test_data, debug=args.debug)
        json.dump(test_data, file)