import json
import argparse
from tqdm import tqdm
from dataloader.dataset import load_openbookqa_data
from model.sentence_similarity import SentenceBERTModel

def add_openbook_facts(data, top_n=3, debug=False):
    sbert = SentenceBERTModel(path='./data/openbook.txt')
    if debug:
        data = data.select(range(2))
    
    facts_column = []
    
    for item in tqdm(data):
        question = item['question_stem']
        choices = item['choices']['text']
        sentence = question + ": " + ", ".join(choices)
        facts = sbert.get_top_similar_sentences(sentence, top_n=top_n)
        facts_column.append(facts)
    
    return data.add_column(name="facts", column=facts_column)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--top_n", type=int, default=3)
    args = parser.parse_args()

    train_data, val_data, test_data = load_openbookqa_data()
    top_n = args.top_n

    with open('./data/train.facts.jsonl', 'w') as file:
        data = add_openbook_facts(train_data, top_n=top_n, debug=args.debug)
        for item in data:
            file.write(json.dumps(item) + "\n")
        print(f"Saved {len(train_data)}*{top_n} facts to ./data/train.facts.jsonl")

    with open('./data/dev.facts.jsonl', 'w') as file:
        data = add_openbook_facts(val_data, top_n=top_n, debug=args.debug)
        for item in data:
            file.write(json.dumps(item) + "\n")
        print(f"Saved {len(val_data)}*{top_n} facts to ./data/dev.facts.jsonl")

    with open('./data/test.facts.jsonl', 'w') as file:
        data = add_openbook_facts(test_data, top_n=top_n, debug=args.debug)
        for item in data:
            file.write(json.dumps(item) + "\n")
        print(f"Saved {len(test_data)}*{top_n} facts to ./data/test.facts.jsonl")