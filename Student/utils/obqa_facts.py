import json
import argparse

def obqa_facts(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            entry = json.loads(line.strip())
            transformed_entry = {
                'id': entry['id'],
                'question': entry['question_stem'],
                'facts': entry['facts'],
                'choices': entry['choices']['text'],
                'answerKey': entry['answerKey']
            }
            data.append(transformed_entry)
        return data    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    
    data = obqa_facts(args.file)

    with open(args.output, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + "\n")