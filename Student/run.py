import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# Load models
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

print(model.parameters)
print(model.config.to_dict())

# Example multiple-choice question
question = "The sun is responsible for"
choices = [
    "A) Deep sea animals",
    "B) Fish",
    "C) Long Sea Fish",
    "D) Far Sea Animals",
]
clue = "The correct answer is (D) plants sprouting, blooming and wilting. The sun is the primary source of energy for life on Earth, and it plays a crucial role in the growth and development of plants. Through the process of photosynthesis, plants use energy from the sun to convert carbon dioxide and water into glucose, which they use for growth and development. This process also results in the release of oxygen, which is essential for the survival of many organisms. As plants grow, they undergo various stages, including sprouting, blooming, and wilting, all of which are influenced by the sun."

prompts = []

for choice in choices:
    prompt = question + ": " + choice + ". " + clue
    print(prompt)
    prompts.append(prompt)

    
encoding = tokenizer(prompts, return_tensors="pt", padding='max_length', truncation=True, max_length=256)
correct = 3
encoding['input_ids'] = encoding['input_ids'].unsqueeze(0)
encoding['labels'] = torch.tensor(correct).unsqueeze(0)

print(encoding)
answer = torch.argmax(model(**encoding).logits)
print(f"Answer for question='{question}': {answer}")
