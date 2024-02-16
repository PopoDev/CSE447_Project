import torch
from model.multiple_choice import RobertaPromptForMultipleChoice
from model.sentence_similarity import SentenceBERTModel
from arguments import parse_arguments

def main():
    # Parse arguments
    train_args, model_args = parse_arguments()

    # Load models
    sbert_model = SentenceBERTModel(data_path_sentences=model_args.data_path_sentences)
    roberta_model = RobertaPromptForMultipleChoice(sbert_model=sbert_model)

    # Example multiple-choice question
    question = "Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as"
    choices = [
        "A) Deep sea animals",
        "B) Fish",
        "C) Long Sea Fish",
        "D) Far Sea Animals",
    ]
    labels = torch.tensor(0).unsqueeze(0)  # choice A is the correct answer

    answer = roberta_model(question, choices, labels)
    print(f"Answer for question='{question}': {answer}")

if __name__ == "__main__":
    main()
