from model.multiple_choice import RobertaPromptForMultipleChoice
from model.sentence_similarity import SentenceBERTModel
from arguments import parse_arguments

def main():
    # Parse arguments
    args = parse_arguments()

    # Load models
    sbert_model = SentenceBERTModel(data_path_sentences=args.data_path_sentences)
    roberta_model = RobertaPromptForMultipleChoice(sbert_model=sbert_model)

    # Example multiple-choice question
    question = "Which of the following animals is a mammal?"
    choices = [
        "a) Dog",
        "b) Parrot",
        "c) Snake",
        "d) Dolphin",
    ]

    # Get similarity scores from Roberta model
    answer = roberta_model(question, choices)
    print(f"Answer for question='{question}': {answer}")

if __name__ == "__main__":
    main()
