from transformers import Trainer
from model.multiple_choice import RobertaPromptForMultipleChoice
from model.sentence_similarity import SentenceBERTModel
from arguments import parse_arguments
from dataloader.dataset import get_openbookqa_dataset

def main():
    # Parse arguments
    train_args, model_args = parse_arguments()

    # Load models
    sbert = SentenceBERTModel(path=model_args.obqa_book_path)
    model = RobertaPromptForMultipleChoice()
    print(model.parameters)

    train_dataset, val_dataset, test_dataset = get_openbookqa_dataset(sbert_model=None)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # change to test dataset for final evaluation
    )
    
    trainer.train()

if __name__ == "__main__":
    main()

