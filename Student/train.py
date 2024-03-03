import torch
import evaluate
import numpy as np
from transformers import Trainer, AutoTokenizer
from model.multiple_choice import RobertaPromptForMultipleChoice
from model.sentence_similarity import SentenceBERTModel
from arguments import parse_arguments
from dataloader.dataset import get_openbookqa_dataset

def main():
    # Parse arguments
    train_args, model_args = parse_arguments()

    # Load models
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    sbert = SentenceBERTModel(path=model_args.obqa_book_path)
    model = RobertaPromptForMultipleChoice()
    print(model.parameters)

    train_dataset, val_dataset, test_dataset = get_openbookqa_dataset(tokenizer, sbert_model=None)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # change to test dataset for final evaluation
        compute_metrics=compute_metrics,
    )
    
    trainer.train()


    trainer.evaluate()
    
    trainer.save_model(output_dir=train_args.output_dir)


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    main()

