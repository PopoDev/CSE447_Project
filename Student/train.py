import evaluate
import numpy as np
from transformers import Trainer, AutoTokenizer, AutoConfig
from model.multiple_choice import ModelForMultipleChoice
from arguments import parse_arguments
from dataloader.dataset import get_openbookqa_dataset

def main():
    # Parse arguments
    train_args, model_args = parse_arguments()
    train_args.evaluation_strategy = "epoch"
    train_args.save_strategy = "epoch"
    train_args.save_total_limit = 5
    train_args.load_best_model_at_end = True

    # Load models
    model = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)
    model = ModelForMultipleChoice(model=model, config=config)
    print(model.parameters)

    train_dataset, val_dataset, test_dataset = get_openbookqa_dataset(tokenizer, model_args.use_book)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # change to test dataset for final evaluation
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.evaluate(eval_dataset=val_dataset)
    trainer.predict(test_dataset=test_dataset)
    
    model.save_pretrained(train_args.output_dir)


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metric = accuracy.compute(predictions=predictions, references=labels)
    print("Accuracy:", metric)
    return metric


if __name__ == "__main__":
    main()

