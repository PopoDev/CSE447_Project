from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
from typing import Optional


@dataclass
class ModelArguments:
    model: Optional[str] = field(
        default="microsoft/deberta-v3-base",
        choices=[
            "microsoft/deberta-v3-base",
            "microsoft/deberta-v3-large",
            "roberta-base",
            "roberta-large",
        ],
        metadata={"help": "The model to use"},
    )

    obqa_book_path: Optional[str] = field(
        default="./data/openbook.txt",
        metadata={"help": "Path to the OpenBookQA book data"},
    )

    use_book: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use the OpenBookQA book data"}
    )

    n_facts: Optional[int] = field(
        default=3, metadata={"help": "Number of facts to use from the book data"}
    )


def parse_arguments():
    parser = HfArgumentParser((TrainingArguments, ModelArguments))
    args = parser.parse_args_into_dataclasses()
    return args
