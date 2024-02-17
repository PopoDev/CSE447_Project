from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
from typing import Optional

@dataclass
class ModelArguments:
    obqa_book_path: Optional[str] = field(
        default='./data/openbook.txt',
        metadata={"help": "Path to the OpenBookQA book data"}
    )

def parse_arguments():
    parser = HfArgumentParser((TrainingArguments, ModelArguments))
    args = parser.parse_args_into_dataclasses()
    return args
