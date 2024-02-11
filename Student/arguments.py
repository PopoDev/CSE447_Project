from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class TrainingArguments:
    num_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs for training."}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for training."}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training."}
    )

@dataclass
class ModelArguments:
    data_path_sentences: str = field(
        default='./data/openbook.txt',
        metadata={"help": "Path to the text file containing sentences."}
    )

def parse_arguments():
    parser = HfArgumentParser((TrainingArguments, ModelArguments))
    args = parser.parse_args_into_dataclasses()
    return args
