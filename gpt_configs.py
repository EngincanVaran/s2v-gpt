from dataclasses import dataclass


@dataclass
class GPTConfig:
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001

    vocab_size: int = 4096
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 128
    block_size: int = 32

    dropout: float = 0.1
    bias: bool = True
