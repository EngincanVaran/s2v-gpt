from dataclasses import dataclass

@dataclass
class GlobalConfigs:
    exp_num: int

@dataclass
class TrainingConfigs:
    batch_size: int
    max_iters: int
    eval_interval: int
    learning_rate: float

@dataclass
class ModelConfigs:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float
    bias: bool

@dataclass
class Config:
    GLOBAL: GlobalConfigs
    TRAINING: TrainingConfigs
    MODEL: ModelConfigs