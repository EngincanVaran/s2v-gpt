from dataclasses import dataclass

@dataclass
class GlobalConfigs:
    exp_num: int

@dataclass
class PredictionConfigs:
    batch_size: int
    top_k: int

@dataclass
class TrainingConfigs:
    batch_size: int
    epochs: int
    eval_interval: int
    learning_rate: float
    milestones: list[int]
    gamma: float

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
    PREDICTION: PredictionConfigs
