"""ハイパーパラメータ設定"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # model
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 128
    dropout: float = 0.1

    # training
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200

    # paths
    data_path: str = "data/input.txt"
    checkpoint_path: str = "checkpoint.pt"

    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
