from typing import Any, Dict, Path, Union

import yaml

from tvarallm.config import Config


class BaseModel:
    def __init__(self, config_path: Union[str, Path]):
        self.config = self.load_config(config_path)
        self.build_model()

    def load_config(self, config_path: Union[str, Path]) -> Config:
        return Config.from_yaml(config_path)

    def build_model(self):
        # Implement model building logic based on self.config
        n_layers = self.config.n_layer
        n_heads = self.config.n_head
        embedding_dim = self.config.n_embd
        vocab_size = self.config.vocab_size

        print(
            f"Building model with {n_layers} layers, {n_heads} heads, "
            f"embedding dimension {embedding_dim}, and vocabulary size {vocab_size}"
        )

        # Here you would typically create your model architecture
        # using the configuration parameters

        raise NotImplementedError("Model building logic needs to be implemented")

    def get_param_count(self) -> int | None:
        # Using the new type union operator
        return getattr(self, "param_count", None)
