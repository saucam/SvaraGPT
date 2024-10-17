from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from typing_extensions import Self


@dataclass
class Config:
    name: str = ""
    hf_config: Dict[str, Any] = field(default_factory=dict)
    scale_embeddings: bool = False
    attention_scores_scalar: Optional[int] = None
    block_size: int = 4096
    sliding_window_size: Optional[int] = None
    sliding_window_layer_placing: Optional[str] = None
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    head_size: Optional[int] = None
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    norm_class_name: str = "LayerNorm"
    post_attention_norm: bool = False
    post_mlp_norm: bool = False
    norm_eps: float = 1e-5
    mlp_class_name: str = "GptNeoxMLP"
    gelu_approximate: str = "none"
    intermediate_size: Optional[int] = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    rope_adjustments: Optional[Dict[str, Any]] = None
    n_expert: int = 0
    n_expert_per_token: int = 0
    attention_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> Self:
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_file: Union[str, Path]) -> None:
        with open(yaml_file, "w") as f:
            yaml.dump(self.__dict__, f)

    def __post_init__(self):
        if self.head_size is None:
            self.head_size = self.n_embd // self.n_head

        if self.padded_vocab_size is None:
            self.padded_vocab_size = (
                (self.vocab_size + self.padding_multiple - 1)
                // self.padding_multiple
                * self.padding_multiple
            )

        if self.n_query_groups is None:
            self.n_query_groups = self.n_head

        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.n_embd

        if self.sliding_window_size is not None:
            self.sliding_window_layer_placing = (
                1 if self.sliding_window_layer_placing in (None, "all") else 2
            )
