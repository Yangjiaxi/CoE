"""MoE-LLaMA model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama import LlamaConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MoeLlamaConfig(LlamaConfig):
    r"""
    This config file is for MoE-LLaMA
    """

    model_type = "moe_llama"

    def __init__(
        self,
        n_routed_experts=None,
        num_experts_per_tok=None,
        router_loss_type="ce",
        router_loss_alpha=0.001,
        ordinal_to_expert=None,  # {str(int): str}
        **kwargs,
    ):
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_loss_type = router_loss_type
        self.router_loss_alpha = router_loss_alpha
        if ordinal_to_expert is None:
            self.ordinal_to_expert = {}  # {str(int): str}
        else:
            self.ordinal_to_expert = ordinal_to_expert

        super().__init__(**kwargs)

    def add_expert(self, idx, name):
        self.ordinal_to_expert[str(idx)] = name
