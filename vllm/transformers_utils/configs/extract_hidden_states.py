# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config definitions for ExtractHiddenStatesModel, to be used with
the extract_hidden_states spec decoding method."""

import os

from transformers import PretrainedConfig

from vllm.transformers_utils.utils import without_trust_remote_code


class ExtractHiddenStatesConfig(PretrainedConfig):
    model_type = "extract_hidden_states"

    def __init__(
        self,
        model: PretrainedConfig | dict | None = None,
        method: str | None = "extract_hidden_states",
        **kwargs,
    ):
        assert method == "extract_hidden_states"

        if isinstance(model, dict):
            model_dict = model
        elif isinstance(model, PretrainedConfig):
            model_dict = model.to_dict()
        else:
            model_dict = {}

        # Combine: model_dict first, then kwargs override
        combined = {**model_dict, **kwargs}
        # Remove architectures from the base, we'll set it explicitly
        combined = {k: v for k, v in combined.items() if k != "architectures"}

        combined["architectures"] = ["ExtractHiddenStatesModel"]

        super().__init__(**combined)

    def get_text_config(self, decoder: bool = False):
        text_config = getattr(self, "text_config", None)
        if isinstance(text_config, PretrainedConfig):
            return text_config

        # Some multimodal configs arrive here through to_dict(), which turns
        # the nested text config into a plain dict. In that case the top-level
        # config still carries the language-model shape attributes we need.
        if isinstance(text_config, dict) and hasattr(self, "num_attention_heads"):
            return self

        return super().get_text_config(decoder=decoder)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "ExtractHiddenStatesConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **without_trust_remote_code(kwargs)
        )
        return cls.from_dict(config_dict, **kwargs)

    def to_json_string(self, use_diff: bool = True) -> str:
        # we override use_diff to False as initializing
        # ExtractHiddenStatesConfig with default arguments is not supported
        del use_diff
        return super().to_json_string(use_diff=False)
