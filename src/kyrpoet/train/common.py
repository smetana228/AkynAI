"""Shared training helpers (config + QLoRA setup).

Config loading is pure/testable; model construction requires the ``train`` extra
(torch, transformers, peft, bitsandbytes) and a CUDA GPU, so it is imported
lazily. Prefer Unsloth where it supports the chosen base (§7); this fallback
path uses TRL + PEFT + bitsandbytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    base_model: str
    output_dir: str
    init_from: str | None = None
    load_in_4bit: bool = True
    lora: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    dpo: dict = field(default_factory=dict)
    preference: dict = field(default_factory=dict)
    seed: int = 42


def supported_kwargs(fn, desired: dict) -> dict:
    """Keep only the kwargs the installed library version actually accepts.

    TRL and transformers rename trainer arguments between releases (e.g.
    ``max_seq_length`` -> ``max_length``, ``tokenizer`` -> ``processing_class``).
    Filtering by the real signature keeps this working across versions instead
    of pinning one exact release.
    """
    import inspect

    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(desired)  # takes **kwargs; can't introspect, pass it all
    return {k: v for k, v in desired.items() if k in params}


def pick_kwarg(fn, names: list[str], value) -> dict:
    """Return {first supported name: value}, or {} if none are supported.

    Use for arguments that were renamed — pass the preferred (newer) name first.
    """
    import inspect

    params = inspect.signature(fn).parameters
    for name in names:
        if name in params:
            return {name: value}
    return {}


def load_config(path: str) -> TrainConfig:
    """Parse a configs/*.yaml file into a TrainConfig."""
    import yaml  # part of the `train` extra

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    known = TrainConfig.__dataclass_fields__.keys()
    return TrainConfig(**{k: v for k, v in raw.items() if k in known})


def build_qlora_model(cfg: TrainConfig):  # pragma: no cover - GPU + heavy deps
    """Load a 4-bit base model + LoRA adapters per config."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=bnb, device_map="auto"
    )
    if cfg.init_from:  # continue from a prior adapter (e.g. SFT from CPT)
        model = PeftModel.from_pretrained(model, cfg.init_from, is_trainable=True)
    else:
        lora = LoraConfig(
            r=cfg.lora.get("r", 32),
            lora_alpha=cfg.lora.get("alpha", 64),
            lora_dropout=cfg.lora.get("dropout", 0.05),
            target_modules=cfg.lora.get("target_modules"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    return model, tok
