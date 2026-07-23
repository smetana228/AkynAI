import pytest

yaml = pytest.importorskip("yaml")  # part of the `train` extra

from kyrpoet.train.common import TrainConfig, load_config


@pytest.mark.parametrize("path,epochs_key", [
    ("configs/cpt.yaml", "train"),
    ("configs/sft.yaml", "train"),
    ("configs/dpo.yaml", "dpo"),
])
def test_configs_load(path, epochs_key):
    cfg = load_config(path)
    assert isinstance(cfg, TrainConfig)
    assert cfg.base_model
    assert cfg.output_dir.startswith("checkpoints/")
    assert cfg.lora["r"] > 0


def test_sft_defaults_to_base_model():
    # unset so an SFT-only run works out of the box; chain off CPT via --init-from
    assert load_config("configs/sft.yaml").init_from is None


def test_resolve_init_from():
    from kyrpoet.train.sft import resolve_init_from

    # no override -> keep whatever the config says
    assert resolve_init_from("checkpoints/cpt", None) == "checkpoints/cpt"
    assert resolve_init_from(None, None) is None
    # explicit checkpoint wins
    assert resolve_init_from(None, "checkpoints/cpt") == "checkpoints/cpt"
    # 'base'/'none' force training from the base model
    assert resolve_init_from("checkpoints/cpt", "base") is None
    assert resolve_init_from("checkpoints/cpt", "NONE") is None


def test_dpo_has_preference_settings():
    cfg = load_config("configs/dpo.yaml")
    assert cfg.dpo["beta"] > 0
    assert cfg.preference["min_score_gap"] > 0
