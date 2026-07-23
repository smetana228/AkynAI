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


def test_supported_kwargs_filters_by_signature():
    from kyrpoet.train.common import supported_kwargs

    def old_api(output_dir=None, max_seq_length=None):
        pass

    def new_api(output_dir=None, max_length=None):
        pass

    desired = {"output_dir": "x", "max_length": 512, "max_seq_length": 512}
    assert supported_kwargs(old_api, desired) == {"output_dir": "x", "max_seq_length": 512}
    assert supported_kwargs(new_api, desired) == {"output_dir": "x", "max_length": 512}

    def takes_kwargs(**kw):
        pass

    assert supported_kwargs(takes_kwargs, desired) == desired  # can't introspect


def test_pick_kwarg_prefers_newer_name():
    from kyrpoet.train.common import pick_kwarg

    def new_api(max_length=None):
        pass

    def old_api(max_seq_length=None):
        pass

    def neither(x=None):
        pass

    assert pick_kwarg(new_api, ["max_length", "max_seq_length"], 1024) == {"max_length": 1024}
    assert pick_kwarg(old_api, ["max_length", "max_seq_length"], 1024) == {"max_seq_length": 1024}
    assert pick_kwarg(neither, ["max_length", "max_seq_length"], 1024) == {}


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
