from pathlib import Path

import yaml


REQUIRED_STAGES = {
    "generate_candidates",
    "snap_components",
    "evaluate_fast",
    "train_surrogate",
    "propose_next",
    "simulate_spice",
    "robustness_mc",
    "rank_pareto",
    "report_topk",
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_dvc_yaml_contains_required_stages() -> None:
    data = _load_yaml(Path("dvc.yaml"))
    stages = data.get("stages", {})
    assert REQUIRED_STAGES.issubset(stages.keys())

    for stage in REQUIRED_STAGES:
        stage_cfg = stages[stage]
        assert "cmd" in stage_cfg
        assert "deps" in stage_cfg
        assert "outs" in stage_cfg

    if "repro_smoke" in stages:
        stage_cfg = stages["repro_smoke"]
        assert "cmd" in stage_cfg
        assert "outs" in stage_cfg


def test_params_yaml_contains_stage_configs() -> None:
    data = _load_yaml(Path("params.yaml"))
    for stage in REQUIRED_STAGES:
        assert stage in data
        assert isinstance(data[stage], dict)

    if "repro_smoke" in data:
        assert isinstance(data["repro_smoke"], dict)
