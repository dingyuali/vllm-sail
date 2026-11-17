# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_DIR = ROOT / "scripts" / "ci"
RECIPES_DIR = ROOT / "tests" / "ppu" / "recipes"
POLICY_DIR = ROOT / "tests" / "ppu" / "policy"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ppu_recipe = _load_module("ppu_recipe", CI_DIR / "ppu_recipe.py")
ppu_recipe_runner = _load_module("ppu_recipe_runner", CI_DIR / "ppu_recipe_runner.py")
ppu_e2e_select = _load_module("ppu_e2e_select", CI_DIR / "ppu_e2e_select.py")


def _sample_vllm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm serve")
    parser.add_argument("model_tag")
    parser.add_argument("--served-model-name")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--data-parallel-size", type=int)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--all2all-backend")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--distributed-timeout-seconds", type=int)
    parser.add_argument("--cpu-distributed-timeout-seconds", type=int)
    parser.add_argument("--enable-auto-tool-choice", action="store_true")
    parser.add_argument("--tool-call-parser")
    parser.add_argument("--tokenizer-mode")
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--reasoning-config")
    parser.add_argument("--speculative-config")
    parser.add_argument("--compilation-config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--no-enable-prefix-caching", action="store_true")
    parser.add_argument("--load-format")
    parser.add_argument("--hf-overrides")
    return parser


def _yaml_files() -> list[Path]:
    return sorted(
        p
        for p in (ROOT / "tests" / "ppu").rglob("*.yaml")
        if not p.name.startswith(".")
    )


def test_checked_in_recipes_and_policies_validate_cleanly() -> None:
    files = _yaml_files()
    assert len(files) == 5
    report = ppu_recipe.validate_paths(
        files,
        vllm_parser_mode="required",
        parser=_sample_vllm_parser(),
    )
    assert report["parser_checked"] is True
    assert report["checked_flag_sets"] >= 10


def test_resolve_glm53_matches_catalog_and_supports_presets() -> None:
    recipe = RECIPES_DIR / "zai-org" / "GLM-5.3.yaml"
    policy = POLICY_DIR / "models" / "glm-5.3.yaml"
    catalog = {
        entry["key"]: entry
        for entry in json.loads((CI_DIR / "ppu_e2e_models.json").read_text())
    }
    cat_flags = catalog["glm-5.3-mxfp4-tp4"]["server_cmd"][9:]

    resolved_ms = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_ms["ppu_count"] == 4
    assert resolved_ms["runtime_env"] == {"VLLM_SERVER_DEV_MODE": "1"}
    assert resolved_ms["flags"][:-1] == [
        "--trust-remote-code",
        "--tensor-parallel-size",
        "4",
        "--max-model-len",
        "16384",
        "--distributed-timeout-seconds",
        "3600",
        "--cpu-distributed-timeout-seconds",
        "3600",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm45",
    ]
    assert resolved_ms["flags"][-1] == "--enable-prefix-caching"
    assert set(cat_flags) <= set(resolved_ms["flags"])

    resolved_fs = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity"
    )
    assert "--load-format" in resolved_fs["flags"]
    assert "dummy" in resolved_fs["flags"]
    idx_ov = resolved_fs["flags"].index("--hf-overrides")
    assert json.loads(resolved_fs["flags"][idx_ov + 1]) == {"num_hidden_layers": 2}

    resolved_fs_full = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity", layer_override=None
    )
    assert "--hf-overrides" not in resolved_fs_full["flags"]
    assert resolved_fs_full["config_digest"] != resolved_fs["config_digest"]

    resolved_tp8 = ppu_recipe.resolve_recipe(recipe, preset="tp8", cache="enabled")
    assert resolved_tp8["ppu_count"] == 8
    assert "--tensor-parallel-size" in resolved_tp8["flags"]
    idx_tp = resolved_tp8["flags"].index("--tensor-parallel-size")
    assert resolved_tp8["flags"][idx_tp + 1] == "8"


def test_resolve_dsv4_matches_catalog_and_dpep_preset() -> None:
    recipe = RECIPES_DIR / "deepseek-ai" / "DeepSeek-V4-Flash.yaml"
    policy = POLICY_DIR / "models" / "deepseek-v4-flash.yaml"
    resolved_default = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="model_sanity"
    )
    assert resolved_default["ppu_count"] == 8
    assert resolved_default["flags"] == [
        "--tensor-parallel-size",
        "8",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "16384",
        "--trust-remote-code",
        "--seed",
        "0",
        "--enable-prefix-caching",
    ]

    resolved_dpep = ppu_recipe.resolve_recipe(
        recipe, preset="dp_ep_low_latency", cache="disabled"
    )
    assert resolved_dpep["ppu_count"] == 8
    assert resolved_dpep["server_env"] == {"VLLM_ENGINE_READY_TIMEOUT_S": "6000"}
    assert "--tensor-parallel-size" not in resolved_dpep["flags"]
    assert "--seed" not in resolved_dpep["flags"]
    assert "--data-parallel-size" in resolved_dpep["flags"]
    assert "--no-enable-prefix-caching" in resolved_dpep["flags"]
    assert "--enable-prefix-caching" not in resolved_dpep["flags"]


def test_config_digest_ignores_ephemeral_endpoint_and_tracks_changes() -> None:
    recipe = RECIPES_DIR / "zai-org" / "GLM-5.3.yaml"
    policy = POLICY_DIR / "models" / "glm-5.3.yaml"
    r1 = ppu_recipe.resolve_recipe(
        recipe,
        policy_path=policy,
        tier="model_sanity",
        port=9985,
        served_model_name="a",
    )
    r2 = ppu_recipe.resolve_recipe(
        recipe,
        policy_path=policy,
        tier="model_sanity",
        port=9999,
        served_model_name="b",
    )
    r3 = ppu_recipe.resolve_recipe(
        recipe, policy_path=policy, tier="fast_sanity", port=9985, served_model_name="a"
    )
    assert r1["config_digest"] == r2["config_digest"]
    assert r1["config_digest"] != r3["config_digest"]


def test_validate_rejects_unknown_keys_bad_env_json_and_leaks(tmp_path: Path) -> None:
    base_recipe = yaml.safe_load((RECIPES_DIR / "zai-org" / "GLM-5.3.yaml").read_text())

    def _check_bad_recipe(mutator, match: str) -> None:
        bad = copy.deepcopy(base_recipe)
        mutator(bad)
        path = tmp_path / "bad_recipe.yaml"
        path.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ppu_recipe.ValidationError, match=match):
            ppu_recipe.validate_paths([path], vllm_parser_mode="skip")

    _check_bad_recipe(lambda d: d.update({"unknown_top": 1}), "unknown key")
    _check_bad_recipe(
        lambda d: d["ppu"].update({"visibility": "internal_restricted"}),
        "internal_restricted",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_env"].update({"HF_HUB_OFFLINE": "1"}),
        "runtime-mode env",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_env"].update({"CUSTOM_VAR": "1"}),
        "allowed prefixes",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_env"].update({"DG_USE_MOE_DYNAMIC_TILE": "1"}),
        "ppu.external_env",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_args"].extend(["--hf-overrides", "not-json"]),
        "invalid JSON",
    )
    _check_bad_recipe(
        lambda d: d["model"]["base_args"].extend(["--hf-overrides", '{"a": "$(id)"}']),
        "shell substitution",
    )
    _check_bad_recipe(
        lambda d: d.update({"notes": "/nas_aisw/datasets/secret"}),
        "local_path",
    )
    _check_bad_recipe(
        lambda d: d.update({"notes": "host.eng.t-head.cn"}),
        "internal_domain",
    )
    _check_bad_recipe(
        lambda d: d.update({"notes": "10.1.2.3"}),
        "ipv4_literal",
    )
    prohibited = "".join(chr(c) for c in (102, 108, 121, 116, 105, 103, 101, 114))
    _check_bad_recipe(
        lambda d: d.update({"notes": prohibited}),
        "prohibited_token",
    )


def test_validate_policy_guardrails_and_redaction_gate(tmp_path: Path) -> None:
    base_pol = yaml.safe_load((POLICY_DIR / "models" / "glm-5.3.yaml").read_text())

    def _check_bad_pol(mutator, match: str) -> None:
        bad = copy.deepcopy(base_pol)
        mutator(bad)
        path = tmp_path / "bad_pol.yaml"
        path.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ppu_recipe.ValidationError, match=match):
            ppu_recipe.validate_paths([path], vllm_parser_mode="skip")

    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][1].update({"cache": "disabled"}),
        "requires cache='enabled'",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][2].__delitem__("cache_mode"),
        "requires cache_mode",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][2]["input"].update(
            {"range_ratio": 1.0}
        ),
        "range_ratio must be in",
    )
    _check_bad_pol(
        lambda d: d["tiers"]["model_sanity"]["stages"][1].update(
            {"dataset": "missing@2026-11"}
        ),
        "referenced dataset definition not found",
    )

    with pytest.raises(ppu_recipe.ValidationError, match="redaction gate blocked"):
        ppu_recipe.redact_and_verify_public_payload({"err": "token ghp_1234567890ab"})
    with pytest.raises(ppu_recipe.ValidationError, match="redaction gate blocked"):
        ppu_recipe.redact_and_verify_public_payload({"path": "/nas_aisw/datasets/foo"})
    ppu_recipe.redact_and_verify_public_payload(
        {"url": "http://127.0.0.1:9985"}, allow_loopback=True
    )


def test_validate_flags_with_parser_rejects_bad_flag_and_type() -> None:
    parser = _sample_vllm_parser()
    with pytest.raises(ppu_recipe.ValidationError, match="unknown vLLM serve flag"):
        ppu_recipe.validate_flags_with_parser(
            parser, ["--nonexistent-flag"], context="test"
        )
    with pytest.raises(ppu_recipe.ValidationError, match="parser rejected args"):
        ppu_recipe.validate_flags_with_parser(
            parser, ["--tensor-parallel-size", "not-an-int"], context="test"
        )


def test_recipe_selector_matrix_and_cli() -> None:
    matrix = ppu_e2e_select.build_recipe_matrix(
        "glm-5.3,deepseek-v4-flash", tier="fast_sanity"
    )
    assert [item["key"] for item in matrix["include"]] == [
        "glm-5.3",
        "deepseek-v4-flash",
    ]
    assert [item["ppu_count"] for item in matrix["include"]] == [4, 8]
    with pytest.raises(ValueError, match="unknown recipe policy key"):
        ppu_e2e_select.build_recipe_matrix("unknown-model")


def test_runner_fast_sanity_falls_back_from_2l_to_full_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int | None] = []

    def fake_attempt(*, layer_override, label, **kwargs):
        calls.append(layer_override)
        if layer_override == 2:
            return {
                "mode": label,
                "layer_override": 2,
                "status": "fail",
                "cold_start_s": 12.5,
                "eval_s": 0.0,
                "config_digest": "a" * 64,
                "ppu_count": 4,
                "error": ppu_recipe_runner._sanitize_error_text(
                    "2-layer failed at /nas_aisw/datasets/x"
                ),
            }
        return {
            "mode": label,
            "layer_override": None,
            "status": "pass",
            "cold_start_s": 45.0,
            "eval_s": 1.2,
            "config_digest": "b" * 64,
            "ppu_count": 4,
            "probe": {"status": "pass", "elapsed_s": 1.2},
        }

    monkeypatch.setattr(ppu_recipe_runner, "_attempt_fast_sanity_once", fake_attempt)
    res = ppu_recipe_runner.run_fast_sanity(
        recipe_path=RECIPES_DIR / "zai-org" / "GLM-5.3.yaml",
        policy_path=POLICY_DIR / "models" / "glm-5.3.yaml",
        results_dir=tmp_path / "out",
        parser_mode="skip",
        queue_s=3.5,
    )
    assert calls == [2, None]
    assert res["status"] == "pass"
    assert res["timings"] == {
        "queue_s": 3.5,
        "cold_start_s": 45.0,
        "eval_s": 1.2,
        "bench_s": 0.0,
    }
    assert (tmp_path / "out" / "summary.json").is_file()


def test_recipe_workflows_follow_security_and_shell_contracts() -> None:
    for wf_name in ("_ppu-recipe-sanity.yaml", "ppu-recipe-sanity.yaml"):
        wf = yaml.safe_load((ROOT / ".github" / "workflows" / wf_name).read_text())
        assert wf["cache-mode"] == "none"
        assert wf["permissions"] == {"contents": "read", "actions": "read"}
    script = CI_DIR / "ppu_e2e_recipe_sanity.sh"
    res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


@pytest.mark.requires_runtime
def test_runtime_vllm_parser_validates_checked_in_recipes() -> None:
    pytest.importorskip("vllm")
    files = _yaml_files()
    report = ppu_recipe.validate_paths(files, vllm_parser_mode="required")
    assert report["parser_checked"] is True
    assert report["checked_flag_sets"] >= 10
