#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Temporary location in vllm-sail; migrates to centralized hub executor/ in Phase 2.
"""Execute ``fast_sanity`` and ``model_sanity`` from ``ppu-serving/v1`` recipes."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ppu_recipe import (  # noqa: E402
    redact_and_verify_public_payload,
    resolve_recipe,
    validate_paths,
)

FATAL_RE = re.compile(
    r"Traceback \(most recent"
    r"|EngineCore failed"
    r"|Engine core initialization failed"
    r"|core dumped"
    r"|Failed to infer device type"
    r"|Error: model architectures"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return data


def _resolve_dataset_spec(dataset_ref: str) -> dict[str, Any]:
    if "@" not in dataset_ref:
        raise ValueError(
            f"dataset reference must be <id>@<version>, got {dataset_ref!r}"
        )
    ds_id, ds_ver = dataset_ref.split("@", 1)
    ds_path = REPO_ROOT / "tests" / "ppu" / "policy" / "datasets" / f"{ds_id}.yaml"
    spec = _load_yaml(ds_path)
    if spec.get("id") != ds_id or str(spec.get("version")) != ds_ver:
        raise ValueError(
            f"{ds_path}: expected id={ds_id!r} version={ds_ver!r}, "
            f"got id={spec.get('id')!r} version={spec.get('version')!r}"
        )
    return spec


def _sanitize_error_text(text: str, *, max_chars: int = 1200) -> str:
    """Strip host paths, IPs, and internal hosts from failure excerpts."""
    cleaned = re.sub(
        r"/(?:nas_aisw|ppusw|mnt|Users|home|root|data)/[^\s'\"]+",
        "<redacted_path>",
        text,
    )
    cleaned = re.sub(
        r"[A-Za-z0-9._-]+\.(?:alibaba-inc\.com|eng\.t-head\.cn)",
        "<redacted_host>",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b", "<redacted_ip>", cleaned
    )
    if len(cleaned) > max_chars:
        cleaned = cleaned[-max_chars:]
    return cleaned


def _http_get_json(url: str, *, timeout: float = 5.0) -> tuple[int, Any]:
    with urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _probe_ready(port: int, expected_served_name: str) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            if resp.status != 200:
                return False
        status, payload = _http_get_json(
            f"http://127.0.0.1:{port}/v1/models", timeout=5
        )
        if status != 200 or not isinstance(payload, dict):
            return False
        for item in payload.get("data", []):
            if isinstance(item, dict) and item.get("id") == expected_served_name:
                return True
    except (URLError, OSError, ValueError):
        return False
    return False


def _stop_server(
    proc: subprocess.Popen[bytes] | None, *, grace_s: float = 30.0
) -> None:
    """Stop vLLM server and all surviving workers in its process group."""
    if proc is None:
        return
    # start_new_session=True sets pgid == proc.pid; workers may outlive the leader.
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass

    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass

    # Unconditionally SIGKILL the process group even if leader already exited.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _wait_server_ready(
    proc: subprocess.Popen[bytes],
    log_path: Path,
    *,
    port: int,
    expected_served_name: str,
    timeout: int,
) -> float:
    t0 = time.monotonic()
    deadline = t0 + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = (
                log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
                if log_path.exists()
                else ""
            )
            raise RuntimeError(
                f"vllm serve exited with code {proc.returncode} before ready.\n{tail}"
            )
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if FATAL_RE.search(text):
                raise RuntimeError(f"fatal error in vllm serve log:\n{text[-2500:]}")
        if _probe_ready(port, expected_served_name):
            return round(time.monotonic() - t0, 3)
        time.sleep(3)
    tail = (
        log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
        if log_path.exists()
        else ""
    )
    raise RuntimeError(f"server not ready within {timeout}s.\n{tail}")


def _send_chat_probe(
    *,
    port: int,
    served_name: str,
    probe: dict[str, Any],
    timeout: int = 180,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": served_name,
        "messages": [{"role": "user", "content": probe["prompt"]}],
        "max_completion_tokens": int(probe.get("max_tokens", 128)),
        "temperature": float(probe.get("temperature", 0.0)),
    }
    if "seed" in probe:
        body["seed"] = int(probe["seed"])
    req = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    elapsed = round(time.monotonic() - t0, 3)
    choice = raw["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    combined = f"{content}\n{reasoning}".strip()
    completion_tokens = int((raw.get("usage") or {}).get("completion_tokens") or 0)
    expected = probe.get("expected_substring")
    if expected:
        if not combined:
            raise RuntimeError(f"empty completion from chat probe: {raw}")
        if expected not in combined:
            raise RuntimeError(
                f"probe response missing expected substring {expected!r}: {combined[:200]!r}"
            )
    elif completion_tokens < 1 and not combined:
        raise RuntimeError(f"empty dummy completion from chat probe: {raw}")
    return {
        "status": "pass",
        "elapsed_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "content_chars": len(content),
        "completion_tokens": completion_tokens,
    }


def _parse_prometheus_metric(metrics_text: str, metric_name: str) -> float | None:
    total = 0.0
    found = False
    prefix = metric_name
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(prefix + " ") or line.startswith(prefix + "{"):
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    total += float(parts[1])
                    found = True
                except ValueError:
                    continue
    return total if found else None


def _drain_and_reset_cache(port: int, *, timeout: int = 120) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        running = _parse_prometheus_metric(text, "vllm:num_requests_running") or 0.0
        waiting = _parse_prometheus_metric(text, "vllm:num_requests_waiting") or 0.0
        if running == 0.0 and waiting == 0.0:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"server queues failed to drain within {timeout}s")

    req = Request(
        f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST"
    )
    with urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"/reset_prefix_cache returned HTTP {resp.status}")
        body_text = resp.read().decode("utf-8", errors="replace").strip()
    if body_text:
        try:
            body_json = json.loads(body_text)
        except ValueError:
            body_json = None
        if isinstance(body_json, dict) and body_json.get("success") is False:
            raise RuntimeError(f"/reset_prefix_cache reported failure: {body_json}")
    return {"cache_reset_evidence": "endpoint_ack_only", "http_status": 200}


def _extract_per_question_evalscope(
    eval_work_dir: Path,
    output_jsonl: Path,
) -> dict[str, Any]:
    """Extract sanitized per-question GSM8K records for future paired McNemar comparison."""
    records: list[dict[str, Any]] = []
    candidates = sorted(
        list(eval_work_dir.rglob("*.jsonl"))
        + list(eval_work_dir.rglob("reviews/**/*.json"))
    )
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        if path.suffix == ".jsonl":
            rows = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        else:
            try:
                loaded = json.loads(text)
            except ValueError:
                continue
            rows = loaded if isinstance(loaded, list) else []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            if not any(
                k in row
                for k in (
                    "score",
                    "is_correct",
                    "correct",
                    "sample_score",
                    "pred",
                    "prediction",
                )
            ):
                continue
            raw_score = row.get("score")
            if raw_score is None and isinstance(row.get("sample_score"), dict):
                sc = row["sample_score"].get("score")
                raw_score = sc.get("value") if isinstance(sc, dict) else sc
            correct = row.get("is_correct")
            if correct is None:
                correct = row.get("correct")
            if correct is None and isinstance(raw_score, (int, float)):
                correct = bool(float(raw_score) > 0.5)
            records.append(
                {
                    "index": row.get("index", idx),
                    "question_id": str(row.get("id", row.get("question_id", idx))),
                    "correct": bool(correct) if correct is not None else None,
                    "score": float(raw_score)
                    if isinstance(raw_score, (int, float))
                    else None,
                }
            )
        if records:
            break

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for item in records:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"count": len(records), "path": output_jsonl.name}


def _launch_resolved_server(
    resolved: dict[str, Any],
    *,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in resolved.get("server_env", {}).items()})
    env.update({k: str(v) for k, v in resolved.get("runtime_env", {}).items()})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            resolved["argv"],
            start_new_session=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
    finally:
        log_fh.close()


def _attempt_fast_sanity_once(
    *,
    recipe_path: Path,
    policy_path: Path,
    preset: str | None,
    assets_map: Path | None,
    port: int,
    ready_timeout: int,
    layer_override: int | None,
    probe: dict[str, Any],
    results_dir: Path,
    label: str,
) -> dict[str, Any]:
    base_name = f"recipe-{label}-{secrets.token_hex(4)}"
    resolved = resolve_recipe(
        recipe_path,
        policy_path=policy_path,
        tier="fast_sanity",
        preset=preset,
        layer_override=layer_override,
        assets_map_path=assets_map,
        port=port,
        served_model_name=base_name,
    )
    log_path = results_dir / f"vllm_serve_{label}.log"
    proc: subprocess.Popen[bytes] | None = None
    t_start = time.monotonic()
    try:
        proc = _launch_resolved_server(resolved, log_path=log_path)
        cold_start_s = _wait_server_ready(
            proc,
            log_path,
            port=port,
            expected_served_name=base_name,
            timeout=ready_timeout,
        )
        t_eval0 = time.monotonic()
        probe_res = _send_chat_probe(port=port, served_name=base_name, probe=probe)
        eval_s = round(time.monotonic() - t_eval0, 3)
        return {
            "mode": label,
            "layer_override": layer_override,
            "status": "pass",
            "cold_start_s": cold_start_s,
            "eval_s": eval_s,
            "config_digest": resolved["config_digest"],
            "ppu_count": resolved["ppu_count"],
            "probe": probe_res,
        }
    except Exception as exc:
        elapsed = round(time.monotonic() - t_start, 3)
        return {
            "mode": label,
            "layer_override": layer_override,
            "status": "fail",
            "cold_start_s": elapsed,
            "eval_s": 0.0,
            "config_digest": resolved["config_digest"],
            "ppu_count": resolved["ppu_count"],
            "error": _sanitize_error_text(str(exc)),
        }
    finally:
        _stop_server(proc)


def run_fast_sanity(
    *,
    recipe_path: Path,
    policy_path: Path,
    results_dir: Path,
    assets_map: Path | None = None,
    preset: str | None = None,
    port: int = 9985,
    ready_timeout: int = 1800,
    layer_mode: str = "auto",
    parser_mode: str = "required",
    queue_s: float = 0.0,
) -> dict[str, Any]:
    """Run ``fast_sanity`` (2-layer dummy first, optional fallback to full-layer dummy)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    val_report = validate_paths(
        [recipe_path, policy_path], vllm_parser_mode=parser_mode
    )
    policy = _load_yaml(policy_path)
    fs_cfg = policy["tiers"]["fast_sanity"]
    probe = fs_cfg["probe"]
    fallback_full = bool(fs_cfg.get("fallback_full_layers", True))

    attempts: list[dict[str, Any]] = []
    if layer_mode in {"auto", "2l", "both"}:
        two_l = int(fs_cfg.get("layer_override", 2))
        res_2l = _attempt_fast_sanity_once(
            recipe_path=recipe_path,
            policy_path=policy_path,
            preset=preset,
            assets_map=assets_map,
            port=port,
            ready_timeout=ready_timeout,
            layer_override=two_l,
            probe=probe,
            results_dir=results_dir,
            label="fast_2l",
        )
        attempts.append(res_2l)
    if layer_mode == "full" or (
        layer_mode == "both"
        or (layer_mode == "auto" and fallback_full and attempts[-1]["status"] != "pass")
    ):
        res_full = _attempt_fast_sanity_once(
            recipe_path=recipe_path,
            policy_path=policy_path,
            preset=preset,
            assets_map=assets_map,
            port=port + (1 if attempts else 0),
            ready_timeout=ready_timeout,
            layer_override=None,
            probe=probe,
            results_dir=results_dir,
            label="fast_full",
        )
        attempts.append(res_full)

    passed = any(item["status"] == "pass" for item in attempts)
    final_attempt = next(
        (item for item in attempts if item["status"] == "pass"), attempts[-1]
    )
    summary = {
        "tier": "fast_sanity",
        "model_id": policy["model_id"],
        "preset": preset,
        "status": "pass" if passed else "fail",
        "parser_checked": val_report["parser_checked"],
        "config_digest": final_attempt["config_digest"],
        "timings": {
            "queue_s": round(float(queue_s), 3),
            "cold_start_s": final_attempt["cold_start_s"],
            "eval_s": final_attempt["eval_s"],
            "bench_s": 0.0,
        },
        "attempts": attempts,
    }
    redact_and_verify_public_payload(summary, allow_loopback=True)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def _run_vllm_bench(
    *,
    served_name: str,
    model_path: str,
    port: int,
    concurrency: int,
    num_prompts: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    trust_remote_code: bool = False,
    result_dir: Path,
    result_filename: str,
) -> dict[str, Any]:
    if not (0.0 <= range_ratio < 1.0):
        raise ValueError(f"range_ratio must be in [0, 1), got {range_ratio!r}")
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--model",
        served_name,
        "--tokenizer",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--endpoint",
        "/v1/chat/completions",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        str(range_ratio),
        "--max-concurrency",
        str(concurrency),
        "--num-prompts",
        str(num_prompts),
        "--ignore-eos",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    subprocess.run(cmd, check=True)
    out_file = result_dir / result_filename
    return json.loads(out_file.read_text(encoding="utf-8"))


def run_model_sanity(
    *,
    recipe_path: Path,
    policy_path: Path,
    results_dir: Path,
    assets_map: Path | None = None,
    preset: str | None = None,
    port: int = 9985,
    ready_timeout: int = 3600,
    parser_mode: str = "required",
    queue_s: float = 0.0,
    limit: int | None = None,
    post_ready_sleep: int = 30,
) -> dict[str, Any]:
    """Run ``model_sanity``: 1 server -> probe -> GSM8K 1319 -> drain/reset -> bench [1,16]."""
    results_dir.mkdir(parents=True, exist_ok=True)
    val_report = validate_paths(
        [recipe_path, policy_path], vllm_parser_mode=parser_mode
    )
    policy = _load_yaml(policy_path)
    stages = {st["kind"]: st for st in policy["tiers"]["model_sanity"]["stages"]}
    eval_stage = stages["evalscope"]
    dataset_spec = _resolve_dataset_spec(eval_stage["dataset"])

    resolved = resolve_recipe(
        recipe_path,
        policy_path=policy_path,
        tier="model_sanity",
        preset=preset,
        assets_map_path=assets_map,
        port=port,
        served_model_name=f"recipe-ms-{secrets.token_hex(4)}",
    )
    unique_served = resolved["served_name"]
    gsm8k_path = os.environ.get("GSM8K_DATASET_PATH", "")
    if not gsm8k_path:
        raise OSError("GSM8K_DATASET_PATH must be set for model_sanity")

    ppu_gsm8k_eval = importlib.import_module("scripts.eval.ppu_gsm8k_eval")
    accuracy_gate = importlib.import_module("tests.e2e.models._accuracy_gate")

    log_path = results_dir / "vllm_serve_model_sanity.log"
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = _launch_resolved_server(resolved, log_path=log_path)
        ready_budget = max(
            ready_timeout,
            int(
                resolved.get("server_env", {}).get("VLLM_ENGINE_READY_TIMEOUT_S", 0)
                or 0
            ),
        )
        cold_start_s = _wait_server_ready(
            proc,
            log_path,
            port=port,
            expected_served_name=unique_served,
            timeout=ready_budget,
        )
        if post_ready_sleep > 0:
            time.sleep(post_ready_sleep)
        ppu_gsm8k_eval._ensure_launched_server_alive(
            proc, log_path, phase="post-readiness warm-up"
        )

        # 1. Deterministic sanity probe
        probe_res = _send_chat_probe(
            port=port,
            served_name=unique_served,
            probe=stages["sanity"]["probe"],
        )

        # 2. Full GSM8K via evalscope
        eval_t0 = time.monotonic()
        eval_work_dir = results_dir / "evalscope"
        gen_cfg = dict(eval_stage["generation"])
        result_raw = ppu_gsm8k_eval.run_evalscope(
            unique_served,
            port,
            gsm8k_path,
            generation_config=gen_cfg,
            limit=limit if limit is not None else eval_stage.get("limit"),
            work_dir=eval_work_dir,
        )
        score, num = ppu_gsm8k_eval.extract_score(result_raw)
        eval_s = round(time.monotonic() - eval_t0, 3)

        per_q_info = _extract_per_question_evalscope(
            eval_work_dir, results_dir / "gsm8k_per_question.jsonl"
        )

        golden_file = (
            REPO_ROOT
            / "tests"
            / "e2e"
            / "models"
            / "_golden"
            / accuracy_gate.golden_filename(resolved["golden_model_key"], "890p")
        )
        golden = accuracy_gate.read_golden(golden_file)
        ok_id, id_err = accuracy_gate.check_golden_identity(
            golden, chip="890p", checkpoint_tail=resolved["hf_model_id"]
        )
        ok_cfg, cfg_err = accuracy_gate.check_generation_config(golden, gen_cfg)
        ok_acc, acc_err = accuracy_gate.check_accuracy_golden(score, num, golden)
        acc_ok = ok_id and ok_cfg and ok_acc
        acc_error = id_err or cfg_err or acc_err

        # 3. Drain & clear prefix cache on the SAME server before bench
        reset_before_bench = _drain_and_reset_cache(port)

        # 4. Micro-perf bench (C=1, C=16) marked sanity_throughput_guard
        bench_stage = stages["bench"]
        bench_input = bench_stage["input"]
        bench_dir = results_dir / "bench"
        bench_dir.mkdir(parents=True, exist_ok=True)
        bench_t0 = time.monotonic()
        bench_results: dict[str, Any] = {}
        trust_remote = "--trust-remote-code" in resolved["flags"]
        warmup_map = bench_stage["warmup_count_by_concurrency"]
        req_map = bench_stage["request_count_by_concurrency"]
        for conc in bench_stage["concurrencies"]:
            warmup_raw = warmup_map.get(conc, warmup_map.get(str(conc)))
            req_raw = req_map.get(conc, req_map.get(str(conc)))
            if warmup_raw is None or req_raw is None:
                raise KeyError(f"missing warmup/request count for concurrency {conc!r}")
            warmup_n = int(warmup_raw)
            req_n = int(req_raw)
            _run_vllm_bench(
                served_name=unique_served,
                model_path=resolved["resolved_model"],
                port=port,
                concurrency=int(conc),
                num_prompts=warmup_n,
                input_len=int(bench_input["input_length"]),
                output_len=int(bench_input["output_length"]),
                range_ratio=float(bench_input.get("range_ratio", 0.0)),
                trust_remote_code=trust_remote,
                result_dir=bench_dir,
                result_filename=f"warmup_c{conc}.json",
            )
            _drain_and_reset_cache(port)
            raw_bench = _run_vllm_bench(
                served_name=unique_served,
                model_path=resolved["resolved_model"],
                port=port,
                concurrency=int(conc),
                num_prompts=req_n,
                input_len=int(bench_input["input_length"]),
                output_len=int(bench_input["output_length"]),
                range_ratio=float(bench_input.get("range_ratio", 0.0)),
                trust_remote_code=trust_remote,
                result_dir=bench_dir,
                result_filename=f"bench_c{conc}.json",
            )
            _drain_and_reset_cache(port)
            bench_results[f"c{conc}"] = {
                "concurrency": int(conc),
                "warmup_prompts": warmup_n,
                "num_prompts": req_n,
                "cache_mode": bench_stage.get("cache_mode", "sanity_throughput_guard"),
                "metrics": raw_bench,
            }
        bench_s = round(time.monotonic() - bench_t0, 3)

        golden_score = float(golden["score"])
        summary = {
            "tier": "model_sanity",
            "model_id": policy["model_id"],
            "hf_model_id": resolved["hf_model_id"],
            "golden_model_key": resolved["golden_model_key"],
            "preset": preset,
            "status": "pass" if acc_ok else "fail",
            "parser_checked": val_report["parser_checked"],
            "config_digest": resolved["config_digest"],
            "dataset": {"id": dataset_spec["id"], "version": dataset_spec["version"]},
            "timings": {
                "queue_s": round(float(queue_s), 3),
                "cold_start_s": cold_start_s,
                "eval_s": eval_s,
                "bench_s": bench_s,
            },
            "sanity_probe": probe_res,
            "accuracy": {
                "score": score,
                "num": num,
                "golden_score": golden_score,
                "delta": round(score - golden_score, 6),
                "within_band": acc_ok,
                "error": acc_error,
                "per_question": per_q_info,
            },
            "cache_reset_before_bench": {
                **reset_before_bench,
                "prior_workload_ids": ["sanity", "eval-gsm8k"],
            },
            "bench": bench_results,
        }
        redact_and_verify_public_payload(summary, allow_loopback=True)
        (results_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        _stop_server(proc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run PPU recipe fast_sanity / model_sanity"
    )
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--tier", choices=["fast_sanity", "model_sanity"], required=True
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--assets-map", type=Path, default=None)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--port", type=int, default=9985)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument(
        "--layer-mode", choices=["auto", "2l", "full", "both"], default="auto"
    )
    parser.add_argument(
        "--vllm-parser", choices=["auto", "required", "skip"], default="required"
    )
    parser.add_argument("--queue-s", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--post-ready-sleep", type=int, default=30)
    args = parser.parse_args(argv)

    if args.tier == "fast_sanity":
        res = run_fast_sanity(
            recipe_path=args.recipe,
            policy_path=args.policy,
            results_dir=args.results_dir,
            assets_map=args.assets_map,
            preset=args.preset,
            port=args.port,
            ready_timeout=args.ready_timeout,
            layer_mode=args.layer_mode,
            parser_mode=args.vllm_parser,
            queue_s=args.queue_s,
        )
    else:
        res = run_model_sanity(
            recipe_path=args.recipe,
            policy_path=args.policy,
            results_dir=args.results_dir,
            assets_map=args.assets_map,
            preset=args.preset,
            port=args.port,
            ready_timeout=args.ready_timeout,
            parser_mode=args.vllm_parser,
            queue_s=args.queue_s,
            limit=args.limit,
            post_ready_sleep=args.post_ready_sleep,
        )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
