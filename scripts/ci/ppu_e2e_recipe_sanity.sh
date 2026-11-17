#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Container entrypoint for recipe fast_sanity / model_sanity verification.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="${HF_HOME:?HF_HOME must be set}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets}"

unset VLLM_VERSION 2>/dev/null || true
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0

WHEELS_DIR="${WHEELS_DIR:?WHEELS_DIR is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
RECIPE_PATH="${RECIPE_PATH:?RECIPE_PATH is required}"
POLICY_PATH="${POLICY_PATH:?POLICY_PATH is required}"
TIER="${TIER:-fast_sanity}"
PRESET="${PRESET:-}"
LAYER_MODE="${LAYER_MODE:-auto}"
VLLM_PORT="${VLLM_PORT:-9985}"
PPU_CI_QUEUE_S="${PPU_CI_QUEUE_S:-0}"

mkdir -p "${RESULTS_DIR}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

WHEELS_DIR="$(cd -- "${WHEELS_DIR}" && pwd)"
python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${WHEELS_DIR}"

export PIP_CONSTRAINT="/tmp/ppu-recipe-image-constraints.txt"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" capture \
    --sdk /usr/local/PPU_SDK --output "/tmp/ppu-recipe-image-stack.json" \
    --constraints "${PIP_CONSTRAINT}"

mapfile -t vllm_wheels < <(find "${WHEELS_DIR}" -type f -name 'vllm-*.whl' -print | sort)
mapfile -t sail_wheels < <(find "${WHEELS_DIR}" -type f -name 'vllm_sail-*.whl' -print | sort)
((${#vllm_wheels[@]} == 1)) || { echo "expected one vLLM wheel under ${WHEELS_DIR}" >&2; exit 1; }
((${#sail_wheels[@]} == 1)) || { echo "expected one vLLM SAIL wheel under ${WHEELS_DIR}" >&2; exit 1; }

python -m pip install --no-deps --force-reinstall "${vllm_wheels[0]}"
python -m pip install --no-deps --force-reinstall "${sail_wheels[0]}"
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    python -m pip install -r "${ROOT_DIR}/requirements/ci-runtime.txt"
if [[ "${TIER}" == "model_sanity" ]]; then
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
        python -m pip install "evalscope==1.14.1"
fi

python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
    --sdk /usr/local/PPU_SDK --output "/tmp/ppu-recipe-image-stack.json"

cd /tmp
unset PYTHONPATH

python - <<'PY'
import torch
from vllm.platforms import current_platform
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility(force=True)
assert current_platform.is_ppu(), f"expected PPU platform, got {current_platform}"
assert torch.cuda.is_available(), "PPU device required"
PY

python - "${MODEL_ID}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], local_files_only=True)
PY

python "${ROOT_DIR}/scripts/ci/ppu_recipe.py" validate --all --require-vllm-parser

runner_args=(
    --recipe "${ROOT_DIR}/${RECIPE_PATH}"
    --policy "${ROOT_DIR}/${POLICY_PATH}"
    --tier "${TIER}"
    --results-dir "/tmp/recipe-run"
    --port "${VLLM_PORT}"
    --layer-mode "${LAYER_MODE}"
    --vllm-parser required
    --queue-s "${PPU_CI_QUEUE_S}"
)
if [[ -n "${PRESET}" ]]; then
    runner_args+=(--preset "${PRESET}")
fi

set +e
python "${ROOT_DIR}/scripts/ci/ppu_recipe_runner.py" "${runner_args[@]}"
rc=$?
set -e

if [[ -f "/tmp/recipe-run/summary.json" ]]; then
    python - "${ROOT_DIR}/scripts/ci/ppu_recipe.py" "/tmp/recipe-run/summary.json" "${RESULTS_DIR}/summary.json" <<'PY'
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).parent))
import ppu_recipe

src = Path(sys.argv[2])
dst = Path(sys.argv[3])
data = json.loads(src.read_text(encoding="utf-8"))
ppu_recipe.redact_and_verify_public_payload(data, allow_loopback=True)
shutil.copy2(src, dst)
PY
fi

if [[ -f "/tmp/recipe-run/gsm8k_per_question.jsonl" ]]; then
    python - "${ROOT_DIR}/scripts/ci/ppu_recipe.py" "/tmp/recipe-run/gsm8k_per_question.jsonl" "${RESULTS_DIR}/gsm8k_per_question.jsonl" <<'PY'
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).parent))
import ppu_recipe

src = Path(sys.argv[2])
dst = Path(sys.argv[3])
ppu_recipe.redact_and_verify_public_payload(src.read_text(encoding="utf-8"), allow_loopback=True)
shutil.copy2(src, dst)
PY
fi

exit "${rc}"
