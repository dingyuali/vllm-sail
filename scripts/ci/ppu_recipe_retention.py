import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

REQUIRED_FILES = ("summary.json", "provenance.json", "launcher.log")
OPTIONAL_FILE = "gsm8k_per_question.jsonl"


def _check_private_dir(directory: Path) -> None:
    mode = directory.lstat().st_mode
    if not stat.S_ISDIR(mode) or stat.S_IMODE(mode) != 0o700:
        raise ValueError("expected a private directory")


def retain(environ) -> None:
    index = {
        "checkout_sha": environ["PPU_CI_SOURCE_SHA"],
        "tier": environ["TIER"],
        "model_key": environ["MODEL_KEY"],
        "run_id": environ["GITHUB_RUN_ID"],
        "attempt": environ["GITHUB_RUN_ATTEMPT"],
    }
    if (
        index["tier"] not in {"fast_sanity", "model_sanity"}
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", index["model_key"])
        or not re.fullmatch(r"[0-9a-f]{40}", index["checkout_sha"])
        or not all(re.fullmatch(r"[0-9]+", index[key]) for key in ("run_id", "attempt"))
    ):
        raise ValueError("invalid retention metadata")

    root = Path(environ["PPU_CI_PRIVATE_RESULTS_ROOT"])
    if not root.is_absolute():
        raise ValueError("expected an absolute retention root")
    _check_private_dir(root)
    root = root.resolve(strict=True)
    for key in ("RUNNER_TEMP", "GITHUB_WORKSPACE"):
        ephemeral = Path(environ[key]).resolve()
        if ephemeral == root or ephemeral in root.parents:
            raise ValueError("retention root must not be job-local")

    parent = root
    for component in (
        "vllm-sail",
        f"{index['run_id']}-{index['attempt']}",
        index["tier"],
    ):
        parent /= component
        parent.mkdir(mode=0o700, exist_ok=True)
        _check_private_dir(parent)

    destination = parent / index["model_key"]
    if os.path.lexists(destination):
        raise FileExistsError("retention destination already exists")

    source_dir = Path(environ["PPU_CI_ARTIFACT_DIR"])
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=parent) as staging_dir:
        staging = Path(staging_dir)
        published = staging / index["model_key"]
        published.mkdir(mode=0o700)
        files = {}
        for name in (*REQUIRED_FILES, OPTIONAL_FILE):
            source = source_dir / name
            try:
                mode = source.lstat().st_mode
            except FileNotFoundError:
                if name == OPTIONAL_FILE:
                    continue
                raise
            if not stat.S_ISREG(mode):
                raise ValueError("expected a regular recipe output")
            copied = published / name
            shutil.copyfile(source, copied)
            copied.chmod(0o600)
            digest = hashlib.sha256()
            with copied.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[name] = digest.hexdigest()
        index_path = published / "index.json"
        index_path.write_text(
            json.dumps({**index, "files": files}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index_path.chmod(0o600)
        published.rename(destination)


def main() -> int:
    os.umask(0o077)
    status = "not_configured"
    exit_code = 0
    if os.environ.get("PPU_CI_PRIVATE_RESULTS_ROOT"):
        try:
            retain(os.environ)
        except (OSError, ValueError, KeyError, RuntimeError):
            status = "failed"
            exit_code = 1
        else:
            status = "retained"
    elif os.environ.get("TIER") == "fast_sanity":
        print("::warning::private results retention not configured")
    else:
        print("::error::private results retention not configured")
        exit_code = 1

    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as outputs:
        outputs.write(f"retention={status}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
