"""Use the compose's argv/environment without starting Docker inside RunPod."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

try:
    import yaml
except ImportError:
    sys.exit("[runpod] Missing PyYAML. Fix: python3 -m pip install PyYAML")

ROOT = Path(__file__).resolve().parents[4]
COMPOSE = Path("models/qwen3.8-27b/sglang/compose/dual/autoround-int4/mtp.yml")
PROFILE = Path("scripts/lib/profiles/models/qwen3.8-27b.yml")
PATCHES = Path("models/qwen3.8-27b/sglang/patches/sglang-v0518-scheduling-and-logging")
SERVICE = "sglang-qwen38-27b-mtp-dual"


def launch_config(root, environ, extra_args):
    """Keep compose argv intact except container paths and explicit overrides."""
    env = dict(environ)
    workspace = Path(env.get("WORKSPACE_DIR", "/workspace"))
    env.setdefault("HF_HOME", str(workspace / "cache/huggingface"))
    env.setdefault("SGLANG_CACHE_DIR", str(workspace / "cache/sglang"))
    # The compose default is the cache volume's in-container path. RunPod has
    # no compose mount, so place L3 on its persistent workspace by default.
    env.setdefault(
        "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR",
        str(Path(env["SGLANG_CACHE_DIR"]) / "hicache-file"),
    )
    service = yaml.safe_load((root / COMPOSE).read_text(encoding="utf-8"))["services"][SERVICE]
    command = service["command"]
    if len(command) < 3 or command[1] != "--" or "sglang.launch_server" not in command[0]:
        raise ValueError("Compose command layout changed; update the RunPod launcher.")
    args = [str(arg) for arg in command[2:]]
    for key, value in service.get("environment", {}).items():
        value = str(value)
        match = re.fullmatch(r"\$\{([A-Z0-9_]+):-([^}]*)\}", value)
        if match:
            value = env.get(match[1]) or match[2]
        elif "${" in value:
            raise ValueError(f"Unsupported compose environment expression: {key}")
        if match:
            env[key] = value
        else:
            env.setdefault(key, value)

    model = env.get("MODEL_PATH")
    if not model:
        candidates = (
            Path("/models/target"),
            Path(env.get("MODEL_DIR", str(workspace / "models")))
            / env.get("TARGET_DIR", "qwen3.8-27b-autoround-int4"),
        )
        model = next((str(p) for p in candidates if (p / "config.json").is_file()), None)
    if not model:
        profile = yaml.safe_load((root / PROFILE).read_text(encoding="utf-8"))
        model = profile["weights"]["autoround-int4"]["hf_repo"]
    overrides = {"--model-path": model}
    for flag, name in (("--tp-size", "TP_SIZE"), ("--port", "PORT")):
        if env.get(name):
            overrides[flag] = env[name]
    args = [f"{arg.split('=', 1)[0]}={overrides[arg.split('=', 1)[0]]}"
            if arg.split("=", 1)[0] in overrides else arg for arg in args]
    args.extend(extra_args)
    env.setdefault("SGLANG_DIR", "/sgl-workspace/sglang")
    # The installer patches this source; make sure launch_server imports it.
    source_python = str(Path(env["SGLANG_DIR"]) / "python")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_python, env.get("PYTHONPATH"))))
    env["PYTHONUTF8"] = "1"
    return args, env


def preflight(args, env):
    version = importlib.metadata.version("sglang")
    if version != "0.5.18":
        raise ValueError(f"Expected SGLang 0.5.18, found {version}. Use lmsysorg/sglang:v0.5.18.")
    source = Path(env["SGLANG_DIR"]) / "python/sglang"
    if not (source / "launch_server.py").is_file():
        raise ValueError(f"No SGLang source at {source}; set SGLANG_DIR to the image source checkout.")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    known, _ = parser.parse_known_args(args)
    if known.tp_size < 1:
        raise ValueError("TP_SIZE must be positive.")
    import torch

    count = torch.cuda.device_count()
    if count < known.tp_size:
        raise ValueError(f"TP={known.tp_size} requires that many visible CUDA GPUs; found {count}. "
                         "Select a matching Pod or explicitly override TP_SIZE.")
    for index in range(count):
        print(f"[runpod] GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)


def main():
    extra = sys.argv[1:]
    dry_run = bool(extra and extra[0] == "--dry-run")
    if dry_run:
        extra = extra[1:]
    if extra and extra[0] == "--":
        extra = extra[1:]
    args, env = launch_config(ROOT, os.environ, extra)
    installer = ROOT / PATCHES / "install.sh"
    print(f"[runpod] Compose: {ROOT / COMPOSE}", flush=True)
    for key in (
        "SGLANG_DIR",
        "HF_HOME",
        "SGLANG_CACHE_DIR",
        "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR",
        "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE",
        "SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO",
        "NCCL_P2P_DISABLE",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        print(f"[runpod] {key}={env[key]}", flush=True)
    # CLI can include an API key; display a redacted copy only.
    display = []
    hide_next = False
    for arg in args:
        display.append("***" if hide_next else "--api-key=***" if arg.startswith("--api-key=") else arg)
        hide_next = arg == "--api-key"
    print("[runpod] " + shlex.join(["bash", str(installer)]), flush=True)
    print("[runpod] " + shlex.join([sys.executable, "-m", "sglang.launch_server", *display]), flush=True)
    if dry_run:
        return
    preflight(args, env)
    for key in (
        "HF_HOME",
        "SGLANG_CACHE_DIR",
        "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR",
    ):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    subprocess.run(["bash", str(installer)], env=env, check=True, stdin=subprocess.DEVNULL)
    print("[runpod] Starting SGLang in foreground; missing HF weights download into HF_HOME.", flush=True)
    os.chdir(env["SGLANG_DIR"])
    os.execvpe(sys.executable, [sys.executable, "-m", "sglang.launch_server", *args], env)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError, subprocess.CalledProcessError) as exc:
        sys.exit(f"[runpod] {exc}")
