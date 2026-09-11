"""CPU-only launcher checks: python3 -m unittest discover -s <this directory>."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_in_runpot", HERE / "run_in_runpot.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunPodTests(unittest.TestCase):
    def test_compose_argument_parity_including_list_flags(self):
        expected = yaml.safe_load((runner.ROOT / runner.COMPOSE).read_text(encoding="utf-8"))["services"][runner.SERVICE]["command"][2:]
        expected = [str(arg) for arg in expected]
        expected[0] = "--model-path=/weights/target"
        args, env = runner.launch_config(runner.ROOT, {"MODEL_PATH": "/weights/target"}, [])
        self.assertEqual(args, expected)
        self.assertEqual(env["NCCL_P2P_DISABLE"], "1")
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

    def test_env_and_cli_overrides_preserve_argument_boundaries(self):
        args, env = runner.launch_config(runner.ROOT, {
            "MODEL_PATH": "/weights/model with spaces", "TP_SIZE": "1", "PORT": "8042",
            "NCCL_P2P_DISABLE": "0", "WORKSPACE_DIR": "/persistent",
        }, ["--max-running-requests=4", "--served-model-name", "custom model"])
        self.assertIn("--model-path=/weights/model with spaces", args)
        self.assertIn("--tp-size=1", args)
        self.assertIn("--port=8042", args)
        self.assertEqual(args[-3:], ["--max-running-requests=4", "--served-model-name", "custom model"])
        self.assertEqual(env["NCCL_P2P_DISABLE"], "0")
        self.assertEqual(env["HF_HOME"], "/persistent/cache/huggingface")

    def test_profile_model_fallback_and_existing_model(self):
        profile = yaml.safe_load((runner.ROOT / runner.PROFILE).read_text(encoding="utf-8"))
        with mock.patch.object(Path, "is_file", return_value=False):
            args, _ = runner.launch_config(runner.ROOT, {}, [])
        self.assertIn("--model-path=" + profile["weights"]["autoround-int4"]["hf_repo"], args)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "target"
            target.mkdir()
            (target / "config.json").write_text("{}", encoding="utf-8")
            is_file = Path.is_file
            with mock.patch.object(Path, "is_file", lambda p: False if str(p).startswith("/models/target/") else is_file(p)):
                args, _ = runner.launch_config(runner.ROOT, {"MODEL_DIR": temp, "TARGET_DIR": "target"}, [])
            self.assertIn(f"--model-path={target}", args)

    def test_wrong_version_refused(self):
        with mock.patch.object(runner.importlib.metadata, "version", return_value="0.5.19"):
            with self.assertRaisesRegex(ValueError, "Expected SGLang 0.5.18"):
                runner.preflight([], {})

    def test_gpu_count_uses_last_tp_override(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1, get_device_name=lambda i: "test GPU"))
        with mock.patch.object(runner.importlib.metadata, "version", return_value="0.5.18"), \
             mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(ValueError, "found 1"):
                runner.preflight(["--tp-size=2"], {"SGLANG_DIR": "/source"})
            runner.preflight(["--tp-size=2", "--tp-size", "1"], {"SGLANG_DIR": "/source"})

    def test_piped_bootstrap_clones_and_reuses_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            origin = temp / "origin"
            origin.mkdir()
            paths = [runner.COMPOSE, runner.PROFILE,
                     HERE.relative_to(runner.ROOT) / "run_in_runpot.py"]
            for relative in paths:
                dest = origin / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((runner.ROOT / relative).read_bytes())
            def git(*args):
                subprocess.run(["git", "-C", str(origin), *args], check=True, capture_output=True)
            git("init", "-b", "bootstrap-test")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "Fixture")
            env = {**os.environ, "CLUB3090_REPO": str(origin), "CLUB3090_REF": "bootstrap-test",
                   "CLUB3090_DIR": str(temp / "clone"), "WORKSPACE_DIR": str(temp / "workspace"),
                   "MODEL_PATH": "/models/a target", "PORT": "8042"}
            for _ in range(2):
                result = subprocess.run(
                    ["bash", "-s", "--", "--dry-run", "--api-key", "do-not-print"],
                    input=(HERE / "run_in_runpot.sh").read_text(encoding="utf-8"),
                    env=env, capture_output=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("--model-path=/models/a target", result.stdout)
                self.assertIn("--port=8042", result.stdout)
                self.assertNotIn("do-not-print", result.stdout)
                self.assertFalse((temp / "workspace").exists())

    def test_installer_precedes_exec_and_failure_prevents_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            env = {"MODEL_PATH": "/target", "WORKSPACE_DIR": temp}
            with mock.patch.dict(os.environ, env, clear=True), \
                 mock.patch.object(sys, "argv", ["run_in_runpot.py"]), \
                 mock.patch.object(runner, "preflight"), \
                 mock.patch.object(runner.os, "chdir"), \
                 mock.patch.object(runner.subprocess, "run") as install, \
                 mock.patch.object(runner.os, "execvpe") as execute:
                events = mock.Mock()
                events.attach_mock(install, "install")
                events.attach_mock(execute, "execute")
                runner.main()
                self.assertEqual([call[0] for call in events.mock_calls], ["install", "execute"])
                self.assertEqual(execute.call_args.args[1][1:3], ["-m", "sglang.launch_server"])
                self.assertEqual(execute.call_args.args[2]["PYTHONPATH"], "/sgl-workspace/sglang/python")
                execute.reset_mock()
                install.side_effect = subprocess.CalledProcessError(1, "install.sh")
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.main()
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
