"""Installer regression tests; run with python3 test_install.py /path/to/sglang."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = Path(sys.argv.pop(1)).resolve()
PATCHES = sorted(HERE.glob("[0-9][0-9][0-9][0-9]-*.patch"))


def run(*args, **kwargs):
    return subprocess.run(args, capture_output=True, encoding="utf-8", **kwargs)


class InstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.base.cleanup)
        stats = run("git", "apply", "--numstat", *map(str, PATCHES), check=True)
        for path in {line.split("\t")[2] for line in stats.stdout.splitlines()}:
            result = run("git", "-C", str(SOURCE), "show", f"v0.5.18:{path}")
            if result.returncode == 0:
                dest = Path(cls.base.name) / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(result.stdout, encoding="utf-8")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tree = Path(self.temp.name) / "tree"
        shutil.copytree(self.base.name, self.tree)

    def apply_prefix(self, count):
        for patch in PATCHES[:count]:
            run("git", "apply", str(patch), cwd=self.tree, check=True)

    def install(self, *args):
        return run("bash", str(HERE / "install.sh"), *args,
                   env={**os.environ, "SGLANG_DIR": str(self.tree)})

    def hashes(self):
        return {str(p.relative_to(self.tree)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.tree.rglob("*") if p.is_file()}

    def test_fresh_and_idempotent(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        before = self.hashes()
        self.assertEqual(self.install().returncode, 0)
        self.assertEqual(self.install("--verify").returncode, 0)
        self.assertEqual(before, self.hashes())

    def test_supported_upgrades(self):
        for count in range(3, len(PATCHES)):
            with self.subTest(count=count):
                shutil.rmtree(self.tree)
                shutil.copytree(self.base.name, self.tree)
                self.apply_prefix(count)
                before = self.hashes()
                self.assertNotEqual(self.install("--verify").returncode, 0)
                self.assertEqual(before, self.hashes())
                result = self.install()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.install("--verify").returncode, 0)
                # Every preexisting quantization file remains byte-identical.
                after = self.hashes()
                for path, digest in before.items():
                    if "quantization/" in path or "marlin_w4a8/" in path:
                        if count >= 4:
                            self.assertEqual(digest, after[path], path)

    def test_partial_scheduling_rejected(self):
        self.apply_prefix(1)
        self.assert_rejected_unchanged()

    def assert_rejected_unchanged(self):
        before = self.hashes()
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(before, self.hashes())

    def test_partial_dflash_rejected(self):
        self.apply_prefix(5)
        (self.tree / "python/sglang/srt/models/dflash.py").write_text(
            "# interrupted patch\n", encoding="utf-8")
        self.assert_rejected_unchanged()

    def test_modified_quantized_head_patch_rejected(self):
        self.apply_prefix(6)
        path = self.tree / "python/sglang/srt/models/dflash.py"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "logits[:, num_org:] = float(\"-inf\")", "logits[:, num_org:] = 0"),
            encoding="utf-8")
        self.assert_rejected_unchanged()

    def test_modified_w4a8_rejected(self):
        self.apply_prefix(4)
        path = self.tree / "python/sglang/kernels/ops/quantization/gptq_marlin_w4a8.py"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "ENABLED = True", "ENABLED = False"), encoding="utf-8")
        self.assert_rejected_unchanged()

    def test_partial_mamba_rejected(self):
        self.apply_prefix(7)
        (self.tree / "test/registered/unit/mem_cache/test_mamba_alloc_req_slots_demand.py").unlink()
        self.assert_rejected_unchanged()

    def test_nine_patch_upgrade_installs_demand_fix_and_fairness(self):
        self.apply_prefix(9)
        before = self.hashes()
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        after = self.hashes()
        self.assertEqual(
            {path for path in before if before[path] != after[path]},
            {"python/sglang/srt/mem_cache/allocation.py",
             "python/sglang/srt/server_args.py",
             "test/registered/unit/mem_cache/test_mamba_alloc_req_slots_demand.py",
             "python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py",
             "test/registered/unit/mem_cache/test_mamba_path_state_cap.py",
             "test/registered/unit/mem_cache/test_mamba_eviction_thinning.py"},
        )
        self.assertEqual(set(after) - set(before), {
            "test/registered/unit/mem_cache/test_mamba_eviction_fairness.py",
            "test/registered/unit/mem_cache/test_mamba_path_cap_coverage.py",
            "test/manual/mem_cache/benchmark_mamba_path_cap.py"})
        self.assertIn("0010-fix-mamba-demand-v0518-request-fields.patch", result.stdout)
        verified = self.install("--verify")
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
        self.assertIn("all twelve applied", verified.stdout)
        self.assertEqual(self.install().returncode, 0)
        self.assertEqual(after, self.hashes())

    def test_ten_patch_upgrade_preserves_exact_demand_files(self):
        self.apply_prefix(10)
        before = self.hashes()
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("0011-mamba-inter-path-eviction-fairness.patch", result.stdout)
        after = self.hashes()
        self.assertEqual(
            {path for path in before if before[path] != after[path]},
            {"python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py",
             "python/sglang/srt/server_args.py",
             "test/registered/unit/mem_cache/test_mamba_path_state_cap.py",
             "test/registered/unit/mem_cache/test_mamba_eviction_thinning.py"},
        )
        self.assertEqual(self.install("--verify").returncode, 0)
        self.assertEqual(self.install().returncode, 0)
        self.assertEqual(after, self.hashes())

    def test_eleven_patch_upgrade_only_changes_path_cap(self):
        self.apply_prefix(11)
        before = self.hashes()
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("0012-mamba-path-cap-minimax-coverage.patch", result.stdout)
        after = self.hashes()
        self.assertEqual(
            {path for path in before if before[path] != after[path]},
            {"python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py",
             "python/sglang/srt/server_args.py",
             "test/registered/unit/mem_cache/test_mamba_path_state_cap.py"},
        )
        self.assertEqual(set(after) - set(before), {
            "test/registered/unit/mem_cache/test_mamba_path_cap_coverage.py",
            "test/manual/mem_cache/benchmark_mamba_path_cap.py"})
        self.assertEqual(self.install("--verify").returncode, 0)
        self.assertEqual(self.install().returncode, 0)
        self.assertEqual(after, self.hashes())

    def test_partial_coverage_patch_rejected(self):
        self.apply_prefix(12)
        (self.tree / "test/registered/unit/mem_cache/test_mamba_path_cap_coverage.py").unlink()
        self.assert_rejected_unchanged()

    def test_modified_coverage_score_rejected(self):
        self.apply_prefix(12)
        path = self.tree / "python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "max_gap = max(max_gap, depth - previous)", "max_gap = 0"), encoding="utf-8")
        self.assert_rejected_unchanged()

    def test_modified_demand_hotfix_rejected(self):
        self.apply_prefix(10)
        path = self.tree / "python/sglang/srt/mem_cache/allocation.py"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "if req.mamba_pool_idx is None:", "if req.mamba_pool_idx is not None:"),
            encoding="utf-8")
        self.assert_rejected_unchanged()

    def test_modified_mamba_thinning_rejected(self):
        self.apply_prefix(9)
        path = self.tree / "python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "or cd.metadata.get(MAMBA_REUSED_KEY)", "or False"), encoding="utf-8")
        self.assert_rejected_unchanged()

    def test_out_of_order_mamba_rejected(self):
        self.apply_prefix(6)
        run("git", "apply", str(PATCHES[7]), cwd=self.tree, check=True)
        self.assert_rejected_unchanged()

    def test_late_mamba_failure_preserves_six_patch_install(self):
        self.apply_prefix(6)
        path = self.tree / "python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py"
        path.write_text("# incompatible tree core\n", encoding="utf-8")
        self.assert_rejected_unchanged()

    def test_late_patch_failure_does_not_apply_earlier_patches(self):
        # 0004 cannot apply; preflight must keep 0001..0003 absent too.
        path = self.tree / "python/sglang/srt/layers/quantization/auto_round.py"
        path.write_text("# incompatible base\n", encoding="utf-8")
        self.assert_rejected_unchanged()


if __name__ == "__main__":
    unittest.main()
