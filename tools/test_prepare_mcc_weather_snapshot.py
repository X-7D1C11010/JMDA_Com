"""Git-only snapshot preparation tests; no GPU or training is launched."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools.prepare_mcc_weather_v2_snapshot import MANIFEST, REQUIRED_FILES, prepare_snapshot


def bash_executable():
    for candidate in (r"D:\Git\Git\bin\bash.exe", r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def fixture(root):
    for name in REQUIRED_FILES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("# "+name+"\r\n").encode("utf-8"))
    weights = root/"pretrained_weights/unused.pth"
    weights.parent.mkdir()
    weights.write_bytes(b"not runtime code")


class SnapshotTests(unittest.TestCase):
    def test_git_checkout_builds_without_archive_and_reuses_verified_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            snapshot = prepare_snapshot(root)
            self.assertEqual(snapshot, prepare_snapshot(root))
            self.assertEqual(snapshot.parent, root/"experiment_updates")
            self.assertFalse(list(snapshot.rglob("*.pth")))
            self.assertFalse(list(snapshot.rglob("*.zip")))
            manifest = json.loads((snapshot/MANIFEST).read_text())
            self.assertFalse(manifest["training_started"])
            for name, expected in manifest["files"].items():
                data = (snapshot/name).read_bytes()
                self.assertNotIn(b"\r\n", data)
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
            self.assertFalse(list(snapshot.parent.glob(".prepare_mcc_weather_*")))

    def test_source_change_creates_new_copy_and_preserves_previous_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            before = prepare_snapshot(root)
            name = "vds2raw/adaptation.py"
            original = (before/name).read_bytes()
            (root/name).write_text("# changed model code\n", encoding="utf-8")
            after = prepare_snapshot(root)
            self.assertNotEqual(before, after)
            self.assertEqual((before/name).read_bytes(), original)
            self.assertEqual((after/name).read_bytes(), b"# changed model code\n")

    def test_modified_runtime_is_not_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            damaged = prepare_snapshot(root)
            (damaged/"vds2raw/adaptation.py").write_bytes(b"# damaged\n")
            repaired = prepare_snapshot(root)
            self.assertNotEqual(damaged, repaired)
            self.assertEqual((damaged/"vds2raw/adaptation.py").read_bytes(), b"# damaged\n")
            self.assertEqual(repaired, prepare_snapshot(root))

    def test_incomplete_git_sync_fails_before_preparing_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            (root/"scripts/run_pending_adaptation_v2.sh").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "run_pending_adaptation_v2.sh"):
                prepare_snapshot(root)
            self.assertFalse((root/"experiment_updates").exists())

    def test_concurrent_preparation_returns_one_verified_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            with ThreadPoolExecutor(max_workers=3) as pool:
                copies = list(pool.map(prepare_snapshot, [root]*3))
            self.assertEqual(len(set(copies)), 1)
            self.assertFalse(list((root/"experiment_updates").glob(".prepare_mcc_weather_*")))

    def test_cli_stdout_is_only_the_runtime_path_for_shell_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            helper = Path(__file__).with_name("prepare_mcc_weather_v2_snapshot.py")
            result = subprocess.run([sys.executable, "-B", str(helper), "--project-root", str(root)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 1)
            self.assertTrue((Path(result.stdout.strip())/MANIFEST).is_file())


@unittest.skipUnless(bash_executable(), "Bash is needed for dispatch checks")
class PendingDispatchTests(unittest.TestCase):
    def capture(self, scope):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root/"runtime/scripts"
            scripts.mkdir(parents=True)
            weights = root/"project/pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"dispatch-only placeholder; never loaded")
            source = Path(__file__).resolve().parents[1]/"scripts/run_pending_adaptation_v2.sh"
            shutil.copyfile(source, scripts/source.name)
            (scripts/"run_vds2raw_da_methods.sh").write_text(
                '#!/usr/bin/env bash\n'
                'printf "VDS|%s|%s|%s|%s\\n" "$VDS2RAW_DA_METHODS" "$VDS2RAW_DA_PROFILE" '
                '"$VDS2RAW_SEEDS" "${VDS2RAW_DA_EPOCHS-unset}"\n', encoding="utf-8")
            (scripts/"run_weather_vds_core_methods.sh").write_text(
                '#!/usr/bin/env bash\n'
                'printf "WEATHER|%s|%s|%s|%s\\n" "$WEATHER_VDS_PROFILE" "$WEATHER_VDS_TARGETS" '
                '"$1" "${WEATHER_VDS_EPOCHS-unset}"\n', encoding="utf-8")
            environment = os.environ.copy()
            environment.update(JMDA_PROJECT_ROOT=(root/"project").as_posix(),
                               VDS2RAW_DA_METHODS="mdd dann deepcoral", VDS2RAW_SEEDS="7",
                               VDS2RAW_DA_EPOCHS="250", WEATHER_VDS_EPOCHS="100")
            result = subprocess.run([bash_executable(), str(scripts/source.name), "42", scope],
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return [row for row in result.stdout.splitlines() if row.startswith(("VDS|", "WEATHER|"))]

    def test_full_queue_includes_optimized_cdan_five_seeds_and_no_completed_vds_methods(self):
        self.assertEqual(self.capture("all"), [
            "VDS|mcc|mcc_v2|42|unset",
            "VDS|cdan|optimized|42 43 44 45 46|unset",
            "WEATHER|timing_v2|backlight rainy foggy dark|42|unset",
        ])

    def test_append_mode_runs_only_cdan_without_repeating_mcc_or_weather(self):
        self.assertEqual(self.capture("cdan-only"), ["VDS|cdan|optimized|42 43 44 45 46|unset"])


if __name__ == "__main__":
    unittest.main()
