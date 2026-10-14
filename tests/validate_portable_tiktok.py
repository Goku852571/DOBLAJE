"""Copy only application source, then deny access to any Downloads folder."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="Video_Sub portable ñ ") as folder:
    target = Path(folder).resolve()
    for name in ("core", "gui", "utils"):
        shutil.copytree(ROOT / name, target / name,
                        ignore=shutil.ignore_patterns("__pycache__", "._*"))
    (target / "tests").mkdir()
    shutil.copyfile(ROOT / "tests/validate_tiktok_bridge.py",
                    target / "tests/validate_tiktok_bridge.py")
    bootstrap = """
import pathlib, runpy, sys
def forbid_downloads(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        value = str(args[0]).replace(chr(92), "/").lower()
        if "downloads" in value.split("/"):
            raise PermissionError("Access to Downloads blocked by portability test")
sys.addaudithook(forbid_downloads)
runpy.run_path("tests/validate_tiktok_bridge.py", run_name="__main__")
from core.tiktok.runtime import ROOT
assert ROOT == pathlib.Path.cwd().resolve()
import core.tiktok.downloader
assert pathlib.Path(core.tiktok.downloader.__file__).resolve().is_relative_to(ROOT)
print("PASS: relocated source, Downloads inaccessible, internal resources created.")
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run([sys.executable, "-c", bootstrap], cwd=target, env=environment,
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    evidence = ROOT / "validation_tiktok/portability.log"
    evidence.write_text(process.stdout + process.stderr, encoding="utf-8")
    print(process.stdout)
    if process.returncode:
        raise RuntimeError(process.stderr)
