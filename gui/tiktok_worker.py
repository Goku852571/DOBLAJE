"""Download/extract/merge worker, separate from the existing dubbing workers."""
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import sys
from core.tiktok.runtime import configure

from PySide6.QtCore import QThread, Signal


class TikTokWorker(QThread):
    log = Signal(str)
    stage = Signal(str)
    completed = Signal(dict)

    def __init__(self, request):
        super().__init__()
        self.request = dict(request)

        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()

    def run(self):
        response = dict(success=False, status="ERROR", series_dir="", chapters_found=0,
                        chapters_downloaded=0, chapters_failed=[], merged_video="", error="")
        try:
            resources = configure(self.request.get("settings"))
            with tempfile.TemporaryDirectory(prefix="video_sub_tiktok_", dir=resources.temp) as tmp:
                request_file = Path(tmp) / "request.json"
                result_file = Path(tmp) / "result.json"
                request_file.write_text(json.dumps(self.request, ensure_ascii=False), encoding="utf-8")
                environment = os.environ.copy()
                environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", TERM="dumb")
                bundled = resources.root / ".python-packages"
                if str(bundled) in sys.path:
                    environment["PYTHONPATH"] = str(bundled)
                else:
                    environment.pop("PYTHONPATH", None)
                with subprocess.Popen(
                    [sys.executable, "-u", "-m", "core.tiktok_adapter", str(request_file), str(result_file)],
                    cwd=str(resources.root), env=environment, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding="utf-8", errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
                    lines = queue.Queue()

                    def read_output():
                        for line in process.stdout:
                            lines.put(line.rstrip())

                    reader = threading.Thread(target=read_output, daemon=True)
                    reader.start()
                    tail = []
                    while process.poll() is None or reader.is_alive() or not lines.empty():
                        if self.cancelled.is_set() and process.poll() is None:
                            if os.name == "nt":
                                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                               capture_output=True,
                                               creationflags=subprocess.CREATE_NO_WINDOW)
                            else:
                                process.terminate()
                            process.wait()
                            response.update(status="CANCELLED",
                                            error="Proceso cancelado. Los capítulos válidos se conservan para reanudar.")
                            break
                        try:
                            line = lines.get(timeout=.1)
                        except queue.Empty:
                            continue
                        tail.append(line)
                        tail = tail[-30:]
                        with (resources.logs / "tiktok.log").open("a", encoding="utf-8") as log_file:
                            log_file.write(line + "\n")
                        self.log.emit(line)
                        if line.startswith("VIDEO_SUB_STAGE "):
                            self.stage.emit(line.removeprefix("VIDEO_SUB_STAGE "))
                        elif "reintent" in line.lower() or "recuperación" in line.lower():
                            self.stage.emit("Reintentos de descarga")
                    process.wait()
                    if not self.cancelled.is_set():
                        if result_file.is_file():
                            response = json.loads(result_file.read_text(encoding="utf-8"))
                        else:
                            response["error"] = "El descargador terminó sin resultado:\n" + "\n".join(tail)
        except Exception as exc:
            response["error"] = str(exc)
        if self.cancelled.is_set():
            response.update(success=False, status="CANCELLED", merged_video="")
        self.completed.emit(response)
