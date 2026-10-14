import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from gui.tiktok_worker import TikTokWorker


class TikTokWorkerTests(unittest.TestCase):
    def test_child_returns_structured_error(self):
        worker = TikTokWorker(dict(mode="url", source="https://example.com/"))
        results = []
        worker.completed.connect(results.append)
        worker.run()
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["success"])
        self.assertEqual(results[0]["merged_video"], "")
        self.assertIn("TikTok", results[0]["error"])

    def test_cancel_stops_real_child_and_blocks_handoff(self):
        worker = TikTokWorker({})
        results = []
        worker.completed.connect(results.append)
        real_popen = subprocess.Popen
        def launch(*args, **kwargs):
            return real_popen([sys.executable, "-u", "-c",
                               "import time; print('esperando', flush=True); time.sleep(30)"], **kwargs)
        proxy = SimpleNamespace(Popen=launch, run=subprocess.run,
                                CREATE_NO_WINDOW=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT, DEVNULL=subprocess.DEVNULL)
        timer = threading.Timer(.3, worker.cancel)
        started = time.monotonic()
        timer.start()
        try:
            with patch("gui.tiktok_worker.subprocess", proxy):
                worker.run()
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(results[0]["status"], "CANCELLED")
        self.assertFalse(results[0]["success"])
        self.assertEqual(results[0]["merged_video"], "")
