import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from core.media import _ffmpeg_binary, run_cmd
from core.tiktok import unifier
from core.tiktok.runtime import configure


class InternalUnifierTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="Unión ñ ' ")
        self.root = Path(self.folder.name).resolve()
        configure(root=self.root)
        self.chapters = self.root / "data/descargas/Serie/capitulos"
        self.chapters.mkdir(parents=True)
        self.first = self.chapters / "Capitulo_01.mp4"
        run_cmd([_ffmpeg_binary(), "-y", "-v", "error", "-f", "lavfi", "-i",
                 "testsrc2=size=320x240:rate=25:duration=2",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                 "-c:v", "mpeg4", "-q:v", "2", "-c:a", "aac", str(self.first)],
                capture_output=True, check=True)
        shutil.copyfile(self.first, self.chapters / "Capitulo_02.mp4")

    def tearDown(self):
        configure()
        self.folder.cleanup()

    def test_failed_fast_merge_uses_real_h264_aac_fallback(self):
        commands = []
        def execute(cmd, **kwargs):
            commands.append(cmd)
            if "concat" in cmd and "copy" in cmd:
                return subprocess.CompletedProcess(cmd, 1, "", "simulated stream mismatch")
            return run_cmd(cmd, **kwargs)
        with patch.object(unifier, "run_cmd", side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            result = unifier.unir_videos_serie(str(self.chapters), self.root / "final.mp4")
        self.assertTrue(result["success"], result)
        self.assertIn("Transcodificación", result["metodo"])
        self.assertTrue(any("libx264" in c and "aac" in c for c in commands))
        normalize = [c for c in commands if "pcm_s16le" in c]
        self.assertEqual(len(normalize), 2)
        self.assertTrue(all("48000" in c and "2" in c for c in normalize))

    def test_missing_audio_fails_instead_of_delivering_partial_movie(self):
        no_audio = self.chapters / "Capitulo_02.mp4"
        run_cmd([_ffmpeg_binary(), "-y", "-v", "error", "-i", str(self.first),
                 "-an", "-c:v", "copy", str(no_audio)], capture_output=True, check=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = unifier.unir_videos_serie(str(self.chapters), self.root / "final.mp4")
        self.assertFalse(result["success"])
        self.assertIn("Audio inválido", result["error"])
