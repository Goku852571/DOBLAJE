import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from core.tiktok import downloader


class InternalDownloaderTests(unittest.TestCase):
    def test_tikwm_failure_uses_playwright_fallback(self):
        fallback = dict(success=True, archivo="fixture.mp4", motor="Playwright")
        with patch.object(downloader, "_obtener_info_tikwm", return_value={"success": False}), patch.object(
                downloader, "_descargar_con_playwright", return_value=fallback) as browser, contextlib.redirect_stdout(io.StringIO()):
            result = downloader.descargar_video("https://www.tiktok.com/@u/video/1")
        self.assertEqual(result, fallback)
        browser.assert_called_once()

    def test_recoverable_error_retries_with_same_chapter_number(self):
        records = [dict(success=False, error="HTTP 429")]
        save = Mock()
        with patch.object(downloader, "descargar_video", return_value={"success": True}) as download, patch.object(
                downloader.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            downloader._reintentar_fallidos(records, ["https://tiktok.com/@u/video/1"], Path("capitulos"), save)
        self.assertTrue(records[0]["success"])
        self.assertEqual(download.call_args.kwargs["nombre_archivo_custom"], "Capitulo_01.mp4")
        save.assert_called_once()

    def test_permanently_removed_video_is_not_retried(self):
        with patch.object(downloader, "descargar_video") as download:
            downloader._reintentar_fallidos([dict(success=False, error="HTTP 404")],
                                           ["https://tiktok.com/@u/video/1"], Path("capitulos"), Mock())
        download.assert_not_called()
