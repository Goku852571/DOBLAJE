import tempfile
import unittest
from pathlib import Path

from core.tiktok_adapter import execute, read_urls, validate_url


class TikTokAdapterTests(unittest.TestCase):
    def test_rejects_other_platforms_and_deceptive_hosts(self):
        for url in ("https://youtube.com/watch?v=1", "https://tiktok.com.evil.test/x",
                    "file:///tmp/tiktok.com", "https://eviltiktok.com/x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_url(url)

    def test_accepts_tiktok_short_and_regular_urls(self):
        for url in ("https://www.tiktok.com/@u/video/123", "https://vm.tiktok.com/abc/",
                    "https://vt.tiktok.com/abc/"):
            self.assertEqual(validate_url(url), url)

    def test_txt_preserves_order_deduplicates_and_handles_bom(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "serie ñ.txt"
            a, b = "https://www.tiktok.com/@u/video/2", "https://www.tiktok.com/@u/video/1"
            path.write_text(a + "\n# comentario\n" + b + "\n" + a, encoding="utf-8-sig")
            self.assertEqual(read_urls(path), [a, b])

    def test_txt_rejects_invalid_line_without_partial_download(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "serie.txt"
            path.write_text("https://tiktok.com/@u/video/1\nhttps://example.com/video", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_urls(path)

    def test_relative_resources_created_inside_copied_project(self):
        from core.tiktok.runtime import configure
        with tempfile.TemporaryDirectory() as folder:
            resources = configure(root=folder)
            for path in (resources.downloads, resources.profile, resources.temp, resources.logs):
                self.assertTrue(path.is_dir())
                self.assertIn(Path(folder).resolve(), path.parents)

    def test_absolute_and_escaping_resource_paths_rejected(self):
        from core.tiktok.runtime import configure
        with tempfile.TemporaryDirectory() as folder:
            for value in (str(Path(folder).resolve()), "../outside"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    configure({"tiktok_download_dir": value}, root=folder)

    def test_errors_have_structured_contract(self):
        result = execute(dict(mode="url", source="https://example.com/"))
        for key in ("success", "status", "series_dir", "chapters_found",
                    "chapters_downloaded", "chapters_failed", "merged_video", "error"):
            self.assertIn(key, result)
        self.assertFalse(result["success"])

    def test_lock_releases_after_close_for_resume(self):
        from core.tiktok.runtime import configure, acquire_pipeline_lock
        with tempfile.TemporaryDirectory() as folder:
            resources = configure(root=folder)
            lock = acquire_pipeline_lock(resources)
            try:
                with self.assertRaises(RuntimeError):
                    acquire_pipeline_lock(resources)
            finally:
                lock.close()
            acquire_pipeline_lock(resources).close()
        configure()
