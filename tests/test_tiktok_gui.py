import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from unittest.mock import Mock, patch
from PySide6.QtWidgets import QApplication, QDialog
from gui.main_window import MainWindow
from gui.tiktok_dialog import TikTokDialog


class TikTokGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()

    def test_local_video_selection_still_resets_segments_and_enables_analysis(self):
        self.window.segments = [{"final_text": "anterior"}]
        path = r"D:\carpeta con espacios\vídeo ñ.mp4"
        with patch("gui.main_window.QFileDialog.getOpenFileName", return_value=(path, "")):
            self.window.select_video()
        self.assertEqual(self.window.video_path, path)
        self.assertEqual(self.window.segments, [])
        self.assertTrue(self.window.btn_analyze.isEnabled())
        self.assertFalse(self.window.btn_tts.isEnabled())

    def test_reviewed_successful_merge_is_passed_to_existing_analysis(self):
        dialog = Mock()
        dialog.exec.return_value = QDialog.Accepted
        dialog.result = dict(success=True, merged_video=r"D:\serie ñ\Pelicula.mp4")
        with patch("gui.tiktok_dialog.TikTokDialog", return_value=dialog), patch.object(
                self.window, "start_analysis") as analyze:
            self.window.download_tiktok()
        analyze.assert_called_once()
        self.assertEqual(self.window.video_path, dialog.result["merged_video"])

    def test_failure_or_cancel_never_starts_analysis(self):
        for success, accepted in ((False, QDialog.Accepted), (True, QDialog.Rejected)):
            dialog = Mock()
            dialog.exec.return_value = accepted
            dialog.result = dict(success=success, merged_video="partial.mp4")
            with patch("gui.tiktok_dialog.TikTokDialog", return_value=dialog), patch.object(
                    self.window, "start_analysis") as analyze:
                self.window.download_tiktok()
            analyze.assert_not_called()

    def test_failed_merge_disables_review_handoff_and_shows_error(self):
        dialog = TikTokDialog(self.window)
        with patch("gui.tiktok_dialog.QMessageBox.warning") as warning:
            dialog.completed(dict(success=False, status="MERGE_PENDING", error="FFmpeg falló"))
        self.assertFalse(dialog.proceed.isEnabled())
        self.assertEqual(dialog.video.text(), "")
        warning.assert_called_once()
        dialog.close()

    def test_valid_merge_waits_for_user_review(self):
        dialog = TikTokDialog(self.window)
        dialog.auto_dub.setChecked(False)
        with patch.object(dialog, "accept") as accept:
            dialog.completed(dict(success=True, merged_video="complete.mp4",
                                  chapters_downloaded=2, chapters_found=2))
        accept.assert_not_called()
        self.assertTrue(dialog.proceed.isEnabled())
        self.assertEqual(dialog.video.text(), "complete.mp4")
        dialog.close()

    def test_auto_mode_queues_analysis_then_voices_then_render(self):
        from types import SimpleNamespace
        window = self.window
        window._auto_dub_output = "data/descargas/serie/Video_Doblado.mp4"
        window.worker = SimpleNamespace(manual_translation=False)
        with patch("gui.main_window.QTimer.singleShot") as schedule:
            window.on_analysis_finished([dict(start=0, end=1, asr_text="Hello", final_text="Hola")])
        self.assertEqual(schedule.call_args.args[1], window.start_tts)
        with patch("gui.main_window.QTimer.singleShot") as schedule:
            window.on_tts_finished(window.segments)
        self.assertEqual(schedule.call_args.args[1], window.start_render)

    def test_incomplete_translation_stops_automatic_voices(self):
        from types import SimpleNamespace
        self.window._auto_dub_output = "Video_Doblado.mp4"
        self.window.worker = SimpleNamespace(manual_translation=False)
        with patch("gui.main_window.QTimer.singleShot") as schedule, patch(
                "gui.main_window.QMessageBox.critical"):
            self.window.on_analysis_finished([dict(start=0, end=1, asr_text="Hello", final_text="")])
        schedule.assert_not_called()
        self.assertEqual(self.window._auto_dub_output, "")

    def test_integrated_dialog_has_no_project_or_python_selector(self):
        dialog = TikTokDialog(self.window)
        self.assertFalse(hasattr(dialog, "project"))
        self.assertFalse(hasattr(dialog, "python"))
        self.assertEqual(dialog.output.text(), "data/descargas")
        self.assertGreaterEqual(dialog.mode.findData("single"), 0)
        dialog.close()
