"""ASR contracts: no lost dialogue, stable timing and offline model loading."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from test_audio_timing import load_source, segment, word
from core.fusion import fuse_segments


class ASRIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.source = load_source("asr_engine", {"WhisperModel": Mock()})
        self.engine = self.source.ASREngine()

    def test_chinese_across_whisper_windows_has_no_inserted_spaces(self):
        result = self.engine._resegment([
            segment([word(" 你", 1, 1.2)]),
            segment([word(" 好", 1.2, 1.4)]),
        ], 10, "zh")
        self.assertEqual([s["text"] for s in result], ["你好"])
        self.assertEqual([(s["start"], s["end"]) for s in result], [(1, 1.4)])
        self.assertEqual(len(result[0]["words"]), 2)

    def test_short_tail_does_not_bridge_a_speech_gap(self):
        result = self.engine._resegment([
            segment([word(" Hello", 1, 2)]),
            segment([word(" Later", 5, 5.2)]),
        ], 10, "en")
        self.assertEqual([(s["start"], s["end"]) for s in result], [(1, 2), (5, 5.2)])

    def test_window_without_word_times_is_not_lost(self):
        result = self.engine._resegment([
            segment([word(" One", 1, 2)]),
            segment([], start=3, end=4, text="Two"),
            segment([word(" Three", 5, 6)]),
        ], 10, "en")
        self.assertEqual([s["text"] for s in result], ["One", "Two", "Three"])
        fused = fuse_segments([], result)
        self.assertEqual([s["source_text"] for s in fused], ["One", "Two", "Three"])
        self.assertEqual([s["asr_text"] for s in fused], ["One", "Two", "Three"])

    def test_decimal_is_not_split_as_sentence_end(self):
        result = self.engine._resegment([
            segment([word(" 3.", 1, 1.2), word("14", 1.2, 1.4)]),
        ], 10, "en")
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["words"]), 2)

    def test_long_sentence_keeps_all_words_and_bounded_intervals(self):
        words = [word(f" w{i}", i * 0.5, (i + 1) * 0.5) for i in range(36)]
        result = self.engine._resegment([segment(words)], 18, "en")
        self.assertEqual([w["word"] for s in result for w in s["words"]],
                         [w.word for w in words])
        self.assertTrue(all(s["end"] - s["start"] <= 8 for s in result))

    def test_incomplete_local_model_fails_without_downloading(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = self.source.ASREngine(model_path=folder)
            with self.assertRaisesRegex(RuntimeError, "Modelo ASR local incompleto"):
                engine.initialize()
        self.source.WhisperModel.assert_not_called()

    def test_loading_failure_does_not_silently_use_base(self):
        with tempfile.TemporaryDirectory() as folder:
            for filename in ("model.bin", "config.json", "tokenizer.json"):
                (Path(folder) / filename).touch()
            engine = self.source.ASREngine(model_path=folder)
            self.source.WhisperModel.side_effect = RuntimeError("memory")
            with self.assertRaisesRegex(RuntimeError, "No se pudo cargar"):
                engine.initialize()
        self.assertEqual(self.source.WhisperModel.call_count, 1)
        self.assertTrue(self.source.WhisperModel.call_args.kwargs["local_files_only"])
        self.assertEqual(engine.model_size, "large-v3-turbo")


if __name__ == "__main__":
    unittest.main()
