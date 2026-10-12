import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.whisper_cpp_engine import WhisperCppEngine
from core.fusion import fuse_segments


class WhisperCppTests(unittest.TestCase):
    def test_original_millisecond_timestamps_and_translation_contract(self):
        payload = {"result": {"language": "zh"}, "transcription": [
            {"text": "你好。", "offsets": {"from": 1200, "to": 2500}},
            {"text": "再见。", "offsets": {"from": 5000, "to": 8000}},
        ]}
        segments = WhisperCppEngine.parse_result(payload, 7)
        self.assertEqual([(s['start'], s['end']) for s in segments], [(1.2, 2.5), (5, 7)])
        self.assertEqual(fuse_segments(segments)[0]['source_text'], '你好。')
        self.assertEqual(segments[0]['language'], 'zh')

    def test_empty_vad_output_produces_no_dialogue(self):
        self.assertEqual(WhisperCppEngine.parse_result({'transcription': []}, 10), [])

    def test_nonfinite_times_are_rejected(self):
        with self.assertRaises(RuntimeError):
            WhisperCppEngine.parse_result({'transcription': [
                {'text': 'Hello', 'offsets': {'from': float('nan'), 'to': 1000}}]}, 10)

    def test_missing_model_stops_without_fallback_or_download(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = WhisperCppEngine(model_path=str(Path(folder) / 'missing.bin'))
            with patch('core.whisper_cpp_engine.subprocess.Popen') as launch:
                with self.assertRaises(RuntimeError):
                    engine.extract_audio_text('unused.wav')
                launch.assert_not_called()

    def test_tokens_map_to_original_speech_after_removed_silence(self):
        mapping = [(0.23, 26.33, 0.0, 26.1), (32.48, 44.83, 26.3, 38.65)]
        payload = {'result': {'language': 'en'}, 'transcription': [{
            'text': 'Sparrow from scolding.', 'offsets': {'from': 26230, 'to': 39520},
            'tokens': [
                {'text': ' Sp', 'offsets': {'from': 26000, 'to': 26200}},
                {'text': 'ar', 'offsets': {'from': 26250, 'to': 26420}},
                {'text': 'row', 'offsets': {'from': 26420, 'to': 26730}},
                {'text': ' from', 'offsets': {'from': 26730, 'to': 27150}},
                {'text': ' scolding', 'offsets': {'from': 27150, 'to': 28000}},
                {'text': '.', 'offsets': {'from': 28000, 'to': 30000}},
            ]}]}
        result = WhisperCppEngine.parse_result(payload, 60, mapping)
        self.assertAlmostEqual(result[0]['start'], 32.48)
        self.assertAlmostEqual(result[0]['end'], 34.18)
        self.assertEqual(result[0]['text'], 'Sparrow from scolding.')
        self.assertEqual(result[0]['words'][0]['word'], 'Sparrow')

    def test_alignment_splits_real_pause_preserving_every_word(self):
        payload = {'transcription': [{'text': 'Hello. Goodbye.', 'offsets': {'from': 0, 'to': 6000}, 'tokens': [
            {'text': ' Hello', 'offsets': {'from': 0, 'to': 700}},
            {'text': '.', 'offsets': {'from': 700, 'to': 1100}},
            {'text': ' Goodbye', 'offsets': {'from': 1200, 'to': 2000}},
            {'text': '.', 'offsets': {'from': 2000, 'to': 2000}},
        ]}]}
        result = WhisperCppEngine.parse_result(payload, 10, [(0, 1, 0, 1), (5, 6, 1.2, 2.2)])
        self.assertEqual([s['text'] for s in result], ['Hello.', 'Goodbye.'])
        self.assertAlmostEqual(result[0]['end'], .7)
        self.assertAlmostEqual(result[1]['start'], 5)
        self.assertAlmostEqual(result[1]['end'], 5.8)

    def test_collapsed_tokens_use_phrase_speech_window_without_losing_text(self):
        payload = {'transcription': [{
            'text': 'Tamori clings to her stone collection.',
            'offsets': {'from': 312520, 'to': 317380},
            'tokens': [
                {'text': ' Tamori', 'offsets': {'from': 183160, 'to': 183160}},
                {'text': ' clings', 'offsets': {'from': 183160, 'to': 183160}},
                {'text': ' to', 'offsets': {'from': 183160, 'to': 183160}},
                {'text': ' her', 'offsets': {'from': 183160, 'to': 183160}},
                {'text': ' stone', 'offsets': {'from': 183160, 'to': 183160}},
                {'text': ' collection.', 'offsets': {'from': 183160, 'to': 185090}},
            ]}]}
        mapping = [(297.63, 312.57, 168.27, 183.21), (315.01, 321.25, 183.41, 189.65)]
        result = WhisperCppEngine.parse_result(payload, 392.626, mapping)
        self.assertEqual(result[0]['text'], payload['transcription'][0]['text'])
        self.assertAlmostEqual(result[0]['start'], 315.01)
        self.assertAlmostEqual(result[0]['end'], 317.38)
        self.assertEqual(result[0]['words'], [])

    def test_leading_boundary_word_stays_with_resumed_phrase(self):
        payload = {'transcription': [{'text': 'Fat Guga walks.',
            'offsets': {'from': 320980, 'to': 335020}, 'tokens': [
                {'text': ' Fat', 'offsets': {'from': 189380, 'to': 189580}},
                {'text': ' Guga', 'offsets': {'from': 189630, 'to': 189890}},
                {'text': ' walks.', 'offsets': {'from': 189890, 'to': 190240}},
            ]}]}
        mapping = [(315.01, 321.25, 183.41, 189.65), (332.83, 344.86, 189.85, 201.88)]
        result = WhisperCppEngine.parse_result(payload, 392.626, mapping)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'Fat Guga walks.')
        self.assertAlmostEqual(result[0]['start'], 332.83)

    def test_missing_vad_map_refuses_unaligned_text(self):
        with self.assertRaises(RuntimeError):
            WhisperCppEngine.parse_result({'transcription': [{'text': 'Hello'}]}, 10, [])

    def test_invalid_token_times_keep_complete_phrase_on_original_vad_timeline(self):
        for offsets in ({"from": -10, "to": -10}, {"from": 500, "to": 100},
                        {}, None, {"from": None, "to": 500},
                        {"from": float("nan"), "to": 500}):
            with self.subTest(offsets=offsets):
                item = {"text": "Hola mundo.", "offsets": {"from": 5100, "to": 8500},
                        "tokens": [{"text": " Hola", "offsets": {"from": 0, "to": 300}},
                                   {"text": " mundo.", "offsets": offsets}]}
                result = WhisperCppEngine.parse_result(
                    {"transcription": [item]}, 8, [(5, 9, 0, 4)])
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["text"], "Hola mundo.")
                self.assertEqual((result[0]["start"], result[0]["end"]), (5.1, 8))
                self.assertEqual(result[0]["words"], [])
                self.assertEqual(result[0]["timing_source"], "vad_phrase")

    def test_missing_tokens_use_phrase_interval(self):
        item = {"text": "Hola.", "offsets": {"from": 5000, "to": 6000}}
        result = WhisperCppEngine.parse_result(
            {"transcription": [item]}, 10, [(5, 7, 0, 2)])
        self.assertEqual(result[0]["text"], "Hola.")
        self.assertEqual(result[0]["words"], [])

    def test_invalid_tokens_do_not_invent_speech_in_vad_silence(self):
        item = {"text": "Hola.", "offsets": {"from": 2000, "to": 3000},
                "tokens": [{"text": " Hola.", "offsets": {"from": -1, "to": -1}}]}
        with self.assertRaises(RuntimeError):
            WhisperCppEngine.parse_result(
                {"transcription": [item]}, 10, [(5, 7, 0, 2)])

    def test_actual_cli_mapping_line_is_parsed(self):
        line = 'whisper_vad: vad_segment_info: orig_start: 32.48, orig_end: 44.83, vad_start: 26.30, vad_end: 38.65'
        self.assertEqual(WhisperCppEngine.parse_vad_mapping(line), (32.48, 44.83, 26.3, 38.65))


if __name__ == '__main__':
    unittest.main()
