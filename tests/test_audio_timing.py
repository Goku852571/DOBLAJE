"""Behavior checks with mocked inference and real, measurable FFmpeg audio."""
import ast
import logging
import math
import shutil
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
RATE = 48000


def load_source(name, replacements):
    tree = ast.parse((ROOT / "core" / f"{name}.py").read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and n.module in
        ("utils.logger", "faster_whisper", "faster_whisper.vad")
    )]
    namespace = dict(logger=logging.getLogger("timing_tests"), **replacements)
    exec(compile(tree, f"<test:{name}>", "exec"), namespace)
    return SimpleNamespace(**namespace)


def word(text, start, end):
    return SimpleNamespace(word=text, start=start, end=end, probability=0.95)


def segment(words, start=0, end=8, text="coarse segment"):
    return SimpleNamespace(words=words, start=start, end=end, text=text, avg_logprob=-0.1)


def write_wav(path, parts, rate=RATE):
    samples = array("h")
    for duration, frequency in parts:
        count = round(duration * rate)
        samples.extend(round(8000 * math.sin(2 * math.pi * frequency * i / rate))
                       if frequency else 0 for i in range(count))
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        samples = array("h", wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    return rate, samples


class ASRTimingTests(unittest.TestCase):
    def setUp(self):
        self.audio = [0] * 160000
        self.source = load_source("asr_engine", {
            "WhisperModel": Mock(), "decode_audio": Mock(return_value=self.audio),
            "get_speech_timestamps": Mock(return_value=[dict(start=16000, end=144000)]),
            "VadOptions": SimpleNamespace,
        })
        self.engine = self.source.ASREngine()
        self.engine.model = Mock()

    def test_language_is_detected_by_transcription_and_kept_original(self):
        for language, text in [("zh", "你好"), ("en", " Hello")]:
            with self.subTest(language=language):
                order = []
                def transcribe(audio, **options):
                    order.append("transcribe")
                    self.assertIs(audio, self.audio)
                    self.assertIsNone(options["language"])
                    self.assertEqual(options["task"], "transcribe")
                    self.assertTrue(options["word_timestamps"])
                    return iter([segment([word(text, 2.25, 2.85)])]), SimpleNamespace(
                        language=language, language_probability=0.96)
                self.engine.model.transcribe.side_effect = transcribe
                result = self.engine.extract_audio_text("audio.wav")
                self.assertEqual(order, ["transcribe"])
                self.engine.model.detect_language.assert_not_called()
                self.assertEqual((result[0]["start"], result[0]["end"]), (2.25, 2.85))
                self.assertEqual(result[0]["text"], text.strip())

    def test_spoken_boundaries_exclude_segment_padding_and_keep_pause(self):
        result = self.engine._aligned_segments(segment([
            word(" Hello", 1.25, 1.5), word(" there", 1.5, 1.9),
            word(" Later", 3.5, 3.9), word(" on.", 3.9, 4.1),
        ]), 10)
        self.assertEqual([(s["start"], s["end"]) for s in result], [(1.25, 1.9), (3.5, 4.1)])
        self.assertEqual([s["text"] for s in result], ["Hello there", "Later on."])

    def test_chinese_sentences_split_at_punctuation(self):
        result = self.engine._aligned_segments(segment([
            word("你好。", 1, 2), word("再见！", 2, 3),
        ]), 10)
        self.assertEqual([s["text"] for s in result], ["你好。", "再见！"])

    def test_small_overlaps_do_not_move_later_dialogue_earlier(self):
        self.engine.model.transcribe.return_value = (iter([
            segment([word(" Hello.", 1, 2)]),
            segment([word(" Again.", 1.98, 3)]),
        ]), SimpleNamespace(language="en", language_probability=1.0))
        result = self.engine.extract_audio_text("audio.wav", language="en")
        self.assertEqual([(s["start"], s["end"]) for s in result], [(1, 2), (2, 3)])
        self.engine.model.detect_language.assert_not_called()

    def test_silence_is_not_transcribed(self):
        self.source.get_speech_timestamps.return_value = []
        self.assertEqual(self.engine.extract_audio_text("silent.wav"), [])
        self.engine.model.detect_language.assert_not_called()
        self.engine.model.transcribe.assert_not_called()


class AudioTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            candidates = list((ROOT / ".python-packages/imageio_ffmpeg/binaries").glob("ffmpeg*.exe"))
            if not candidates:
                raise unittest.SkipTest("FFmpeg no está disponible para las pruebas de audio.")
            ffmpeg = str(candidates[0])
        cls.media = load_source("media", {})
        # Functions use the namespace created by exec, rather than this wrapper.
        cls.media.run_cmd.__globals__["_ffmpeg_binary"] = lambda: ffmpeg

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="timing_test_")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_edge_silence_removed_without_removing_internal_pause(self):
        raw, trimmed = self.folder / "raw.wav", self.folder / "trimmed.wav"
        write_wav(raw, [(0.3, 0), (0.2, 440), (0.15, 0), (0.2, 440), (0.4, 0)])
        details = self.media._trim_pcm_silence(str(raw), str(trimmed))
        self.assertAlmostEqual(details["duration"], 0.56, delta=0.001)
        self.assertAlmostEqual(details["leading_silence"], 0.295, delta=0.001)
        rate, data = read_wav(trimmed)
        self.assertTrue(all(s == 0 for s in data[round(0.22 * rate):round(0.33 * rate)]))

    def test_accelerated_voice_is_measured_and_fits_without_tail_truncation(self):
        raw, fitted = self.folder / "raw.wav", self.folder / "fitted.wav"
        write_wav(raw, [(0.3, 0), (0.7, 440), (0.2, 0), (0.7, 880), (0.3, 0)], rate=22050)
        details = {}
        self.assertTrue(self.media.time_stretch_audio(str(raw), 0.55, str(fitted), fit_info=details))
        rate, data = read_wav(fitted)
        self.assertEqual(len(data), round(0.55 * rate))
        self.assertGreater(details["tts_speed"], 2.8)
        self.assertAlmostEqual(details["tts_speech_duration"], 1.61, delta=0.002)
        self.assertLess(next(i for i, s in enumerate(data) if abs(s) > 100) / rate, 0.015)
        # The last burst is 880 Hz: verify it survives the fit, rather than only
        # testing the output container duration.
        active = [i for i, s in enumerate(data) if abs(s) > 100]
        tail = data[active[-1] - round(0.05 * rate):active[-1]]
        crossings = sum(a <= 0 < b for a, b in zip(tail, tail[1:]))
        self.assertAlmostEqual(crossings / 0.05, 880, delta=60)

    def test_short_voice_keeps_natural_rate_and_only_pads_the_end(self):
        raw, fitted = self.folder / "raw.wav", self.folder / "fitted.wav"
        write_wav(raw, [(0.25, 0), (0.3, 440), (0.2, 0)])
        details = {}
        self.assertTrue(self.media.time_stretch_audio(str(raw), 1.0, str(fitted), fit_info=details))
        rate, data = read_wav(fitted)
        self.assertEqual(details["tts_speed"], 1.0)
        self.assertEqual(len(data), rate)
        self.assertLess(next(i for i, s in enumerate(data) if abs(s) > 100) / rate, 0.015)
        self.assertTrue(all(s == 0 for s in data[round(0.32 * rate):]))

    def test_duration_matching_slows_speech_moderately_and_keeps_onset(self):
        raw, fitted = self.folder / 'match_raw.wav', self.folder / 'match_fitted.wav'
        write_wav(raw, [(0.2, 0), (0.82, 440), (0.2, 0)])
        details = {}
        self.assertTrue(self.media.time_stretch_audio(str(raw), 0.9, str(fitted),
                                                    fit_info=details, match_duration=True))
        rate, data = read_wav(fitted)
        active = [i for i, sample in enumerate(data) if abs(sample) > 100]
        self.assertGreaterEqual(details['tts_speed'], 0.85)
        self.assertLess(details['tts_speed'], 1.0)
        self.assertLess(active[0] / rate, 0.015)
        self.assertGreater(active[-1] / rate, 0.85)
        self.assertEqual(len(data), round(0.9 * rate))

    def test_mix_preserves_stereo_background_channels(self):
        import numpy as np
        # Use distinct left/right frequencies to detect unwanted mono downmix.
        original, voice, output = [self.folder / name for name in ('stereo.wav', 'voice.wav', 'stereo_mix.wav')]
        left = array('h', (round(2000 * math.sin(2 * math.pi * 330 * i / RATE)) for i in range(2 * RATE)))
        right = array('h', (round(2000 * math.sin(2 * math.pi * 660 * i / RATE)) for i in range(2 * RATE)))
        stereo = array('h', (value for pair in zip(left, right) for value in pair))
        with wave.open(str(original), 'wb') as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(stereo.tobytes())
        write_wav(voice, [(0.1, 440)])
        self.assertTrue(self.media.build_dubbed_audio(
            [dict(start=1, end=1.1, tts_audio=str(voice))], str(original), 2,
            str(output), str(self.folder), original_volume=1, preserve_stereo=True))
        with wave.open(str(output), 'rb') as wav:
            self.assertEqual(wav.getnchannels(), 2)
            result = np.frombuffer(wav.readframes(RATE // 2), dtype='<i2').reshape(-1, 2)
        # First half-second contains only background: its channels and gain survive.
        self.assertLess(np.max(np.abs(result[:, 0].astype(float) - np.asarray(left[:RATE // 2]))), 5)
        self.assertLess(np.max(np.abs(result[:, 1].astype(float) - np.asarray(right[:RATE // 2]))), 5)

    def test_unintelligibly_fast_translation_is_rejected_without_chopping(self):
        raw, output = self.folder / 'too_long.wav', self.folder / 'rejected.wav'
        write_wav(raw, [(1.5, 440)])
        with self.assertLogs('timing_tests', level='ERROR') as logs:
            self.assertFalse(self.media.time_stretch_audio(str(raw), 0.3, str(output), max_speed=2.0))
        self.assertFalse(output.exists())
        self.assertIn('Acorte el texto', '\n'.join(logs.output))

    def test_background_is_unchanged_even_when_loud_dubbing_requires_limiting(self):
        import numpy as np
        import soundfile as sf
        bed, voice, output = [self.folder / name for name in ('loud_bed.wav', 'loud_dub.wav', 'preserved.wav')]
        background = np.tile([.85, .30], (2 * RATE, 1))
        dubbing = np.zeros((2 * RATE, 1))
        dubbing[RATE // 2:3 * RATE // 2] = .8
        sf.write(bed, background, RATE, subtype='FLOAT')
        sf.write(voice, dubbing, RATE, subtype='FLOAT')
        self.assertTrue(self.media._mix_preserving_background(
            str(bed), str(voice), 2, str(output), str(self.folder), 1, 1, True))
        mixed, rate = sf.read(output, always_2d=True)
        self.assertEqual(rate, RATE)
        self.assertEqual(mixed.shape, background.shape)
        self.assertLessEqual(np.max(np.abs(mixed)), 1.000001)
        # The channel difference isolates the bed: voice is identical in both.
        # A global limiter would reduce this difference during the overlap.
        np.testing.assert_allclose(mixed[:, 0] - mixed[:, 1], .55, atol=1e-6)
        np.testing.assert_allclose(mixed[:RATE // 2], background[:RATE // 2], atol=1e-6)
        self.assertGreater(mixed[RATE, 0], .99)

    def test_background_preserving_mix_keeps_requested_voice_gain_without_auto_boost(self):
        import numpy as np
        import soundfile as sf
        bed, voice, output = [self.folder / name for name in ('quiet_bed.wav', 'quiet_dub.wav', 'quiet_mix.wav')]
        sf.write(bed, np.zeros(RATE), RATE, subtype='FLOAT')
        sf.write(voice, np.full(RATE, .1), RATE, subtype='FLOAT')
        self.assertTrue(self.media._mix_preserving_background(
            str(bed), str(voice), 1, str(output), str(self.folder), 1, .5, False))
        mixed, _ = sf.read(output)
        np.testing.assert_allclose(mixed, .05, atol=1e-6)

    def test_mix_places_voices_at_absolute_times_even_if_original_is_shorter(self):
        raw, first, second = [self.folder / name for name in ("raw.wav", "first.wav", "second.wav")]
        write_wav(raw, [(0.22, 0), (0.5, 440), (0.18, 0)], rate=22050)
        self.assertTrue(self.media.time_stretch_audio(str(raw), 0.6, str(first)))
        self.assertTrue(self.media.time_stretch_audio(str(raw), 0.45, str(second)))
        original, output = self.folder / "original.wav", self.folder / "mix.wav"
        write_wav(original, [(4, 0)])
        windows = [(1.234567, 1.834567, first), (4.200113, 4.650113, second)]
        segments = [dict(start=start, end=end, tts_audio=str(path)) for start, end, path in windows]
        self.assertTrue(self.media.build_dubbed_audio(segments, str(original), 5.0,
                                                    str(output), str(self.folder), original_volume=0))
        rate, data = read_wav(output)
        self.assertEqual(len(data), 5 * rate)
        for start, end, _ in windows:
            a, b = round(start * rate), round(end * rate)
            active = [i for i in range(a, b) if abs(data[i]) > 100]
            self.assertTrue(active)
            self.assertLess(active[0] / rate - start, 0.015)
            self.assertTrue(all(s == 0 for s in data[a - 100:a]))
            self.assertTrue(all(s == 0 for s in data[b:b + 100]))
        self.assertTrue(all(s == 0 for s in data[round(2 * rate):round(4.2 * rate)]))

    def test_silent_tts_is_rejected(self):
        raw = self.folder / "silent.wav"
        write_wav(raw, [(1, 0)])
        self.assertFalse(self.media.time_stretch_audio(str(raw), 1, str(self.folder / "out.wav")))

    def test_video_extraction_preserves_delayed_audio_track(self):
        source, video, decoded = [self.folder / name for name in ("tone.wav", "delayed.mkv", "decoded.wav")]
        write_wav(source, [(1.0, 440)])
        self.media.run_cmd([
            self.media.run_cmd.__globals__["_ffmpeg_binary"](), "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:r=25:d=3",
            "-itsoffset", "1.2", "-i", str(source),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(video),
        ], capture_output=True, text=True, check=True)
        self.assertTrue(self.media.extract_audio(str(video), str(decoded)))
        rate, data = read_wav(decoded)
        onset = next(i for i, s in enumerate(data) if abs(s) > 100) / rate
        self.assertAlmostEqual(onset, 1.2, delta=0.005)

    def test_final_mp4_keeps_voice_time_after_aac_encoding(self):
        source, raw, fitted, mixed, output, decoded = [self.folder / name for name in
            ("source.mp4", "raw.wav", "fitted.wav", "mixed.wav", "output.mp4", "decoded.wav")]
        self.media.run_cmd([
            self.media.run_cmd.__globals__["_ffmpeg_binary"](), "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:r=25:d=3",
            "-c:v", "mpeg4", str(source),
        ], capture_output=True, text=True, check=True)
        write_wav(raw, [(0.3, 0), (0.6, 440), (0.2, 0)])
        write_wav(mixed, [(1.25, 0)])
        self.assertTrue(self.media.time_stretch_audio(str(raw), 0.65, str(fitted)))
        # Create a PCM timeline first and only encode to AAC in the final MP4.
        original = self.folder / "original.wav"
        write_wav(original, [(3, 0)])
        self.assertTrue(self.media.build_dubbed_audio(
            [dict(start=1.25, end=1.9, tts_audio=str(fitted))],
            str(original), 3, str(mixed), str(self.folder), original_volume=0,
        ))
        self.assertTrue(self.media.render_final_video(str(source), str(mixed), str(output)))
        self.assertTrue(self.media.extract_audio(str(output), str(decoded)))
        rate, data = read_wav(decoded)
        active = [i for i, s in enumerate(data) if abs(s) > 100]
        self.assertAlmostEqual(active[0] / rate, 1.255, delta=0.015)
        self.assertLessEqual(active[-1] / rate, 1.9 + 0.015)


    def test_old_voice_is_invalidated_after_translation_changes(self):
        entry = dict(start=1, end=2, final_text="nuevo", tts_audio="old.wav",
                     tts_text="antiguo", tts_start=1, tts_end=2)
        self.media.clear_segment_tts(entry)
        self.assertEqual(entry, dict(start=1, end=2, final_text="nuevo"))

    def test_mix_refuses_audio_generated_for_different_times_or_text(self):
        path = self.folder / "old.wav"
        write_wav(path, [(1, 440)])
        entry = dict(start=1, end=2, final_text="nuevo", tts_audio=str(path),
                     tts_text="antiguo", tts_start=0, tts_end=1)
        with self.assertLogs("timing_tests", level="ERROR"):
            self.assertFalse(self.media.build_dubbed_audio(
                [entry], str(path), 3, str(self.folder / "out.wav"), str(self.folder),
            ))


if __name__ == "__main__":
    unittest.main()
