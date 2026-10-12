"""Stem routing and boundary padding without neural inference in unit tests."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import soundfile as sf
from core import voice_separator


class SeparatorTests(unittest.TestCase):
    def test_uvr_background_first_and_added_context_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'UVR-MDX-NET-Inst_HQ_4.onnx').touch()
            source, vocals, background = [folder / name for name in ('source.wav', 'voice.wav', 'background.wav')]
            sf.write(source, np.full((1000, 2), .5), 8000, subtype='FLOAT')
            def process(sample_rate, samples):
                self.assertEqual(samples.shape, (2, 17000))
                self.assertEqual(sample_rate, 8000)
                bed = np.full((2, 17000), .2, dtype=np.float32)
                bed[:, :10] = 2000
                bed[:, -10:] = -2000
                return SimpleNamespace(sample_rate=8000, stems=[
                    SimpleNamespace(data=bed), SimpleNamespace(data=np.full((2, 17000), .3))])
            api = SimpleNamespace(
                OfflineSourceSeparationUvrModelConfig=Mock(),
                OfflineSourceSeparationModelConfig=Mock(),
                OfflineSourceSeparationConfig=Mock(return_value=SimpleNamespace(validate=lambda: True)),
                OfflineSourceSeparation=Mock(return_value=SimpleNamespace(process=process)),
            )
            with patch.object(voice_separator, 'MODEL_DIR', folder), patch.dict('sys.modules', {'sherpa_onnx': api}):
                voice_separator.separate_original_voice(str(source), str(vocals), str(background))
            bed, rate = sf.read(background, always_2d=True)
            speech, _ = sf.read(vocals, always_2d=True)
            self.assertEqual(rate, 8000)
            self.assertEqual(bed.shape, (1000, 2))
            np.testing.assert_allclose(bed, .2, atol=1e-6)
            np.testing.assert_allclose(speech, .3, atol=1e-6)

    def test_missing_uvr_model_stops_without_silent_spleeter_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(voice_separator, 'MODEL_DIR', Path(directory)):
                with self.assertRaisesRegex(RuntimeError, 'UVR-MDX-NET-Inst_HQ_4'):
                    voice_separator.separate_original_voice('unused.wav', 'voice.wav', 'bed.wav')
