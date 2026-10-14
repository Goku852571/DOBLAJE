"""Recuperación de TTS: reintentos acotados y conservación de trabajo válido."""
import json
import logging
import math
import os
import tempfile
import unittest
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from test_manual_translation import load_class
from test_audio_timing import load_source


def write_audio(path, duration):
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(b'\x01\x00' * round(duration * 48000))


def audio_duration(path):
    with wave.open(str(path), 'rb') as stream:
        return stream.getnframes() / stream.getframerate()


class TTSRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='tts recuperación ')
        self.root = Path(self.directory.name)
        self.provider = Mock()
        self.provider.synthesize.side_effect = self.synthesize
        self.provider_module = ModuleType('core.tts_provider')
        self.provider_module.SapiTTSProvider = Mock(return_value=self.provider)
        self.fitter = Mock(side_effect=self.fit)
        self.namespace = dict(QThread=object, Signal=lambda *args: Mock(), List=list, Dict=dict,
            math=math, os=os, config_manager=SimpleNamespace(get=lambda key, default=None: default),
            clear_segment_tts=load_source('media', {}).clear_segment_tts,
            get_audio_duration=audio_duration, time_stretch_audio=self.fitter,
            logger=logging.getLogger('tts_recovery'))
        self.cls = load_class('gui/workers.py', 'TTSWorker', self.namespace)
        self.segments = [{'start': 0., 'end': 1., 'final_text': 'Hola'},
                         {'start': 2., 'end': 3., 'final_text': 'Después'}]

    def tearDown(self):
        self.directory.cleanup()

    def synthesize(self, **kwargs):
        write_audio(kwargs['output_path'], .5)
        return True

    def fit(self, source, duration, output, **kwargs):
        write_audio(output, duration)
        return True

    def run_worker(self):
        worker = self.cls(self.segments, str(self.root))
        # Signals in the AST fixture are class mocks; reset per run.
        worker.error.emit.reset_mock()
        worker.finished.emit.reset_mock()
        with patch.dict('sys.modules', {'core.tts_provider': self.provider_module}), patch('time.sleep'):
            worker.run()
        return worker

    def test_transient_synthesis_error_is_retried_and_later_segments_complete(self):
        count = 0
        def synthesize(**kwargs):
            nonlocal count
            count += 1
            if count == 1:
                raise TimeoutError('Tiempo de espera agotado')
            return self.synthesize(**kwargs)
        self.provider.synthesize.side_effect = synthesize
        worker = self.run_worker()
        worker.error.emit.assert_not_called()
        self.assertEqual(self.provider.synthesize.call_count, 3)
        self.assertTrue(all(s.get('tts_audio') for s in self.segments))
        checkpoint = json.loads((self.root/'tts_progreso.json').read_text(encoding='utf-8'))
        self.assertEqual(len(checkpoint['segments']), 2)
        self.assertNotIn('api_key', checkpoint)

    def test_partial_failure_keeps_success_and_reruns_only_pending(self):
        def synthesize(**kwargs):
            if kwargs['text'] == 'Hola':
                return False
            return self.synthesize(**kwargs)
        self.provider.synthesize.side_effect = synthesize
        worker = self.run_worker()
        worker.error.emit.assert_called_once()
        self.assertEqual(self.provider.synthesize.call_count, 4)
        self.assertIn('tts_error', self.segments[0])
        self.assertNotIn('tts_audio', self.segments[0])
        saved = self.segments[1]['tts_audio']
        self.provider.synthesize.reset_mock(side_effect=True)
        self.provider.synthesize.side_effect = self.synthesize
        worker = self.run_worker()
        worker.error.emit.assert_not_called()
        self.assertEqual(self.provider.synthesize.call_count, 1)
        self.assertEqual(self.segments[1]['tts_audio'], saved)
        self.assertNotIn('tts_error', self.segments[0])

    def test_sync_retry_reuses_synthesis_without_modifying_times_on_failure(self):
        self.segments = [{'start': 1., 'end': 2., 'final_text': 'Hola'},
                         {'start': 4., 'end': 5., 'final_text': 'Después'}]
        def synthesize(**kwargs):
            write_audio(kwargs['output_path'], 2.)
            return True
        self.provider.synthesize.side_effect = synthesize
        observed = []
        def fit(source, duration, output, **kwargs):
            observed.append(duration)
            if len(observed) == 1:
                self.assertEqual(self.segments[0]['start'], 1.)
                self.assertEqual(self.segments[0]['end'], 2.)
                return False
            return self.fit(source, duration, output, **kwargs)
        self.fitter.side_effect = fit
        worker = self.run_worker()
        worker.error.emit.assert_not_called()
        self.assertEqual(observed[0], observed[1])
        self.assertEqual(self.provider.synthesize.call_count, 2)
        self.assertLessEqual(self.segments[-1]['end'], 5.)

    def test_non_retryable_fit_records_reason_without_blind_retries(self):
        def fit(source, duration, output, **kwargs):
            kwargs['fit_info'].update(error='La traducción requiere una velocidad de 3.0x.', retryable=False)
            return False
        self.fitter.side_effect = fit
        before = [(s['start'], s['end']) for s in self.segments]
        worker = self.run_worker()
        worker.error.emit.assert_called_once()
        self.assertEqual(self.fitter.call_count, 2)
        self.assertEqual(self.provider.synthesize.call_count, 2)
        self.assertEqual(before, [(s['start'], s['end']) for s in self.segments])
        self.assertIn('acorte', self.segments[0]['tts_error'])

    def test_empty_audio_retried_and_changed_text_or_voice_invalidates_cache(self):
        count = 0
        def synthesize(**kwargs):
            nonlocal count
            count += 1
            if count == 1:
                Path(kwargs['output_path']).write_bytes(b'')
                return True
            return self.synthesize(**kwargs)
        self.provider.synthesize.side_effect = synthesize
        self.run_worker().error.emit.assert_not_called()
        self.assertEqual(self.provider.synthesize.call_count, 3)
        self.segments[0]['final_text'] = 'Texto cambiado'
        self.provider.synthesize.reset_mock()
        self.run_worker().error.emit.assert_not_called()
        self.assertEqual(self.provider.synthesize.call_count, 1)
        self.namespace['config_manager'].get = lambda key, default=None: 'Otra voz' if key == 'tts_voice_name' else default
        self.provider.synthesize.reset_mock()
        self.run_worker().error.emit.assert_not_called()
        self.assertEqual(self.provider.synthesize.call_count, 2)

    def test_invalid_times_do_not_call_provider_or_block_following_segment(self):
        self.segments[0]['start'] = float('nan')
        worker = self.run_worker()
        worker.error.emit.assert_called_once()
        self.assertEqual(self.provider.synthesize.call_count, 1)
        self.assertIn('Intervalo inválido', self.segments[0]['tts_error'])
        self.assertTrue(self.segments[1].get('tts_audio'))

    def test_auth_failure_stops_batch_without_charging_or_retrying_remaining_segments(self):
        self.provider.synthesize.side_effect = RuntimeError('OpenSpeaker HTTP 402: Créditos insuficientes.')
        worker = self.run_worker()
        worker.error.emit.assert_called_once()
        self.assertEqual(self.provider.synthesize.call_count, 1)
        self.assertIn('créditos', worker.error.emit.call_args.args[0])
        self.assertTrue((self.root/'tts_progreso.json').is_file())

    def test_repeated_provider_failure_stops_after_five_segments(self):
        self.segments = [dict(start=float(i*2), end=float(i*2+1), final_text=f'Texto {i}') for i in range(20)]
        self.provider.synthesize.side_effect = RuntimeError('Servicio no disponible')
        worker = self.run_worker()
        worker.error.emit.assert_called_once()
        self.assertEqual(self.provider.synthesize.call_count, 15)
        self.assertEqual(sum(bool(s.get('tts_error')) for s in self.segments), 5)

    def test_oversized_phrase_is_repaired_and_committed_only_after_valid_fit(self):
        self.segments = [dict(start=0., end=1., source_text='看', final_text='Una frase demasiado larga')]
        def fit(source, duration, output, **kwargs):
            if self.fitter.call_count == 1:
                kwargs['fit_info'].update(error='La traducción requiere una velocidad de 8.92x.', retryable=False)
                return False
            return self.fit(source, duration, output, **kwargs)
        self.fitter.side_effect = fit
        with patch.object(self.cls, '_repair_segment_text', return_value='Mira.') as repair:
            worker = self.run_worker()
        worker.error.emit.assert_not_called()
        repair.assert_called_once()
        self.assertEqual(self.segments[0]['final_text'], 'Mira.')
        self.assertEqual(self.segments[0]['tts_text'], 'Mira.')
        self.assertEqual(self.provider.synthesize.call_count, 2)

    def test_failed_repair_does_not_overwrite_original_translation_or_times(self):
        self.segments = [dict(start=0., end=1., source_text='看', final_text='Texto original')]
        def fit(source, duration, output, **kwargs):
            kwargs['fit_info'].update(error='La traducción requiere una velocidad de 8.92x.', retryable=False)
            return False
        self.fitter.side_effect = fit
        with patch.object(self.cls, '_repair_segment_text', side_effect=['Mira.', None]):
            worker = self.run_worker()
        worker.error.emit.assert_called_once()
        self.assertEqual(self.segments[0]['final_text'], 'Texto original')
        self.assertEqual((self.segments[0]['start'], self.segments[0]['end']), (0.,1.))
        self.assertNotIn('tts_audio', self.segments[0])

    def test_checkpoint_loader_checks_analysis_and_marks_missing_audio_pending(self):
        from core.tts_recovery import load_tts_progress
        for index, seg in enumerate(self.segments):
            seg['source_text'] = f'Original {index}'
        self.run_worker().error.emit.assert_not_called()
        path = self.root/'tts_progreso.json'
        restored, valid = load_tts_progress(self.segments, path)
        self.assertEqual(valid, 2)
        self.assertEqual(restored[1]['tts_audio'], self.segments[1]['tts_audio'])
        wrong = [dict(s) for s in self.segments]
        wrong[0]['source_text'] = 'Otro video'
        with self.assertRaises(ValueError):
            load_tts_progress(wrong, path)
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['segments'][0]['tts_audio'] = str(self.root/'no_existe.wav')
        path.write_text(json.dumps(payload), encoding='utf-8')
        restored, valid = load_tts_progress(self.segments, path)
        self.assertEqual(valid, 1)
        self.assertNotIn('tts_audio', restored[0])
        self.assertIn('tts_error', restored[0])
