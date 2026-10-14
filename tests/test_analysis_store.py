"""Persistencia de ASR y reutilización sin volver a ejecutar inferencia."""
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch
from core import analysis_store as store
from core.fusion import fuse_segments
from test_manual_translation import load_class


class AnalysisStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='análisis espacios ñ ')
        self.root = Path(self.directory.name)
        self.video = self.root/'capítulo 01.mp4'
        self.video.write_bytes(b'video de prueba')
        self.cache = patch.object(store, 'CACHE_DIR', self.root/'cache')
        self.cache.start()
        self.segments = [dict(start=1., end=2., asr_text='Hello', source_text='Hello', final_text='Hola',
            language='en', words=[dict(start=1., end=1.8, word='Hello')], translation_source='manual_json',
            tts_audio='temporal.wav', tts_error='error antiguo')]

    def tearDown(self):
        self.cache.stop()
        self.directory.cleanup()

    def test_roundtrip_preserves_analysis_translation_words_and_omits_temporary_tts(self):
        path = store.save_analysis(self.video, self.segments)
        restored = store.load_analysis(self.video)
        self.assertTrue(path.is_file())
        self.assertEqual(restored[0]['final_text'], 'Hola')
        self.assertEqual(restored[0]['words'], self.segments[0]['words'])
        self.assertEqual(restored[0]['language'], 'en')
        self.assertEqual(restored[0]['translation_source'], 'manual_json')
        self.assertNotIn('tts_audio', restored[0])
        self.assertNotIn('tts_error', restored[0])
        self.assertIn('tts_audio', self.segments[0])

    def test_export_can_be_loaded_without_default_cache(self):
        destination = self.root/'exportado ñ.json'
        store.save_analysis(self.video, self.segments, destination)
        self.assertFalse(store.cache_path(self.video).exists())
        self.assertEqual(store.load_analysis(self.video, destination)[0]['asr_text'], 'Hello')

    def test_wrong_or_changed_video_is_rejected(self):
        path = store.save_analysis(self.video, self.segments)
        other = self.root/'otro.mp4'
        other.write_bytes(b'otro video')
        with self.assertRaisesRegex(ValueError, 'otro video'):
            store.load_analysis(other, path)
        self.video.write_bytes(b'video modificado')
        with self.assertRaisesRegex(ValueError, 'modificado'):
            store.load_analysis(self.video)

    def test_invalid_and_unknown_json_or_times_are_rejected(self):
        path = store.save_analysis(self.video, self.segments)
        original = json.loads(path.read_text(encoding='utf-8'))
        for payload in ({}, dict(original, version=99), dict(original, segments=[dict(self.segments[0], end=-1)])):
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                store.load_analysis(self.video)
        path.write_text('JSON roto', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'JSON válido'):
            store.load_analysis(self.video)

    def test_failed_atomic_save_keeps_previous_analysis_and_never_overwrites_video(self):
        path = store.save_analysis(self.video, self.segments)
        before = path.read_bytes()
        with patch('core.analysis_store.os.replace', side_effect=OSError('Disco lleno')):
            with self.assertRaises(OSError):
                store.save_analysis(self.video, [dict(self.segments[0], final_text='Otra traducción')])
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'sobrescribir'):
            store.save_analysis(self.video, self.segments, self.video)
        self.assertEqual(self.video.read_bytes(), b'video de prueba')

    def window(self):
        namespace = dict(QMainWindow=object, List=list, Dict=dict, Path=Path,
            config_manager=SimpleNamespace(get=lambda key, default=None: 'manual' if key == 'translation_mode' else default),
            logger=Mock(), QMessageBox=SimpleNamespace(Yes=1, No=2, Cancel=4, question=Mock(return_value=1)),
            QTimer=Mock(), AnalyzeWorker=Mock())
        cls = load_class('gui/main_window.py', 'MainWindow', namespace)
        window = cls.__new__(cls)
        window.video_path = str(self.video)
        window.segments = []
        window.populate_table = Mock()
        for name in ('btn_analyze','btn_tts','btn_render','btn_retry_translation','progress_bar','pipeline_status',
                     'btn_load_analysis','btn_save_analysis'):
            setattr(window, name, Mock())
        return window, namespace

    def test_reuse_on_analyze_button_never_starts_asr_or_requires_ollama(self):
        store.save_analysis(self.video, self.segments)
        window, namespace = self.window()
        window.start_analysis()
        namespace['AnalyzeWorker'].assert_not_called()
        self.assertEqual(window.segments[0]['final_text'], 'Hola')
        window.btn_tts.setEnabled.assert_called_with(True)
        window.btn_render.setEnabled.assert_called_with(False)

    def test_untranslated_recovered_analysis_keeps_tts_disabled(self):
        window, _ = self.window()
        window._apply_saved_analysis([dict(self.segments[0], final_text='')])
        window.btn_tts.setEnabled.assert_called_with(False)
        window.btn_render.setEnabled.assert_called_with(False)

    def test_worker_saves_asr_before_failed_translation(self):
        asr = ModuleType('core.asr_engine')
        asr.ASREngine = Mock()
        asr.ASREngine.return_value.extract_audio_text.return_value = [dict(start=1.,end=2.,text='Hello',words=[])]
        translator = ModuleType('core.ollama_translator')
        translator.OllamaTranslator = Mock(side_effect=RuntimeError('Servicio de traducción caído'))
        namespace = dict(QThread=object, Signal=lambda *args: Mock(), List=list, os=os,
            config_manager=SimpleNamespace(get=lambda key, default=None: 'faster_whisper' if key == 'asr_engine' else default),
            get_video_info=lambda path: {'has_audio':True}, extract_audio=Mock(return_value=True),
            fuse_segments=fuse_segments, logger=logging.getLogger('analysis_store'))
        cls = load_class('gui/workers.py','AnalyzeWorker',namespace)
        worker = cls(str(self.video))
        with patch.dict('sys.modules', {'core.asr_engine':asr,'core.ollama_translator':translator}):
            worker.run()
        worker.error.emit.assert_called_once()
        restored = store.load_analysis(self.video)
        self.assertEqual(restored[0]['asr_text'], 'Hello')
        self.assertEqual(restored[0]['final_text'], '')
