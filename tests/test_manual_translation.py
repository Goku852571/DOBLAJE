"""Manual JSON round trip, atomic imports and analysis without Ollama."""
import ast
import copy
import hashlib
import json
import logging
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from core.fusion import fuse_segments
from test_audio_timing import load_source

ROOT = Path(__file__).resolve().parents[1]


def load_class(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class ManualTranslationTests(unittest.TestCase):
    def setUp(self):
        namespace = dict(QMainWindow=object, List=list, Dict=dict, json=json,
                         hashlib=hashlib, clear_segment_tts=load_source('media', {}).clear_segment_tts)
        cls = load_class('gui/main_window.py', 'MainWindow', namespace)
        self.window = cls.__new__(cls)
        self.window.segments = [
            dict(start=5.0, end=7.0, source_text='Later', asr_text='Later', final_text='old',
                 tts_audio='old.wav', tts_text='old'),
            dict(start=1.0, end=2.0, source_text='Hello', asr_text='Hello', final_text='old'),
        ]
        self.window.translation_mode = SimpleNamespace(currentData=lambda: 'manual')

    def test_export_order_template_and_import_reordered_ids(self):
        payload = self.window._script_json_payload()
        self.assertEqual([s['source_text'] for s in payload['source_script']], ['Hello', 'Later'])
        self.assertEqual(payload['translated_script'], [{'id': 1, 'text': ''}, {'id': 2, 'text': ''}])
        payload['translated_script'] = [{'id': 2, 'text': 'Después'}, {'id': 1, 'text': 'Hola'}]
        before_times = [(s['start'], s['end']) for s in self.window.segments]
        self.assertEqual(self.window._apply_translation_items(
            self.window._parse_translation_json(json.dumps(payload))), 2)
        self.assertEqual([s['final_text'] for s in self.window.segments], ['Después', 'Hola'])
        self.assertEqual([(s['start'], s['end']) for s in self.window.segments], before_times)
        self.assertNotIn('tts_audio', self.window.segments[0])
        self.assertTrue(all(s['translation_source'] == 'manual_json' for s in self.window.segments))
        self.assertFalse(self.window._can_auto_translate())

    def test_wrong_script_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'otro guion'):
            self.window._parse_translation_json(json.dumps({
                'script_id': 'wrong', 'translated_script': [{'id': 1, 'text': 'Hola'}]}))

    def test_invalid_import_is_atomic(self):
        cases = [
            [{'id': 1, 'text': 'Hola'}],
            [{'id': 1, 'text': 'Hola'}, {'id': 1, 'text': 'Después'}],
            [{'id': 1, 'text': 'Hola'}, {'id': 3, 'text': 'Después'}],
            [{'id': 1, 'text': 'Hola'}, {'id': 2, 'text': ' '}],
            [{'id': True, 'text': 'Hola'}, {'id': 2, 'text': 'Después'}],
            [{'id': 1, 'text': 'Hola'}, {'id': 2, 'text': None}],
        ]
        for items in cases:
            with self.subTest(items=items):
                before = copy.deepcopy(self.window.segments)
                with self.assertRaises(ValueError):
                    self.window._apply_translation_items(items)
                self.assertEqual(self.window.segments, before)

    def test_imported_text_is_protected_even_in_automatic_mode(self):
        self.window._apply_translation_items([{'id': 1, 'text': 'Hola'}, {'id': 2, 'text': 'Después'}])
        self.window.translation_mode = SimpleNamespace(currentData=lambda: 'automatic')
        self.assertFalse(self.window._can_auto_translate())

    def test_manual_analysis_finishes_without_constructing_translator(self):
        asr_module = ModuleType('core.asr_engine')
        asr_module.ASREngine = Mock()
        asr_module.ASREngine.return_value.extract_audio_text.return_value = [
            dict(start=1, end=2, text='Hello', words=[]),
        ]
        translator_module = ModuleType('core.ollama_translator')
        translator_module.OllamaTranslator = Mock(side_effect=AssertionError('Ollama called'))
        namespace = dict(QThread=object, Signal=lambda *args: Mock(), List=list,
                         os=__import__('os'), config_manager=SimpleNamespace(
                             get=lambda key, default=None: False if key == 'ocr_enabled' else default),
                         get_video_info=lambda path: {'has_audio': True}, extract_audio=Mock(return_value=True),
                         fuse_segments=fuse_segments, logger=logging.getLogger('manual_test'))
        worker_cls = load_class('gui/workers.py', 'AnalyzeWorker', namespace)
        worker = worker_cls('video.mp4', manual_translation=True)
        with patch.dict('sys.modules', {'core.asr_engine': asr_module,
                                       'core.ollama_translator': translator_module}):
            worker.run()
        worker.error.emit.assert_not_called()
        worker.finished.emit.assert_called_once()
        segments = worker.finished.emit.call_args.args[0]
        self.assertEqual(segments[0]['source_text'], 'Hello')
        self.assertEqual(segments[0]['final_text'], '')
        translator_module.OllamaTranslator.assert_not_called()

    def test_tts_uses_imported_translation(self):
        self.window._apply_translation_items([{'id': 1, 'text': 'Hola'}, {'id': 2, 'text': 'Después'}])
        provider = Mock()
        provider.synthesize.return_value = True
        module = ModuleType('core.tts_provider')
        module.SapiTTSProvider = Mock(return_value=provider)
        namespace = dict(QThread=object, Signal=lambda *args: Mock(), List=list, Dict=dict,
                         math=__import__('math'), os=SimpleNamespace(path=SimpleNamespace(
                             join=__import__('os').path.join, exists=lambda path: True)),
                         config_manager=SimpleNamespace(get=lambda key, default=None: default),
                         clear_segment_tts=load_source('media', {}).clear_segment_tts,
                         get_audio_duration=lambda path: 1.0, time_stretch_audio=Mock(return_value=True),
                         logger=logging.getLogger('manual_tts_test'))
        worker_cls = load_class('gui/workers.py', 'TTSWorker', namespace)
        worker = worker_cls(self.window.segments, 'temporary')
        with patch.dict('sys.modules', {'core.tts_provider': module}):
            worker.run()
        worker.error.emit.assert_not_called()
        self.assertEqual([call.kwargs['text'] for call in provider.synthesize.call_args_list],
                         ['Después', 'Hola'])


if __name__ == '__main__':
    unittest.main()
