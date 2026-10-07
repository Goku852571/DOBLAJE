"""Dependency-free contracts; heavy inference is mocked explicitly."""
import ast
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from core.translation_engine import TranslationEngine
from core.fusion import fuse_segments

ROOT = Path(__file__).resolve().parents[1]


def ocr_class():
    tree = ast.parse((ROOT / 'core/ocr_engine.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    # Import neither EasyOCR nor Qt for pure geometry/timing tests.
    namespace = dict(List=list, Dict=dict, Optional=__import__('typing').Optional,
                     Callable=__import__('typing').Callable, Tuple=tuple,
                     np=SimpleNamespace(ndarray=object, mean=lambda xs: sum(xs)/len(xs)),
                     fuzz=SimpleNamespace(ratio=lambda a, b: 100 if a == b else 0))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<ocr>', 'exec'), namespace)
    return namespace['OCREngine']


def ollama_translator_class():
    tree = ast.parse((ROOT / 'core/ollama_translator.py').read_text(encoding='utf-8'))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'OllamaTranslator'
    )
    namespace = dict(
        json=json, re=re, List=list, Dict=dict,
        Optional=__import__('typing').Optional,
        logger=SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<ollama_translator>', 'exec'), namespace)
    return namespace['OllamaTranslator']


class RegressionTests(unittest.TestCase):
    def test_sources_compile(self):
        for folder in ('core', 'gui', 'utils', 'tests'):
            for path in (ROOT / folder).glob('*.py'):
                if not path.name.startswith('._'):
                    compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    def test_reading_order_tolerates_vertical_box_jitter(self):
        def box(x, y, text):
            return dict(bbox=[[x,y],[x+30,y],[x+30,y+20],[x,y+20]], text=text)
        items = [box(80,10,'right'), box(10,13,'left'), box(10,50,'next')]
        ordered = ocr_class()._reading_order(items)
        self.assertEqual([i['text'] for i in ordered], ['left','right','next'])

    def test_subtitle_gap_is_not_bridged(self):
        engine = ocr_class()()
        readings = [dict(time=t, text='你好', prob=0.9) for t in (0, .25, 1)]
        result = engine._consolidate_results(readings, .25)
        self.assertEqual([(s['start'],s['end']) for s in result], [(0,.5),(1,1.25)])

    def test_consensus_preserves_observed_phrase(self):
        readings = [dict(text='你好世界',prob=.8),dict(text='你好世界',prob=.9),
                    dict(text='你好',prob=.95)]
        self.assertEqual(ocr_class()._vote_words(readings), '你好世界')

    def test_silent_video_keeps_ocr(self):
        result = fuse_segments([dict(start=0,end=1,text='你好')], [])
        self.assertEqual(result[0]['final_text'], '你好')

    def test_blank_translation_never_loads_model(self):
        engine = TranslationEngine()
        with patch.object(engine, 'initialize') as init:
            self.assertEqual(engine.translate('  '), '')
            init.assert_not_called()

    def test_repeated_text_uses_cache(self):
        engine = TranslationEngine()
        engine.cache['你好'] = 'Hola'
        with patch.object(engine, 'initialize') as init:
            self.assertEqual(engine.translate(' 你好 '), 'Hola')
            init.assert_not_called()

    def test_tts_mix_does_not_auto_normalize_many_segments(self):
        media_source = (ROOT / 'core/media.py').read_text(encoding='utf-8')
        self.assertIn('normalize=0[tts_mix]', media_source)
        self.assertIn('amix=inputs=2:duration=first:dropout_transition=0:normalize=0', media_source)

    def test_contextual_translation_parser_accepts_fenced_json(self):
        response = '```json\n[{"id":2,"text":"segunda"},{"id":1,"text":"primera"}]\n```'
        cls = ollama_translator_class()
        translations = cls._parse_segment_translations(object(), response, 2)
        self.assertEqual(translations, ['primera', 'segunda'])

    def test_translation_prompts_require_spanish_not_english(self):
        source = (ROOT / 'core/ollama_translator.py').read_text(encoding='utf-8')
        self.assertIn('español latino neutro', source)
        self.assertIn('prohibido responder en inglés', source.lower())
        self.assertIn('No uses inglés', source)
        self.assertIn('doblaje profesional', source)
        self.assertIn('máximo de palabras', source)

    def test_missing_contextual_lines_use_individual_translation(self):
        cls = ollama_translator_class()
        translator = object.__new__(cls)
        translator.translate_with_qwen = lambda text: f'español:{text}'
        result = cls._fill_missing_translations(
            translator,
            ['hello', 'world', ''],
            ['hola', '', '']
        )
        self.assertEqual(result, ['hola', 'español:world', ''])

    def test_gui_exposes_json_script_import_export_contract(self):
        source = (ROOT / 'gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('def copy_script_json', source)
        self.assertIn('def import_translation_json', source)
        self.assertIn('"source_script"', source)
        self.assertIn('"translated_script"', source)
        self.assertIn('"source_text"', source)
        self.assertIn('QApplication.clipboard().setText', source)
        self.assertIn('def export_script_json', source)
        self.assertIn('QFileDialog.getOpenFileName', source)
        self.assertIn('QFileDialog.getSaveFileName', source)


if __name__ == '__main__':
    unittest.main()
