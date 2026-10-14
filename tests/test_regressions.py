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


def ollama_translator_class():
    tree = ast.parse((ROOT / 'core/ollama_translator.py').read_text(encoding='utf-8'))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'OllamaTranslator'
    )
    namespace = dict(
        json=json, re=re, List=list, Dict=dict,
        ENGLISH_MARKERS=ast.literal_eval(next(node.value for node in tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ENGLISH_MARKERS" for target in node.targets))),
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

    def test_empty_transcription_has_no_segments(self):
        self.assertEqual(fuse_segments([]), [])

    def test_audio_segments_preserve_text_and_timing(self):
        words = [{"word": "Hola", "start": 1.2, "end": 1.8}]
        result = fuse_segments([
            dict(start=3, end=4, text="Adiós"),
            dict(start=1, end=2, text="Hola", words=words, language="es"),
        ])
        self.assertEqual([s['start'] for s in result], [1, 3])
        self.assertEqual(result[0]['source_text'], 'Hola')
        self.assertEqual(result[0]['final_text'], 'Hola')
        self.assertEqual(result[0]['words'], words)
        self.assertEqual(result[0]['language'], 'es')

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
        gui_source = (ROOT / 'gui/main_window.py').read_text(encoding='utf-8')
        self.assertIn('español latino neutro', source)
        self.assertIn('prohibido responder en inglés', source.lower())
        self.assertIn('No uses inglés', source)
        self.assertIn('doblaje profesional', source)
        self.assertIn('máximo de palabras', source)
        self.assertIn('nombres latinos', source)
        self.assertIn('yuanes', source)
        self.assertIn('dólares', source)
        self.assertIn('nombres latinos', gui_source)
        self.assertIn('yuanes', gui_source)
        self.assertIn('dólares', gui_source)

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
        self.assertIn('QFileDialog.getOpenFileName', source)
        self.assertIn('QPlainTextEdit', source)
        self.assertIn('text_edit.toPlainText()', source)


if __name__ == '__main__':
    unittest.main()
