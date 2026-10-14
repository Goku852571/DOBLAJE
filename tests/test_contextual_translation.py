"""Traducción contextual acotada y recuperación de respuestas incompletas."""
import json
import unittest
from unittest.mock import Mock, patch
from core.ollama_translator import OllamaTranslator


class ContextualTranslationTests(unittest.TestCase):
    def segments(self, count):
        return [dict(start=i*3, end=i*3+3, source_text=f'原文{i}', asr_text=f'原文{i}') for i in range(count)]

    def test_1738_segments_are_bounded_ordered_with_context_and_progress(self):
        translator = OllamaTranslator()
        calls = []
        def batch(segments, context):
            calls.append((segments, context))
            return [f"Español {s['source_text'][2:]}" for s in segments]
        progress = []
        with patch.object(translator, '_translate_contextual_batch', side_effect=batch):
            result = translator.translate_segments_contextual(self.segments(1738), lambda *args: progress.append(args))
        self.assertEqual(len(calls), 73)
        self.assertTrue(all(len(segments)<=24 for segments, _ in calls))
        self.assertEqual(result[0], 'Español 0')
        self.assertEqual(result[-1], 'Español 1737')
        self.assertIn('Español 23', calls[1][1])
        self.assertEqual(progress[-1][:2], (1738,1738))

    def test_numbered_missing_ids_do_not_shift_neighbor_translations(self):
        translator = OllamaTranslator()
        self.assertEqual(translator._parse_segment_translations('1. Uno\n3. Tres',3), ['Uno','','Tres'])
        for response in ('[{"id":1,"text":"Uno"},{"id":1,"text":"Otro"}]',
                         '[{"id":9,"text":"Otro"}]', '[{"id":1,"text":null}]'):
            with self.assertRaises(ValueError):
                translator._parse_segment_translations(response,2)

    def test_partial_good_translation_survives_bad_retry_only_missing_line_uses_fallback(self):
        translator = OllamaTranslator()
        with patch.object(translator, '_post_generate', side_effect=[json.dumps([{'id':1,'text':'Hola'}]), 'respuesta rota']) as post, patch.object(translator, 'translate_with_qwen', return_value='Adiós') as fallback:
            result = translator.translate_segments_contextual(self.segments(2))
        self.assertEqual(result, ['Hola','Adiós'])
        self.assertEqual(fallback.call_count,1)
        self.assertLessEqual(post.call_args.kwargs['num_predict'],3072)
        self.assertIsInstance(post.call_args.kwargs['response_format'],dict)

    def test_fallback_failure_leaves_pending_without_losing_successful_lines(self):
        translator = OllamaTranslator()
        with patch.object(translator, '_post_generate', return_value='inválido'), patch.object(translator, 'translate_with_qwen', side_effect=RuntimeError('sin conexión')), patch.object(translator, 'translate', side_effect=RuntimeError('sin conexión')):
            self.assertEqual(translator.translate_segments_contextual(self.segments(2)), ['', ''])
        self.assertFalse(translator.cache)

    def test_schema_is_sent_to_ollama_with_bounded_context(self):
        translator = OllamaTranslator()
        response = Mock(status_code=200, json=lambda: {'response':'[]','done_reason':'stop'})
        with patch('core.ollama_translator.httpx.post',return_value=response) as post:
            translator._post_generate('qwen','sistema','prompt',0,1024,response_format={'type':'array'})
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['format'], {'type':'array'})
        self.assertEqual(payload['options']['num_ctx'],8192)
        self.assertEqual(payload['options']['num_predict'],1024)

    def test_truncated_output_is_reported_not_parsed_as_complete(self):
        translator = OllamaTranslator()
        response = Mock(status_code=200, json=lambda: {'response':'[{','done_reason':'length'})
        with patch('core.ollama_translator.httpx.post',return_value=response), patch('core.ollama_translator.time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'truncó'):
                translator._post_generate('qwen','sistema','prompt',0)

    def test_chinese_response_is_not_accepted_as_spanish(self):
        translator = OllamaTranslator()
        with patch.object(translator,'_post_generate',return_value='[{"id":1,"text":"你好"}]'), patch.object(translator,'translate_with_qwen',return_value='你好'):
            self.assertEqual(translator.translate_segments_contextual(self.segments(1)), [''])

    def test_translation_checkpoint_survives_failure_of_later_batch(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from core import analysis_store
        from test_manual_translation import load_class
        with tempfile.TemporaryDirectory() as folder, patch.object(analysis_store, 'CACHE_DIR', Path(folder)/'cache'):
            video = Path(folder)/'video.mp4'
            video.write_bytes(b'video de prueba')
            segments = self.segments(2)
            for seg in segments:
                seg['final_text'] = ''
            provider = Mock()
            provider.is_available.return_value = True
            def fail_after_first_batch(values, progress_callback=None):
                progress_callback(1, 2, ['Hola'])
                raise RuntimeError('Fallo de un bloque posterior')
            provider.translate_segments_contextual.side_effect = fail_after_first_batch
            namespace = dict(QThread=object, Signal=lambda *args: Mock(), List=list, Dict=dict,
                             clear_segment_tts=Mock(), logger=Mock())
            cls = load_class('gui/workers.py', 'RetryTranslationWorker', namespace)
            worker = cls(segments, 'http://localhost:11434', 'español', video_path=str(video))
            with patch('core.ollama_translator.OllamaTranslator', return_value=provider):
                worker.run()
            worker.error.emit.assert_called_once()
            recovered = analysis_store.load_analysis(video)
            self.assertEqual(recovered[0]['final_text'], 'Hola')
            self.assertEqual(recovered[1]['final_text'], '')

    def test_short_phrase_repair_enforces_budget_and_rejects_expanded_translation(self):
        translator = OllamaTranslator()
        with patch.object(translator, '_post_generate', return_value='Mira.'):
            self.assertEqual(translator.repair_for_dubbing('看', .24, 'Una frase larga'), 'Mira.')
        for bad in ('Tenemos alimentos ligeros. Cada uno puede llevar algunos y partir.', '你好', ''):
            with patch.object(translator, '_post_generate', return_value=bad):
                with self.assertRaises(ValueError):
                    translator.repair_for_dubbing('看', .24)

    def test_repair_accepts_two_words_for_short_title_and_longer_chinese_source(self):
        translator = OllamaTranslator()
        with patch.object(translator, 'translate_simple', return_value='Pequeño hermanito'), patch.object(translator, '_post_generate') as contextual:
            self.assertEqual(translator.repair_for_dubbing('小老弟', .58, 'Frase larga inventada'), 'Pequeño hermanito')
            contextual.assert_not_called()
        with patch.object(translator, 'translate_simple', return_value='Se atreven a engañarme') as direct, patch.object(translator, '_post_generate') as contextual:
            self.assertEqual(translator.repair_for_dubbing('竟敢骗到我头上', 1.4, 'Denunciaré a la frutería'), 'Se atreven a engañarme')
            direct.assert_called_once_with('竟敢骗到我头上')
            contextual.assert_not_called()
