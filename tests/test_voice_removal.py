"""Render routing: never mix the original dialogue after voice removal."""
import logging
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from test_manual_translation import load_class


class VoiceRemovalTests(unittest.TestCase):
    def make_worker(self, enabled):
        mix = Mock(return_value=True)
        namespace = dict(QThread=object, Signal=lambda *args: Mock(), List=list, Dict=dict,
                         config_manager=SimpleNamespace(get=lambda key, default=None:
                             enabled if key == 'remove_original_voice' else default),
                         os=__import__('os'), get_video_info=lambda path: {'duration': 10},
                         extract_audio=Mock(return_value=True), build_dubbed_audio=mix,
                         render_final_video=Mock(return_value=True), logger=logging.getLogger('separation_test'))
        cls = load_class('gui/workers.py', 'RenderWorker', namespace)
        return cls('video.mp4', [], 'temp', 'output.mp4'), namespace

    def test_checked_mixes_only_background_at_original_gain(self):
        worker, namespace = self.make_worker(True)
        module = ModuleType('core.voice_separator')
        module.separate_original_voice = Mock()
        with patch.dict('sys.modules', {'core.voice_separator': module}):
            worker.run()
        worker.error.emit.assert_not_called()
        module.separate_original_voice.assert_called_once()
        args = namespace['build_dubbed_audio'].call_args
        self.assertTrue(args.args[1].endswith('background_without_voice.wav'))
        self.assertEqual(args.kwargs['original_volume'], 1.0)
        self.assertTrue(args.kwargs['preserve_background'])
        self.assertTrue(namespace['extract_audio'].call_args.kwargs['for_mixing'])

    def test_unchecked_keeps_existing_mix(self):
        worker, namespace = self.make_worker(False)
        worker.run()
        worker.error.emit.assert_not_called()
        args = namespace['build_dubbed_audio'].call_args
        self.assertTrue(args.args[1].endswith('original_audio.wav'))
        self.assertEqual(args.kwargs['original_volume'], 0.5)
        self.assertFalse(args.kwargs['preserve_background'])

    def test_failed_separation_stops_instead_of_reusing_original_voice(self):
        worker, namespace = self.make_worker(True)
        module = ModuleType('core.voice_separator')
        module.separate_original_voice = Mock(side_effect=RuntimeError('separation failed'))
        with patch.dict('sys.modules', {'core.voice_separator': module}):
            worker.run()
        worker.error.emit.assert_called_once()
        namespace['build_dubbed_audio'].assert_not_called()
        namespace['render_final_video'].assert_not_called()
