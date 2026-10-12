import unittest
from unittest.mock import Mock, patch
from core.openspeaker_provider import OpenSpeakerTTSProvider


class OpenSpeakerTests(unittest.TestCase):
    def test_missing_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                OpenSpeakerTTSProvider()

    def test_invalid_voice_never_sends(self):
        with patch("core.openspeaker_provider.requests.request") as request:
            with self.assertRaises(ValueError):
                OpenSpeakerTTSProvider("secret").synthesize("Hola", "out.wav", "Jorge")
            request.assert_not_called()

    def test_multipart_and_failed_task(self):
        responses = [
            Mock(status_code=200, json=lambda: {"success": True, "task_id": "abc"}),
            Mock(status_code=200, json=lambda: {"status": "failed"}),
        ]
        with patch("core.openspeaker_provider.requests.request", side_effect=responses) as request:
            with self.assertRaises(RuntimeError):
                OpenSpeakerTTSProvider("secret").synthesize("Hola", "out.wav", "edge_es-ES-ElviraNeural")
            call = request.call_args_list[0]
            self.assertEqual(call.kwargs["headers"], {"xi-api-key": "secret"})
            self.assertEqual(call.kwargs["files"]["text"], (None, "Hola"))
            self.assertEqual(request.call_args_list[1].args[1], "https://api.ai33.pro/v1/task/abc")

    def test_rate_limit_then_success(self):
        limited = Mock(status_code=429, headers={"Retry-After": "3"})
        ok = Mock(status_code=200, json=lambda: {"success": True, "data": []})
        with patch("core.openspeaker_provider.requests.request", side_effect=[limited, ok]), patch(
            "core.openspeaker_provider.time.sleep") as sleep:
            self.assertEqual(OpenSpeakerTTSProvider("secret").list_voices()["data"], [])
            self.assertGreaterEqual(sleep.call_args.args[0], 3)

    def test_auth_failure_does_not_retry(self):
        with patch("core.openspeaker_provider.requests.request", return_value=Mock(status_code=401)) as request:
            with self.assertRaisesRegex(RuntimeError, "Clave API"):
                OpenSpeakerTTSProvider("secret").list_voices()
            self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
