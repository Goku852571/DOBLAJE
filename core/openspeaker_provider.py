"""OpenSpeaker v3 TTS, using the documented asynchronous task API."""
import math
import os
import random
import tempfile
import time
from urllib.parse import urlparse

import requests

BASE_URL = "https://api.ai33.pro"
PROVIDERS = ("elevenlabs", "minimax", "clone", "edge", "kokoro", "vbee", "fishaudio")


class OpenSpeakerTTSProvider:
    def __init__(self, api_key="", speed=1.0, timeout=600):
        self.api_key = (api_key or os.environ.get("OPENSPEAKER_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("Introduce la clave OpenSpeaker o define OPENSPEAKER_API_KEY.")
        self.speed = float(speed)
        if not math.isfinite(self.speed) or not 0.5 <= self.speed <= 1.5:
            raise ValueError("La velocidad OpenSpeaker debe estar entre 0.5 y 1.5.")
        self.timeout = timeout

    def _request(self, method, path, deadline, **kwargs):
        for attempt in range(6):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("OpenSpeaker agotó el tiempo de espera.")
            response = requests.request(
                method, BASE_URL + path, headers={"xi-api-key": self.api_key},
                timeout=(min(15, remaining), min(60, remaining)), **kwargs)
            if response.status_code in (429, 503) and attempt < 5:
                try:
                    retry = float(response.headers.get("Retry-After", "0"))
                except ValueError:
                    retry = 0
                delay = max(retry, 2 ** attempt + random.random())
                if delay >= deadline - time.monotonic():
                    raise TimeoutError("OpenSpeaker sigue ocupado; inténtalo más tarde.")
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                messages = {401: "Clave API inválida.", 403: "Acceso denegado.",
                            402: "Créditos insuficientes.", 429: "Límite de solicitudes alcanzado."}
                raise RuntimeError("OpenSpeaker HTTP %s: %s" % (
                    response.status_code, messages.get(response.status_code, "La solicitud falló.")))
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("success") is False:
                raise RuntimeError("OpenSpeaker rechazó la solicitud.")
            return payload
        raise RuntimeError("OpenSpeaker no está disponible.")

    def list_voices(self, provider="edge", search="", page=1):
        if provider not in PROVIDERS:
            raise ValueError("Proveedor de voces no válido.")
        return self._request("GET", "/v3/voices", time.monotonic() + self.timeout,
                             params={"provider": provider, "search": search,
                                     "page": page, "page_size": 100})

    def synthesize(self, text, output_path, voice_id="", target_duration=None, log=None):
        if not text.strip() or len(text) > 1_000_000:
            raise ValueError("El texto debe contener entre 1 y 1.000.000 caracteres.")
        if not any(voice_id.startswith(p + "_") and len(voice_id) > len(p) + 1 for p in PROVIDERS):
            raise ValueError("Usa un voice_id con prefijo de proveedor, obtenido de /v3/voices.")
        deadline = time.monotonic() + self.timeout
        result = self._request("POST", "/v3/text-to-speech", deadline, files={
            "text": (None, text), "voice_id": (None, voice_id),
            "speed": (None, str(self.speed)), "with_transcript": (None, "false")})
        task_id = result.get("task_id")
        if not task_id:
            raise RuntimeError("OpenSpeaker no devolvió task_id.")
        if log:
            log("OpenSpeaker: esperando el audio generado...")
        while True:
            task = self._request("GET", "/v1/task/" + requests.utils.quote(str(task_id), safe=""), deadline)
            status = task.get("status")
            if status == "done":
                audio_url = (task.get("metadata") or {}).get("audio_url")
                if not audio_url or urlparse(audio_url).scheme != "https":
                    raise RuntimeError("OpenSpeaker completó la tarea sin una URL HTTPS de audio.")
                break
            if status in ("failed", "error", "cancelled", "canceled"):
                raise RuntimeError("OpenSpeaker: la generación de audio falló.")
            if deadline - time.monotonic() < 2:
                raise TimeoutError("OpenSpeaker agotó el tiempo de espera de la tarea.")
            time.sleep(2)
        from core.media import _ffmpeg_binary, run_cmd
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="openspeaker_", dir=os.path.dirname(output_path)) as work:
            source = os.path.join(work, "source.audio")
            # Download without the API key: output may live on a separate CDN.
            with requests.get(audio_url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                size = 0
                with open(source, "wb") as handle:
                    for chunk in response.iter_content(65536):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Tiempo de descarga de audio agotado.")
                        size += len(chunk)
                        if size > 200 * 1024 * 1024:
                            raise RuntimeError("El audio supera el límite de descarga de 200 MB.")
                        handle.write(chunk)
            converted = os.path.join(work, "converted.wav")
            run_cmd([_ffmpeg_binary(), "-y", "-i", source, "-vn", "-ac", "1",
                     "-ar", "48000", "-c:a", "pcm_s16le", converted],
                    capture_output=True, check=True, timeout=120)
            if not os.path.isfile(converted) or os.path.getsize(converted) <= 44:
                raise RuntimeError("OpenSpeaker devolvió audio vacío.")
            os.replace(converted, output_path)
        return True
