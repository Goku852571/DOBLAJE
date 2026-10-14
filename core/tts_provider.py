"""Proveedor TTS local mediante System.Speech de Windows."""
import base64
import os
import subprocess
import tempfile
from typing import Callable, Optional

from utils.logger import logger


_VOICE_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try {
    $voices = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled })
    $selected = $voices | Where-Object { $_.VoiceInfo.Name -like "*$env:VIDEO_SUB_TTS_VOICE*" } | Select-Object -First 1
    if ($null -eq $selected) { $selected = $voices | Where-Object { $_.VoiceInfo.Culture.Name -like 'es*' } | Select-Object -First 1 }
    if ($null -eq $selected) { $selected = $voices | Select-Object -First 1 }
    if ($null -eq $selected) { throw 'No hay voces SAPI habilitadas en Windows.' }
    $synth.SelectVoice($selected.VoiceInfo.Name)
    $text = Get-Content -LiteralPath $env:VIDEO_SUB_TTS_TEXT -Raw -Encoding UTF8
    $synth.SetOutputToWaveFile($env:VIDEO_SUB_TTS_OUTPUT)
    $synth.Speak($text)
    Write-Output $selected.VoiceInfo.Name
}
finally { $synth.Dispose() }
'''

_LIST_VOICES_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try { $synth.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name } }
finally { $synth.Dispose() }
'''


def _encoded_powershell(script: str) -> str:
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


class SapiTTSProvider:
    """Genera WAV con las voces locales de Windows, incluido Vocalizer Jorge."""

    def __init__(self, voice_name: str = "Jorge"):
        self.voice_name = voice_name

    @staticmethod
    def _run_powershell(script: str, env: Optional[dict] = None) -> subprocess.CompletedProcess:
        command = ["powershell.exe", "-Sta", "-NoProfile", "-ExecutionPolicy", "Bypass",
                   "-EncodedCommand", _encoded_powershell(script)]
        # System.Speech accesses voices from the interactive Windows session.
        # A hidden, detached process cannot reliably enumerate Vocalizer voices.
        return subprocess.run(command, capture_output=True, text=True, check=False, env=env, timeout=120)

    def synthesize(self, text: str, output_path: str, voice_id: str = "",
                   target_duration: Optional[float] = None,
                   log: Callable[[str], None] = None) -> bool:
        if not text.strip():
            if log:
                log("Advertencia: texto vacío, omitiendo síntesis de voz.")
            return False
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        text_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="videodub_tts_",
                                             delete=False, encoding="utf-8",
                                             dir=os.path.dirname(output_path)) as text_file:
                text_file.write(text)
                text_path = text_file.name
            environment = os.environ.copy()
            environment.update({
                "VIDEO_SUB_TTS_TEXT": text_path,
                "VIDEO_SUB_TTS_OUTPUT": output_path,
                "VIDEO_SUB_TTS_VOICE": voice_id or self.voice_name,
            })
            if log:
                log(f"Sintetizando con voz local '{voice_id or self.voice_name}'...")
            result = self._run_powershell(_VOICE_SCRIPT, environment)
            if result.returncode != 0 or not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                detail = (result.stderr or result.stdout or "sin detalle").strip()
                if log:
                    log(f"Error en síntesis SAPI5: {detail}")
                return False
            if log:
                log(f"Sintetización completada con: {result.stdout.strip()}")
            return True
        except subprocess.TimeoutExpired:
            if log:
                log("La síntesis SAPI superó 120 segundos; se podrá reintentar este segmento.")
            return False
        except OSError as exc:
            if log:
                log(f"Error al crear el audio local: {exc}")
            return False
        finally:
            if text_path and os.path.exists(text_path):
                try:
                    os.remove(text_path)
                except OSError:
                    pass

    def get_voice_names(self) -> list:
        result = self._run_powershell(_LIST_VOICES_SCRIPT)
        if result.returncode != 0:
            logger.error("Error obteniendo voces SAPI5: %s", result.stderr.strip())
            return []
        return [name.strip() for name in result.stdout.splitlines() if name.strip()]
