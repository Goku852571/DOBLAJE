import os
import subprocess
import json
import math
import platform
import wave
import sys
import tempfile
from array import array
from typing import List, Dict, Any, Tuple
from utils.logger import logger


def _ffmpeg_binary() -> str:
    """Use system FFmpeg, or the binary supplied by imageio-ffmpeg."""
    from shutil import which
    if which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("FFmpeg no está instalado. Ejecuta pip install -r requirements.txt.") from exc


def _ffprobe_binary() -> str:
    from shutil import which
    if which("ffprobe"):
        return "ffprobe"
    ffmpeg = _ffmpeg_binary()
    name = "ffprobe.exe" if platform.system() == "Windows" else "ffprobe"
    candidate = os.path.join(os.path.dirname(ffmpeg), name)
    if os.path.isfile(candidate):
        return candidate
    raise RuntimeError("FFprobe no está disponible. Instala FFmpeg completo y añádelo a PATH.")

def run_cmd(cmd: List[str], **kwargs):
    if platform.system() == "Windows":
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
    return subprocess.run(cmd, **kwargs)


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Retrieve video duration, fps and check for audio stream using ffprobe."""
    try:
        cmd = [
            _ffprobe_binary(), "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        result = run_cmd(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        duration = float(data['format'].get('duration', 0.0))
        fps = 0.0
        has_audio = False
        
        for stream in data.get('streams', []):
            if stream['codec_type'] == 'video' and fps == 0.0:
                # e.g., "30000/1001" or "30/1"
                r_frame_rate = stream.get('r_frame_rate', '0/1')
                if '/' in r_frame_rate:
                    num, den = r_frame_rate.split('/')
                    if den != '0':
                        fps = float(num) / float(den)
            elif stream['codec_type'] == 'audio':
                has_audio = True
                
        return {
            "duration": duration,
            "fps": fps,
            "has_audio": has_audio
        }
    except Exception:
        # imageio-ffmpeg bundles ffmpeg but usually not ffprobe. Fall back to
        # OpenCV for visual metadata and FFmpeg's stream description for audio.
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            probe = run_cmd([_ffmpeg_binary(), "-hide_banner", "-i", video_path],
                            capture_output=True, text=True)
            return {
                "duration": frames / fps if fps > 0 else 0.0,
                "fps": fps if fps > 0 else 0.0,
                "has_audio": "Audio:" in probe.stderr,
            }
        except Exception as fallback_error:
            logger.error(f"Error getting video info for {video_path}: {fallback_error}")
            return {"duration": 0.0, "fps": 0.0, "has_audio": False}

def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio from video to a WAV file."""
    cmd = [
        _ffmpeg_binary(), "-y", "-i", video_path,
        "-map", "0:a:0", "-vn", "-af", "aresample=16000:async=1:first_pts=0",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path
    ]
    try:
        run_cmd(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e.stderr}")
        return False

def get_audio_duration(audio_path: str) -> float:
    """Return duration without requiring ffprobe for WAV files."""
    if audio_path.lower().endswith(".wav"):
        try:
            with wave.open(audio_path, "rb") as wav:
                if wav.getframerate() > 0:
                    return wav.getnframes() / wav.getframerate()
        except (wave.Error, OSError):
            pass
    cmd = [
        _ffprobe_binary(), "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    try:
        result = run_cmd(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error getting audio duration for {audio_path}: {e}")
        return 0.0


def _atempo_chain(speed: float) -> List[str]:
    """Build valid atempo stages. FFmpeg accepts each stage only in 0.5..2.0."""
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("La velocidad del audio debe ser positiva y finita.")
    stages = []
    while speed > 2.0:
        stages.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        stages.append("atempo=0.5")
        speed /= 0.5
    if not math.isclose(speed, 1.0, rel_tol=1e-6) or not stages:
        stages.append(f"atempo={speed:.8f}")
    return stages


def clear_segment_tts(segment: Dict):
    """Discard audio tied to an older text or older timestamps."""
    for key in list(segment):
        if key.startswith("tts_"):
            segment.pop(key)


def _trim_pcm_silence(input_path: str, output_path: str) -> Dict:
    """Remove only edge silence from a mono PCM16 WAV; keep internal pauses."""
    with wave.open(input_path, "rb") as wav:
        params = wav.getparams()
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError("El ajuste de voz requiere WAV PCM16 mono.")
        samples = array("h", wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    peak = max((abs(s) for s in samples), default=0)
    # Conservative threshold protects low-volume consonants. TTS edge silence
    # is normally zero; keep 5 ms of margin for a natural attack and release.
    threshold = max(16, int(peak * 0.001))
    first = next((i for i, s in enumerate(samples) if abs(s) > threshold), None)
    if first is None:
        raise ValueError("La voz generada contiene solo silencio.")
    last = len(samples) - next(i for i, s in enumerate(reversed(samples)) if abs(s) > threshold)
    margin = round(params.framerate * 0.005)
    start, end = max(0, first - margin), min(len(samples), last + margin)
    clipped = samples[start:end]
    if sys.byteorder != "little":
        clipped.byteswap()
    with wave.open(output_path, "wb") as wav:
        wav.setparams(params)
        wav.writeframes(clipped.tobytes())
    return dict(duration=(end - start) / params.framerate,
                leading_silence=start / params.framerate,
                trailing_silence=(len(samples) - end) / params.framerate)


def time_stretch_audio(input_path: str, target_duration: float, output_path: str,
                       log=None, fit_info=None) -> bool:
    """Align the audible onset, accelerate if needed, then pad to exact length."""
    if not math.isfinite(target_duration) or target_duration <= 0:
        logger.error("Duración inválida para sincronizar voz.")
        return False
    rate = 48000
    target_samples = round(target_duration * rate)
    if target_samples < 1:
        return False
    target_duration = target_samples / rate
    try:
        with tempfile.TemporaryDirectory(prefix="tts_fit_", dir=os.path.dirname(os.path.abspath(output_path))) as work:
            pcm = os.path.join(work, "pcm.wav")
            speech = os.path.join(work, "speech.wav")
            tempo = os.path.join(work, "tempo.wav")
            fitted = os.path.join(work, "fitted.wav")
            run_cmd([_ffmpeg_binary(), "-y", "-i", input_path, "-vn", "-ar", str(rate),
                     "-ac", "1", "-c:a", "pcm_s16le", pcm],
                    capture_output=True, text=True, check=True)
            trimmed = _trim_pcm_silence(pcm, speech)
            speech_duration = trimmed["duration"]
            speed = max(1.0, speech_duration / target_duration)
            # atempo does not always output the exact requested duration.
            # Measure and retry instead of chopping off the final spoken word.
            for attempt in range(3):
                run_cmd([_ffmpeg_binary(), "-y", "-i", speech,
                         "-af", ",".join(_atempo_chain(speed)), "-c:a", "pcm_s16le", tempo],
                        capture_output=True, text=True, check=True)
                fitted_duration = _trim_pcm_silence(tempo, fitted)["duration"]
                if fitted_duration <= target_duration:
                    break
                speed *= fitted_duration / target_duration * 1.005
            else:
                raise ValueError("No se pudo ajustar la voz completa al intervalo; regenere el segmento.")
            run_cmd([_ffmpeg_binary(), "-y", "-i", fitted, "-af",
                     f"apad=whole_len={target_samples},atrim=end_sample={target_samples},asetpts=PTS-STARTPTS",
                     "-c:a", "pcm_s16le", output_path],
                    capture_output=True, text=True, check=True)
            if abs(get_audio_duration(output_path) - target_duration) > 1 / rate:
                raise ValueError("La duración final de la voz no coincide con su intervalo.")
            if fit_info is not None:
                fit_info.update(tts_speech_duration=speech_duration,
                                tts_leading_silence=trimmed["leading_silence"],
                                tts_trailing_silence=trimmed["trailing_silence"], tts_speed=speed)
            if log:
                action = f"acelerando {speed:.2f}x" if speed > 1.0001 else "velocidad natural"
                log(f"Voz hablada: {speech_duration:.3f}s; intervalo: {target_duration:.3f}s; {action}. "
                    f"Silencio inicial eliminado: {trimmed['leading_silence']:.3f}s.")
        return True
    except (subprocess.CalledProcessError, OSError, ValueError, wave.Error) as exc:
        logger.error("Error sincronizando voz: %s", getattr(exc, "stderr", None) or exc)
        return False

def build_dubbed_audio(segments: List[Dict], original_audio: str, target_duration: float, output_path: str, temp_dir: str,
                       original_volume: float = 0.5, tts_volume: float = 1.0) -> bool:
    """Mix original audio with TTS segments. Volumes are configurable (0.0 a 1.0+)."""
    if not math.isfinite(target_duration) or target_duration <= 0:
        logger.error("La duración del video no es válida para la mezcla.")
        return False
    rate = 48000
    total_samples = round(target_duration * rate)
    valid_segments = [s for s in segments if s.get('tts_audio') and os.path.isfile(s['tts_audio'])]
    logger.info("Segmentos TTS disponibles para mezcla: %d/%d", len(valid_segments), len(segments))
    if not valid_segments:
        logger.error("No hay archivos de voz generada para mezclar.")
        return False
    if any(s.get("final_text", "").strip() and s not in valid_segments for s in segments):
        logger.error("Faltan voces de segmentos traducidos. Genere las voces antes de crear el video.")
        return False
    for seg in valid_segments:
        start, end = float(seg['start']), float(seg['end'])
        if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end <= target_duration + 1 / rate:
            logger.error("Intervalo de voz inválido: %s - %s.", start, end)
            return False
        if (seg.get("tts_start", start) != start or seg.get("tts_end", end) != end or
                seg.get("tts_text", seg.get("final_text", "")).strip() != seg.get("final_text", "").strip()):
            logger.error("El texto o los tiempos cambiaron después de generar la voz. Genere las voces de nuevo.")
            return False
    # 1. Create silent audio of target_duration
    silent_audio = os.path.join(temp_dir, "silence.wav")
    cmd_silence = [
        _ffmpeg_binary(), "-y", "-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=mono",
        "-t", str(target_duration), silent_audio
    ]
    run_cmd(cmd_silence, capture_output=True, check=True)
    
    # 2. Mix all TTS clips onto the silent track using amix or adelay
    # The command can get very long if we have many segments, so we use a complex filter
    
    inputs = ["-i", silent_audio]
    filter_complex = ""
    
    for i, seg in enumerate(valid_segments):
        inputs.extend(["-i", seg['tts_audio']])
        # Absolute sample positions avoid accumulated rounding or PTS offsets.
        start_sample = round(float(seg['start']) * rate)
        end_sample = round(float(seg['end']) * rate)
        clip_samples = end_sample - start_sample
        filter_complex += (
            f"[{i+1}:a]aresample={rate},aformat=channel_layouts=mono,"
            f"atrim=end_sample={clip_samples},asetpts=PTS-STARTPTS,"
            f"adelay={start_sample}S:all=1[a{i}];"
        )
        
    # Now mix all delayed audios with the silence
    mix_inputs = "[0:a]" + "".join([f"[a{i}]" for i in range(len(valid_segments))])
    if valid_segments:
        filter_complex += (
            f"{mix_inputs}amix=inputs={len(valid_segments)+1}:duration=first:"
            "dropout_transition=0:normalize=0[tts_mix]"
        )

    tts_mixed_audio = os.path.join(temp_dir, "tts_mixed.wav")
    
    cmd_mix = [_ffmpeg_binary(), "-y"] + inputs + ["-filter_complex", filter_complex, "-map", "[tts_mix]", tts_mixed_audio]
    try:
        run_cmd(cmd_mix, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error mixing TTS: {e.stderr}")
        return False
        
    # 3. Mix original (volumen configurable) with TTS mixed (volumen configurable)
    cmd_final = [
        _ffmpeg_binary(), "-y",
        "-i", original_audio,
        "-i", tts_mixed_audio,
        "-filter_complex",
        (
            f"[0:a]aresample={rate},asetpts=PTS-STARTPTS,apad=whole_len={total_samples},"
            f"atrim=end_sample={total_samples},volume={original_volume}[org];"
            f"[1:a]volume={tts_volume},dynaudnorm=f=150:g=15,asetpts=PTS-STARTPTS[tts];"
            "[org][tts]amix=inputs=2:duration=first:dropout_transition=0:normalize=0"
        ),
        "-acodec", "pcm_s16le" if output_path.lower().endswith(".wav") else "aac",
        "-ar", str(rate),
        output_path
    ]
    
    try:
        run_cmd(cmd_final, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error final audio mix: {e.stderr}")
        return False

def render_final_video(video_path: str, new_audio_path: str, output_path: str) -> bool:
    """Combine original video (copy) with the new audio."""
    cmd = [
        _ffmpeg_binary(), "-y",
        "-i", video_path,
        "-i", new_audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path
    ]
    try:
        run_cmd(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error rendering video: {e.stderr}")
        return False
