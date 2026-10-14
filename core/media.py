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

def extract_audio(video_path: str, output_path: str, for_mixing: bool = False) -> bool:
    """Extract audio from video to a WAV file."""
    rate = 48000 if for_mixing else 16000
    cmd = [
        _ffmpeg_binary(), "-y", "-err_detect", "ignore_err", "-i", video_path,
        "-map", "0:a:0", "-vn", "-af", "aresample=async=1:first_pts=0",
        "-acodec", "pcm_s16le", "-ar", str(rate), "-ac", "2" if for_mixing else "1",
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
                       log=None, fit_info=None, match_duration: bool = False,
                       max_speed: float = None) -> bool:
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
            # Modest slowing aligns short phrases without stretching them unnaturally.
            speed = max(0.85 if match_duration else 1.0, speech_duration / target_duration)
            # atempo does not always output the exact requested duration.
            # Measure and retry instead of chopping off the final spoken word.
            for attempt in range(2):
                if max_speed is not None and speed > max_speed:
                    raise ValueError(f"La traducción requiere una velocidad de {speed:.2f}x. "
                                     f"Acorte el texto para este intervalo; el máximo es {max_speed:.2f}x.")
                run_cmd([_ffmpeg_binary(), "-y", "-i", speech,
                         "-af", ",".join(_atempo_chain(speed)), "-c:a", "pcm_s16le", tempo],
                        capture_output=True, text=True, check=True)
                fitted_duration = _trim_pcm_silence(tempo, fitted)["duration"]
                if fitted_duration <= target_duration:
                    break
                # Leave a small safety margin for atempo's analysis windows so
                # the second and final attempt preserves the entire spoken tail.
                speed *= fitted_duration / target_duration * 1.05
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
                action = f"ajustando velocidad {speed:.2f}x" if not math.isclose(speed, 1.0, abs_tol=0.0001) else "velocidad natural"
                log(f"Voz hablada: {speech_duration:.3f}s; intervalo: {target_duration:.3f}s; {action}. "
                    f"Silencio inicial eliminado: {trimmed['leading_silence']:.3f}s.")
        return True
    except (subprocess.CalledProcessError, OSError, ValueError, wave.Error) as exc:
        if fit_info is not None:
            fit_info.update(error=str(getattr(exc, "stderr", None) or exc),
                            retryable=not isinstance(exc, ValueError))
        logger.error("Error sincronizando voz: %s", getattr(exc, "stderr", None) or exc)
        return False

def build_dubbed_audio(segments: List[Dict], original_audio: str, target_duration: float, output_path: str, temp_dir: str,
                       original_volume: float = 0.5, tts_volume: float = 1.0,
                       preserve_stereo: bool = False, preserve_background: bool = False) -> bool:
    """Mix original audio with TTS segments. Volumes are configurable (0.0 a 1.0+)."""
    if not math.isfinite(target_duration) or target_duration <= 0:
        logger.error("La duración del video no es válida para la mezcla.")
        return False
    rate = 48000
    total_samples = round(target_duration * rate)
    layout = "stereo" if preserve_stereo else "mono"
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
    # 2. Batch mixing to avoid command line limits
    batch_size = 40
    batch_files = []
    
    for b_idx in range(0, len(valid_segments), batch_size):
        batch = valid_segments[b_idx:b_idx+batch_size]
        b_inputs = ["-i", silent_audio]
        b_filter = ""
        for i, seg in enumerate(batch):
            b_inputs.extend(["-i", seg['tts_audio']])
            # Absolute sample positions avoid accumulated rounding or PTS offsets.
            start_sample = round(float(seg['start']) * rate)
            end_sample = round(float(seg['end']) * rate)
            clip_samples = end_sample - start_sample
            b_filter += (
                f"[{i+1}:a]aresample={rate},aformat=channel_layouts=mono,"
                f"atrim=end_sample={clip_samples},asetpts=PTS-STARTPTS,"
                f"adelay={start_sample}S:all=1[a{i}];"
            )
            
        b_mix_inputs = "[0:a]" + "".join([f"[a{i}]" for i in range(len(batch))])
        b_filter += (
            f"{b_mix_inputs}amix=inputs={len(batch)+1}:duration=first:"
            "dropout_transition=0:normalize=0[tts_mix]"
        )
        b_out = os.path.join(temp_dir, f"batch_{b_idx}.wav")
        batch_files.append(b_out)
        
        cmd_b = [_ffmpeg_binary(), "-y"] + b_inputs + ["-filter_complex", b_filter, "-map", "[tts_mix]", b_out]
        try:
            run_cmd(cmd_b, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error mixing TTS batch {b_idx}: {e.stderr}")
            return False

    tts_mixed_audio = os.path.join(temp_dir, "tts_mixed.wav")

    # Mix the batches together
    if not batch_files:
        return False
        
    if len(batch_files) == 1:
        import shutil
        shutil.copyfile(batch_files[0], tts_mixed_audio)
    else:
        final_inputs = []
        for f in batch_files:
            final_inputs.extend(["-i", f])
            
        final_filter = "".join([f"[{i}:a]" for i in range(len(batch_files))])
        final_filter += f"amix=inputs={len(batch_files)}:duration=first:dropout_transition=0:normalize=0[final_mix]"
        
        cmd_final = [_ffmpeg_binary(), "-y"] + final_inputs + ["-filter_complex", final_filter, "-map", "[final_mix]", tts_mixed_audio]
        try:
            run_cmd(cmd_final, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error final mixing TTS: {e.stderr}")
            return False
        
    if preserve_background:
        return _mix_preserving_background(original_audio, tts_mixed_audio, target_duration,
                                          output_path, temp_dir, original_volume, tts_volume,
                                          preserve_stereo)

    # 3. Mix original (volumen configurable) with TTS mixed (volumen configurable)
    cmd_final = [
        _ffmpeg_binary(), "-y",
        "-i", original_audio,
        "-i", tts_mixed_audio,
        "-filter_complex",
        (
            f"[0:a]aresample={rate},aformat=channel_layouts={layout},asetpts=PTS-STARTPTS,apad=whole_len={total_samples},"
            f"atrim=end_sample={total_samples},volume={original_volume}[org];"
            f"[1:a]aformat=channel_layouts={layout},volume={tts_volume},dynaudnorm=f=150:g=15,asetpts=PTS-STARTPTS[tts];"
            "[org][tts]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            "alimiter=limit=0.95:level=0:latency=1"
        ),
        "-acodec", "pcm_s16le" if output_path.lower().endswith(".wav") else "aac",
        "-ar", str(rate),
        "-ac", "2" if preserve_stereo else "1",
        output_path
    ]
    
    try:
        run_cmd(cmd_final, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error final audio mix: {e.stderr}")
        return False

def _mix_preserving_background(background, voice, duration, output_path, temp_dir,
                               background_volume, voice_volume, stereo):
    """Preserve the background component; clipping control affects dubbing only."""
    import numpy as np
    import soundfile as sf
    rate, channels = 48000, 2 if stereo else 1
    frames = round(duration * rate)
    converted = os.path.join(temp_dir, "background_original_level.wav")
    floating_mix = output_path if output_path.lower().endswith('.wav') else os.path.join(temp_dir, 'preserved_mix.wav')
    try:
        run_cmd([_ffmpeg_binary(), '-y', '-i', background, '-af',
                 f'aresample={rate},apad=whole_len={frames},atrim=end_sample={frames}',
                 '-ar', str(rate), '-ac', str(channels), '-c:a', 'pcm_f32le', converted],
                capture_output=True, text=True, check=True)
        gain_previous, limited_frames = 1.0, 0
        release_step = 1 / (rate * .1)
        with sf.SoundFile(converted) as bed, sf.SoundFile(voice) as speech, sf.SoundFile(
                floating_mix, 'w', samplerate=rate, channels=channels, subtype='FLOAT') as mixed:
            remaining = frames
            while remaining:
                count = min(rate, remaining)
                bg = bed.read(count, dtype='float64', always_2d=True) * background_volume
                dub = speech.read(count, dtype='float64', always_2d=True) * voice_volume
                if len(bg) < count:
                    bg = np.pad(bg, ((0, count - len(bg)), (0, 0)))
                if len(dub) < count:
                    dub = np.pad(dub, ((0, count - len(dub)), (0, 0)))
                if dub.shape[1] != channels:
                    dub = np.repeat(dub, channels, axis=1) if dub.shape[1] == 1 else dub.mean(axis=1, keepdims=True)
                if not np.isfinite(bg).all() or not np.isfinite(dub).all():
                    raise RuntimeError('La mezcla contiene muestras no válidas.')
                if np.max(np.abs(bg)) > 1.0:
                    logger.warning('El fondo separado tiene picos fuera del rango. Se limitarán (clipping) para evitar fallos.')
                    bg = np.clip(bg, -1.0, 1.0)
                allowance = np.ones_like(dub)
                np.divide(1.0 - np.sign(dub) * bg, np.abs(dub), out=allowance, where=np.abs(dub) > 1e-12)
                allowed = np.clip(allowance.min(axis=1), 0, 1)
                # Immediate attack and 100 ms recovery affect only the added
                # voice. No ducking or global normalization acts on the bed.
                ticks = np.arange(1, count + 1) * release_step
                gain = np.minimum.accumulate(np.minimum(allowed - ticks, gain_previous)) + ticks
                gain = np.clip(gain, 0, 1)
                gain_previous = float(gain[-1])
                limited_frames += int(np.count_nonzero(gain < .9999))
                mixed.write(bg + dub * gain[:, None])
                remaining -= count
        if floating_mix != output_path:
            run_cmd([_ffmpeg_binary(), '-y', '-i', floating_mix, '-c:a', 'aac', output_path],
                    capture_output=True, text=True, check=True)
        logger.info('Fondo conservado a ganancia %.2f; control de picos solo en el doblaje durante %.3f s.',
                    background_volume, limited_frames / rate)
        return True
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        logger.error('Error conservando el volumen del fondo: %s', getattr(exc, 'stderr', None) or exc)
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
