import os
import math
from typing import List, Dict, Any
from PySide6.QtCore import QThread, Signal
from core.fusion import fuse_segments
from core.tts_provider import SapiTTSProvider
from core.media import (
    extract_audio, time_stretch_audio, build_dubbed_audio, render_final_video,
    get_video_info, get_audio_duration,
    clear_segment_tts,
)
from utils.logger import logger
from utils.config import config_manager

class TestConnectionWorker(QThread):
    """Worker para probar la conexión con Ollama en segundo plano."""
    log = Signal(str)
    finished_connection = Signal(dict)  # {"ok": bool, "message": str, "models": list}

    def __init__(self, translator):
        super().__init__()
        self.translator = translator

    def run(self):
        try:
            self.log.emit("Probando conexión con Ollama...")
            result = self.translator.test_connection()
            self.finished_connection.emit(result)
        except Exception as e:
            logger.exception("Error en TestConnectionWorker")
            self.finished_connection.emit({
                "ok": False,
                "message": f"Error inesperado: {e}",
                "models": []
            })


class RetryTranslationWorker(QThread):
    """Worker para reintentar la traducción de segmentos existentes."""
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(list)  # Updated segments
    error = Signal(str)

    def __init__(self, segments: List[Dict], host: str, target_language: str, video_path: str = ""):
        super().__init__()
        self.segments = segments
        self.host = host
        self.video_path = video_path
        self.target_language = target_language

    def run(self):
        from core.ollama_translator import OllamaTranslator
        translator = None
        try:
            translator = OllamaTranslator(
                host=self.host,
                target_language=self.target_language
            )
            
            if not translator.is_available():
                raise RuntimeError(
                    "Ollama no está disponible. "
                    "Verifique que esté ejecutándose."
                )
            
            total = len(self.segments)
            sources = []
            for seg in self.segments:
                clear_segment_tts(seg)
                source = (
                    seg.get('source_text', '').strip() or
                    seg.get('asr_text', '').strip()
                )
                sources.append(source)
                seg['source_text'] = source

            self.log.emit(
                "Analizando el guion completo para mejorar contexto antes de traducir..."
            )
            def on_batch(done, count, values):
                for segment, translated in zip(self.segments, values):
                    segment['final_text'] = translated
                self.progress.emit(int(done / count * 100))
                self.log.emit(f"Traducción contextual completada: {done}/{count} segmentos.")
                if self.video_path:
                    try:
                        from core.analysis_store import save_analysis
                        save_analysis(self.video_path, self.segments)
                    except (OSError, ValueError, TypeError) as exc:
                        self.log.emit(f"No se pudo guardar el progreso de traducción: {exc}")
            translations = translator.translate_segments_contextual(self.segments, progress_callback=on_batch)
            for i, (seg, source, translated) in enumerate(zip(self.segments, sources, translations)):
                if source and translated:
                    seg['source_text'] = source
                    seg['final_text'] = translated
                    self.log.emit(f"  [{i+1}/{total}] Traducción contextual: {translated[:60]}...")
                elif source:
                    seg['final_text'] = ""
                    self.log.emit(f"  [{i+1}/{total}] ERROR: no se obtuvo traducción al español.")
                else:
                    self.log.emit(f"  [{i+1}/{total}] Sin texto original, saltando.")
            
            self.log.emit(f"Traducción completada: {total} segmentos procesados.")
            self.finished.emit(self.segments)
            
        except Exception as e:
            logger.exception("Error en RetryTranslationWorker")
            self.error.emit(f"Error en reintento de traducción: {e}")
        finally:
            if translator is not None:
                translator.close()


class AnalyzeWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(list) # List of dict segments
    error = Signal(str)

    def __init__(self, video_path: str, manual_translation: bool = False):
        super().__init__()
        self.video_path = video_path
        self.manual_translation = manual_translation

    def run(self):
        import tempfile
        import gc
        translator = None
        try:
            with tempfile.TemporaryDirectory(prefix="videodub_analysis_") as temp_dir:
                info = get_video_info(self.video_path)
                asr_results = []
                if info['has_audio']:
                    self.log.emit("Extrayendo audio para transcripción local...")
                    audio_temp = os.path.join(temp_dir, "audio.wav")
                    if not extract_audio(self.video_path, audio_temp):
                        raise RuntimeError("Fallo al extraer el audio del video.")
                    if config_manager.get("asr_engine", "whisper_cpp") == "whisper_cpp":
                        from core.whisper_cpp_engine import WhisperCppEngine
                        asr_engine = WhisperCppEngine(
                            executable=config_manager.get("whisper_cpp_executable", "tools/whisper_cpp/whisper-cli.exe"),
                            model_path=config_manager.get("whisper_cpp_model_path", "models/whisper_cpp/ggml-large-v3-turbo.bin"),
                            vad_path=config_manager.get("whisper_cpp_vad_path", "tools/whisper_cpp/ggml-silero-v6.2.0.bin"),
                            threads=config_manager.get("whisper_cpp_threads", 4))
                        self.log.emit("Motor seleccionado: Whisper CPP local (CPU) con Silero VAD.")
                    else:
                        from core.asr_engine import ASREngine
                        asr_engine = ASREngine(
                            model_size=config_manager.get("asr_model", "large-v3-turbo"),
                            device=config_manager.get("asr_device", "cpu"),
                            compute_type=config_manager.get("asr_compute_type", "default"),
                            pause_threshold=config_manager.get("asr_pause_threshold", 0.35),
                            model_path=config_manager.get("asr_model_path")
                            if config_manager.get("asr_model", "large-v3-turbo") in
                            ("large-v3-turbo", "turbo") else None)
                    asr_results = asr_engine.extract_audio_text(
                        audio_temp, lambda p: self.progress.emit(int(p * 0.8)),
                        language=config_manager.get("asr_language", "auto"))
                    del asr_engine
                    gc.collect()
                else:
                    raise RuntimeError("El video no tiene una pista de audio para transcribir.")
                segments = fuse_segments(asr_results)
                if segments:
                    try:
                        from core.analysis_store import save_analysis
                        # Todavía no hay traducción: no confundir el original con español.
                        for segment in segments:
                            segment['final_text'] = ''
                        saved = save_analysis(self.video_path, segments)
                        self.log.emit(f"Análisis ASR guardado antes de traducir: {saved}")
                    except (OSError, ValueError, TypeError) as exc:
                        self.log.emit(f"No se pudo guardar el análisis ASR: {exc}")
                if self.manual_translation:
                    for seg in segments:
                        seg['source_text'] = seg.get('source_text') or seg.get('asr_text', '')
                        seg['final_text'] = ''
                    self.progress.emit(100)
                    self.log.emit(f"Guion listo: {len(segments)} segmentos para traducción manual en JSON.")
                    self.finished.emit(segments)
                    return
                if segments:
                    from core.ollama_translator import OllamaTranslator
                    self.log.emit("Conectando con Ollama para traducción...")
                    translator = OllamaTranslator(
                        host=config_manager.get("ollama_host", "http://localhost:11434"),
                        target_language=config_manager.get("translation_target", "español")
                    )
                    if not translator.is_available():
                        raise RuntimeError(
                            "Ollama no está disponible. "
                            "Asegúrese de que Ollama esté ejecutándose en localhost:11434"
                        )
                    available_models = translator.get_available_models()
                    self.log.emit(f"Modelos disponibles en Ollama: {available_models}")
                    self.log.emit(
                        "Analizando guion completo para traducción contextual con Ollama..."
                    )
                    sources = []
                    for seg in segments:
                        source = (
                            seg.get('source_text', '').strip() or
                            seg.get('asr_text', '').strip() or
                            seg.get('final_text', '').strip()
                        )
                        sources.append(source)
                        seg['source_text'] = source

                    def on_batch(done, count, values):
                        for segment, translated in zip(segments, values):
                            segment['final_text'] = translated
                        self.progress.emit(80 + int(done / count * 20))
                        self.log.emit(f"Traducción contextual completada: {done}/{count} segmentos.")
                        try:
                            from core.analysis_store import save_analysis
                            save_analysis(self.video_path, segments)
                        except (OSError, ValueError, TypeError) as exc:
                            self.log.emit(f"No se pudo guardar el progreso de traducción: {exc}")
                    translations = translator.translate_segments_contextual(segments, progress_callback=on_batch)
                    for i, (seg, source, translated) in enumerate(zip(segments, sources, translations)):
                        if source and translated:
                            seg['final_text'] = translated
                            self.log.emit(f"  [{i+1}] Original: {source[:50]}...")
                            self.log.emit(f"  [{i+1}] Contextual: {translated[:60]}...")
                        elif source:
                            seg['final_text'] = ""
                            self.log.emit(f"  [{i+1}] ERROR: no se obtuvo traducción al español.")
                        else:
                            self.log.emit(f"  [{i+1}] Sin texto original, saltando.")
                self.progress.emit(100)
                self.log.emit(f"Análisis completado: {len(segments)} segmentos traducidos.")
                self.finished.emit(segments)
        except Exception as e:
            logger.exception("Error en AnalyzeWorker")
            self.error.emit(f"No se pudo completar el análisis/traducción local: {e}")
        finally:
            if translator is not None:
                translator.close()


class TTSWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(list) # Updated segments with audio paths
    error = Signal(str)

    def __init__(self, segments: List[Dict], temp_dir: str, api_key: str = ""):
        super().__init__()
        self.segments = segments
        self.temp_dir = temp_dir
        self.voice_name = config_manager.get("tts_voice_name", "Jorge")
        self.tts_engine = config_manager.get("tts_engine", "sapi")
        self.api_key = api_key
        self.api_voice = config_manager.get("openspeaker_voice_id", "")
        self.api_speed = config_manager.get("openspeaker_speed", 1.0)

    def _valid_audio(self, path, expected=None):
        if not path or not os.path.exists(path):
            return False
        try:
            duration = get_audio_duration(path)
            return (math.isfinite(duration) and duration > 0 and
                    (expected is None or abs(duration - expected) <= 2 / 48000))
        except Exception:
            return False

    def _save_checkpoint(self):
        import json
        import os as filesystem
        if not filesystem.path.isdir(self.temp_dir):
            return
        destination = filesystem.path.join(self.temp_dir, "tts_progreso.json")
        try:
            with open(destination + ".tmp", "w", encoding="utf-8") as handle:
                json.dump({"segments": self.segments}, handle, ensure_ascii=False, indent=2)
            filesystem.replace(destination + ".tmp", destination)
        except (OSError, TypeError, ValueError) as exc:
            self.log.emit(f"No se pudo guardar el progreso TTS: {exc}")

    def _repair_segment_text(self, seg, previous):
        source = (seg.get('source_text') or seg.get('asr_text') or '').strip()
        if not source:
            return None
        from core.ollama_translator import OllamaTranslator
        translator = OllamaTranslator(host=config_manager.get('ollama_host', 'http://localhost:11434'))
        try:
            if not translator.is_available():
                self.log.emit("Ollama no está disponible para reparar la frase; se conserva el texto anterior.")
                return None
            candidate = translator.repair_for_dubbing(source, seg['end']-seg['start'], previous)
            return candidate if candidate != previous else None
        except Exception as exc:
            self.log.emit(f"No se pudo reparar automáticamente la frase: {exc}")
            return None
        finally:
            translator.close()

    def _generate_segment(self, provider, i, seg, text, signature):
        import uuid
        original_start, original_end = seg['start'], seg['end']
        target_dur = (round(original_end * 48000) - round(original_start * 48000)) / 48000
        raw_audio = seg.get('tts_raw_audio')
        if not (seg.get('tts_raw_text') == text and seg.get('tts_signature') == signature
                and self._valid_audio(raw_audio)):
            raw_audio = os.path.join(self.temp_dir, f"tts_raw_{i}_{uuid.uuid4().hex}.wav")
            messages = []
            def report(message):
                messages.append(str(message))
                self.log.emit(f"  Segmento {i+1}: {message}")
            success = provider.synthesize(text=text, output_path=raw_audio, voice_id=self.voice_name,
                                          target_duration=target_dur, log=report)
            if not success or not self._valid_audio(raw_audio):
                raise RuntimeError(messages[-1] if messages else "La síntesis devolvió audio ausente, vacío o inválido.")
            seg.update(tts_raw_audio=raw_audio, tts_raw_text=text, tts_signature=signature)
        else:
            self.log.emit(f"  Segmento {i+1}: reutilizando síntesis válida para ajustar la sincronización.")
        raw_duration = get_audio_duration(raw_audio)
        start, end = original_start, original_end
        if raw_duration > target_dur:
            extra = raw_duration - target_dur
            prev_end = self.segments[i-1]['end'] if i > 0 else 0.0
            # No extender el último segmento fuera de la duración conocida.
            next_start = self.segments[i+1]['start'] if i+1 < len(self.segments) else end
            before = max(0.0, start - prev_end - .1)
            after = max(0.0, next_start - end - .1)
            expand_before, expand_after = min(before, extra/2), min(after, extra/2)
            remaining = extra - expand_before - expand_after
            add_before = min(before - expand_before, remaining)
            expand_before += add_before
            expand_after += min(after - expand_after, remaining - add_before)
            start, end = start - expand_before, end + expand_after
            target_dur = (round(end * 48000) - round(start * 48000)) / 48000
            if expand_before or expand_after:
                self.log.emit(f"  Segmento {i+1}: aprovechando silencios (-{expand_before:.2f}s, +{expand_after:.2f}s).")
        fit_info = {}
        stretched_audio = os.path.join(self.temp_dir, f"tts_stretch_{i}_{uuid.uuid4().hex}.wav")
        self.log.emit(f"Ajustando segmento {i+1}: voz {raw_duration:.2f}s para intervalo {target_dur:.2f}s...")
        success = time_stretch_audio(raw_audio, target_dur, stretched_audio,
            log=lambda message: self.log.emit(f"  Segmento {i+1}: {message}"),
            fit_info=fit_info, match_duration=True, max_speed=2.0)
        if not success:
            detail = fit_info.get('error', 'Falló la sincronización de audio.')
            if fit_info.get('retryable') is False:
                raise ValueError(detail + " Revise o acorte el texto del segmento; no se recortará la voz.")
            raise RuntimeError(detail)
        if not self._valid_audio(stretched_audio, target_dur):
            raise RuntimeError("El audio sincronizado está vacío o su duración no coincide con el intervalo.")
        seg.update(fit_info)
        seg.update(start=start, end=end, tts_audio=stretched_audio, tts_raw_duration=raw_duration,
                   tts_target_duration=target_dur, tts_start=start, tts_end=end, tts_text=text,
                   tts_signature=signature)
        if seg.get('final_text') != text:
            seg['tts_previous_text'] = seg.get('final_text', '')
            seg['final_text'] = text
            seg['translation_source'] = 'reparacion_doblaje'
        seg.pop('tts_error', None)
        self.log.emit(f"  Segmento {i+1} sincronizado: {start:.3f}s a {end:.3f}s.")

    def run(self):
        import time
        try:
            from core.tts_provider import SapiTTSProvider
            if self.tts_engine == "openspeaker":
                from core.openspeaker_provider import OpenSpeakerTTSProvider
                provider = OpenSpeakerTTSProvider(self.api_key, self.api_speed)
                self.voice_name = self.api_voice
            else:
                provider = SapiTTSProvider(voice_name=self.voice_name)
            signature = [self.tts_engine, self.voice_name, self.api_speed if self.tts_engine == 'openspeaker' else None]
            total, failures, reused = len(self.segments), [], 0
            consecutive_failures = 0
            checkpoint_dirty = False
            for i, seg in enumerate(self.segments):
                text = seg.get('final_text', '').strip()
                if not text:
                    clear_segment_tts(seg)
                    self.progress.emit(int((i+1) / total * 100))
                    continue
                try:
                    start, end = float(seg['start']), float(seg['end'])
                    if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
                        raise ValueError("Intervalo inválido; corrija los tiempos antes de reintentar.")
                    target = (round(end*48000) - round(start*48000)) / 48000
                    if (seg.get('tts_signature') == signature and seg.get('tts_text') == text and
                            seg.get('tts_start') == start and seg.get('tts_end') == end and
                            self._valid_audio(seg.get('tts_audio'), target)):
                        consecutive_failures = 0
                        reused += 1
                        self.log.emit(f"Segmento {i+1}/{total}: voz válida conservada.")
                    else:
                        # Conservar la síntesis original si todavía coincide con texto y voz.
                        raw_cache = {key: seg[key] for key in ('tts_raw_audio', 'tts_raw_text', 'tts_signature') if key in seg}
                        clear_segment_tts(seg)
                        seg.update(raw_cache)
                        for attempt in range(1, 4):
                            try:
                                self.log.emit(f"Generando voz para segmento {i+1}/{total}, intento {attempt}/3...")
                                self._generate_segment(provider, i, seg, text, signature)
                                checkpoint_dirty = True
                                consecutive_failures = 0
                                break
                            except Exception as exc:
                                seg['tts_error'] = str(exc)
                                self.log.emit(f"Error en segmento {i+1}, intento {attempt}/3: {exc}")
                                if isinstance(exc, ValueError) and 'requiere una velocidad' in str(exc) and attempt < 3:
                                    self.log.emit(f"Reparando traducción del segmento {i+1} desde el texto original...")
                                    corrected = self._repair_segment_text(seg, text)
                                    if corrected:
                                        self.log.emit(f"Texto propuesto para segmento {i+1}: {corrected}")
                                        text = corrected
                                        continue
                                permanent = isinstance(exc, ValueError) or any(code in str(exc) for code in (
                                    'HTTP 401', 'HTTP 402', 'HTTP 403', 'Control de aplicaciones'))
                                if permanent or attempt == 3:
                                    raise
                                delay = .5 * 2**(attempt-1)
                                self.log.emit(f"Reintentando segmento {i+1} en {delay:.1f}s...")
                                time.sleep(delay)
                except Exception as exc:
                    seg.pop('tts_audio', None)
                    seg['tts_error'] = str(exc)
                    failures.append(i+1)
                    checkpoint_dirty = True
                    self.log.emit(f"Segmento {i+1} pendiente: {exc}")
                    consecutive_failures += 1
                    if any(code in str(exc) for code in ('HTTP 401', 'HTTP 402', 'HTTP 403', 'Control de aplicaciones')):
                        raise RuntimeError(f"Proceso detenido en segmento {i+1}: {exc}. "
                                           "Corrija el acceso o los créditos antes de reintentar. "
                                           "Las voces válidas se conservaron.") from exc
                    if consecutive_failures >= 5:
                        raise RuntimeError("Se detuvo la generación tras cinco segmentos fallidos consecutivos. "
                                           "Revise el proveedor y los errores registrados antes de reintentar. "
                                           "Las voces válidas se conservaron.") from exc
                self.progress.emit(int((i+1) / total * 100))
                if checkpoint_dirty and ((failures and failures[-1] == i+1) or (i+1) % 10 == 0):
                    self._save_checkpoint()
                    checkpoint_dirty = False
            self._save_checkpoint()
            if failures:
                raise RuntimeError(
                    f"Quedaron pendientes los segmentos {failures}. Las voces correctas se conservaron. "
                    "Pulse 'Generar Voces' para reintentar solo los pendientes. Si la frase no cabe, "
                    "acorte su Texto Final y reintente. Consulte el registro y tts_progreso.json en " + self.temp_dir)
            if not any(s.get('tts_audio') for s in self.segments):
                raise RuntimeError("No hay texto traducido para generar las voces.")
            self.log.emit(f"Generación completada; {reused} voces válidas reutilizadas.")
            self.finished.emit(self.segments)
        except Exception as exc:
            self._save_checkpoint()
            logger.exception("Error en TTSWorker")
            self.error.emit(str(exc))


class RenderWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(str) # Ruta final del video
    error = Signal(str)

    def __init__(self, video_path: str, segments: List[Dict], temp_dir: str, output_path: str):
        super().__init__()
        self.video_path = video_path
        self.segments = segments
        self.temp_dir = temp_dir
        self.output_path = output_path
        self.original_volume = config_manager.get("original_audio_volume", 0.5)
        self.tts_volume = config_manager.get("tts_volume", 1.0)
        self.remove_original_voice = config_manager.get("remove_original_voice", False)

    def run(self):
        try:
            self.log.emit("Iniciando renderizado final...")
            
            info = get_video_info(self.video_path)
            duration = info['duration']
            if duration <= 0:
                self.error.emit("Duración de video inválida.")
                return

            self.progress.emit(10)
            self.log.emit("Extrayendo audio original...")
            original_audio = os.path.join(self.temp_dir, "original_audio.wav")
            if not extract_audio(self.video_path, original_audio, for_mixing=True):
                self.error.emit("Fallo extrayendo audio original.")
                return

            mixing_audio = original_audio
            background_volume = self.original_volume
            if self.remove_original_voice:
                from core.voice_separator import separate_original_voice
                vocal_audio = os.path.join(self.temp_dir, "original_voice.wav")
                mixing_audio = os.path.join(self.temp_dir, "background_without_voice.wav")
                separate_original_voice(original_audio, vocal_audio, mixing_audio, log=self.log.emit,
                                        model_name=config_manager.get("separation_model", "UVR-MDX-NET-Inst_HQ_4"))
                background_volume = 1.0

            self.progress.emit(30)
            self.log.emit("Mezclando audio de doblaje y original...")
            # Encode to AAC only once, when muxing into the final MP4.
            dubbed_audio = os.path.join(self.temp_dir, "dubbed_audio.wav")
            if not build_dubbed_audio(self.segments, mixing_audio, duration, dubbed_audio, self.temp_dir,
                                      original_volume=background_volume, tts_volume=self.tts_volume,
                                      preserve_stereo=True, preserve_background=self.remove_original_voice):
                self.error.emit("Fallo en la mezcla de audio.")
                return

            self.progress.emit(70)
            self.log.emit("Construyendo video final...")
            if not render_final_video(self.video_path, dubbed_audio, self.output_path):
                self.error.emit("Fallo renderizando el video final.")
                return

            self.progress.emit(100)
            self.log.emit(f"Renderizado completado: {self.output_path}")
            self.finished.emit(self.output_path)

        except Exception as e:
            logger.exception("Error en RenderWorker")
            self.error.emit(str(e))
