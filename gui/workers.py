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

    def __init__(self, segments: List[Dict], host: str, target_language: str):
        super().__init__()
        self.segments = segments
        self.host = host
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
            translations = translator.translate_segments_contextual(self.segments)
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
                self.progress.emit(int((i + 1) / total * 100))
            
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

                    translations = translator.translate_segments_contextual(segments)
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
                        self.progress.emit(80 + int((i + 1) / len(segments) * 20))
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

    def run(self):
        try:
            from core.tts_provider import SapiTTSProvider
            if self.tts_engine == "openspeaker":
                from core.openspeaker_provider import OpenSpeakerTTSProvider
                provider = OpenSpeakerTTSProvider(self.api_key, self.api_speed)
                self.voice_name = self.api_voice
            else:
                provider = SapiTTSProvider(voice_name=self.voice_name)
            total = len(self.segments)
            failures = []
            
            for i, seg in enumerate(self.segments):
                clear_segment_tts(seg)
                text = seg.get("final_text", "").strip()
                if not text:
                    self.log.emit(f"Segmento {i+1} vacío, saltando.")
                    continue
                    
                target_dur = (round(seg['end'] * 48000) - round(seg['start'] * 48000)) / 48000
                if not math.isfinite(target_dur) or target_dur <= 0 or seg['start'] < 0:
                    self.log.emit(f"Segmento {i+1} con duración inválida, saltando.")
                    failures.append(i + 1)
                    continue

                self.log.emit(f"Generando voz para segmento {i+1}/{total}...")
                
                raw_audio = os.path.join(self.temp_dir, f"tts_raw_{i}.wav")
                stretched_audio = os.path.join(self.temp_dir, f"tts_stretch_{i}.wav")
                
                success = provider.synthesize(
                    text=text,
                    output_path=raw_audio,
                    voice_id=self.voice_name,
                    target_duration=target_dur,
                    log=lambda msg: self.log.emit(msg)
                )
                
                if success and os.path.exists(raw_audio):
                    raw_duration = get_audio_duration(raw_audio)
                    
                    if raw_duration > target_dur:
                        extra_needed = raw_duration - target_dur
                        prev_end = self.segments[i-1]['end'] if i > 0 else 0.0
                        next_start = self.segments[i+1]['start'] if i + 1 < len(self.segments) else float('inf')
                        
                        margin = 0.1
                        max_expand_before = max(0.0, seg['start'] - prev_end - margin)
                        max_expand_after = max(0.0, next_start - seg['end'] - margin)
                        
                        expand_before = min(max_expand_before, extra_needed / 2.0)
                        expand_after = min(max_expand_after, extra_needed / 2.0)
                        
                        rem = extra_needed - (expand_before + expand_after)
                        if rem > 0:
                            add_before = min(max_expand_before - expand_before, rem)
                            expand_before += add_before
                            rem -= add_before
                        if rem > 0:
                            add_after = min(max_expand_after - expand_after, rem)
                            expand_after += add_after
                            
                        if expand_before > 0 or expand_after > 0:
                            seg['start'] = max(0.0, seg['start'] - expand_before)
                            seg['end'] += expand_after
                            target_dur = (round(seg['end'] * 48000) - round(seg['start'] * 48000)) / 48000
                            self.log.emit(f"  Expansión inteligente: aprovecha silencios (-{expand_before:.2f}s inicio, +{expand_after:.2f}s fin).")
                    
                    self.log.emit(
                        f"Ajustando segmento {i+1}: voz {raw_duration:.2f}s "
                        f"para intervalo {target_dur:.2f}s..."
                    )
                    fit_info = {}
                    if time_stretch_audio(
                        raw_audio, target_dur, stretched_audio,
                        log=lambda message: self.log.emit(f"  {message}"),
                        fit_info=fit_info,
                        match_duration=True,
                        max_speed=2.0,
                    ):
                        seg['tts_audio'] = stretched_audio
                        seg['tts_raw_duration'] = raw_duration
                        seg['tts_target_duration'] = target_dur
                        seg.update(fit_info)
                        seg['tts_start'] = seg['start']
                        seg['tts_end'] = seg['end']
                        seg['tts_text'] = text
                        self.log.emit(f"  Segmento {i+1} sincronizado: {seg['start']:.3f}s a {seg['end']:.3f}s.")
                    else:
                        self.log.emit(f"Error sincronizando el segmento {i+1}; no se mezclará.")
                        failures.append(i + 1)
                else:
                    self.log.emit(f"Fallo síntesis de voz para segmento {i+1}.")
                    failures.append(i + 1)

                self.progress.emit(int(((i+1) / total) * 100))

            if failures:
                raise RuntimeError(f"No se generaron voces para los segmentos {failures}. Genere todas las voces antes de crear el video.")
            if not any(s.get("tts_audio") for s in self.segments):
                raise RuntimeError("No hay texto traducido para generar las voces.")
            self.log.emit("Generación de voces completada.")
            self.finished.emit(self.segments)

        except Exception as e:
            logger.exception("Error en TTSWorker")
            self.error.emit(str(e))


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
