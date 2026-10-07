"""Local Faster Whisper/Silero transcription on the original timeline."""
import math
import re
from pathlib import Path
from typing import List, Dict, Callable

from faster_whisper import WhisperModel, decode_audio
from faster_whisper.vad import get_speech_timestamps, VadOptions
from utils.logger import logger


class ASREngine:
    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cpu",
                 compute_type: str = "default", pause_threshold: float = 0.35,
                 model_path: str = None):
        # English-only weights cannot recognize Chinese videos.
        self.model_size = model_size.removesuffix(".en")
        self.device = device
        self.compute_type = compute_type
        self.pause_threshold = pause_threshold
        self.model_path = model_path
        self.model = None
        self.detected_language = None
        self.language_probability = 0.0

    def initialize(self):
        if self.model is not None:
            return
        logger.info("Cargando Whisper multilingüe: %s (%s, %s).",
                    self.model_size, self.device, self.compute_type)
        model_source = self.model_size
        if self.model_path:
            model_source = str(Path(self.model_path) if Path(self.model_path).is_absolute()
                               else Path(__file__).resolve().parents[1] / self.model_path)
        elif self.model_size in ("large-v3-turbo", "turbo"):
            model_source = str(Path(__file__).resolve().parents[1] / "models" /
                               "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo")
        if self.model_path or self.model_size in ("large-v3-turbo", "turbo"):
            for filename in ("model.bin", "config.json", "tokenizer.json"):
                if not (Path(model_source) / filename).is_file():
                    raise RuntimeError(f"Modelo ASR local incompleto: {model_source}/{filename}")
        try:
            self.model = WhisperModel(
                model_source, device=self.device,
                compute_type=self.compute_type, num_workers=1, local_files_only=True,
            )
        except RuntimeError as exc:
            # Never silently replace the requested quality with a smaller model.
            raise RuntimeError(f"No se pudo cargar el modelo ASR {self.model_size}: {exc}") from exc

    def detect_language(self, audio):
        """Inspect up to three voiced windows before running transcription."""
        language, probability, probabilities = self.model.detect_language(
            audio=audio, vad_filter=True,
            vad_parameters=VadOptions(min_silence_duration_ms=2000, min_speech_duration_ms=0, threshold=0.45),
            language_detection_segments=3, language_detection_threshold=0.85,
        )
        scores = dict(probabilities)
        logger.info("Detección previa: chino %.1f%%; inglés %.1f%%.",
                    100 * scores.get("zh", 0), 100 * scores.get("en", 0))
        if probability < 0.5:
            logger.warning("Detección de idioma poco segura (%s, %.1f%%). "
                           "Puede seleccionar chino o inglés en la interfaz.",
                           language, probability * 100)
        return language, probability

    def extract_audio_text(self, audio_path: str,
                           progress_callback: Callable[[int], None] = None,
                           language: str = "auto") -> List[Dict]:
        self.initialize()
        self.detected_language = None
        self.language_probability = 0.0
        audio = decode_audio(audio_path, sampling_rate=16000)
        duration = len(audio) / 16000
        if not duration:
            if progress_callback:
                progress_callback(100)
            return []
        if progress_callback:
            progress_callback(5)
        vad_parameters = dict(min_silence_duration_ms=2000, min_speech_duration_ms=0, threshold=0.45)
        if not get_speech_timestamps(audio, vad_options=VadOptions(**vad_parameters)):
            logger.info("No se detectó habla en el audio; no se generará una transcripción de silencio.")
            if progress_callback:
                progress_callback(100)
            return []
        automatic_language = language in (None, "", "auto")
        if not automatic_language:
            language = language.lower().split("-")[0]
            logger.info("Idioma ASR seleccionado manualmente: %s.", language)
        if progress_callback:
            progress_callback(15)
        # Transcribe the original array. faster-whisper restores the original
        # timestamps after VAD; do not shift them a second time here.
        # Parameters match pyvideotrans/process/stt_faster.py. Detect the
        # language in this same call, rather than fixing an earlier guess.
        segments, info = self.model.transcribe(
            audio, language=None if automatic_language else language, task="transcribe",
            beam_size=5, best_of=5, temperature=0.0, condition_on_previous_text=False,
            word_timestamps=True, vad_filter=True,
            vad_parameters=vad_parameters, no_speech_threshold=0.6,
            repetition_penalty=1.0, compression_ratio_threshold=2.4,
            hotwords=None, initial_prompt=None,
        )
        language = info.language
        probability = info.language_probability
        self.detected_language = language
        self.language_probability = probability
        logger.info("Idioma ASR: %s (confianza %.1f%%).", language, probability * 100)
        raw_segments = []
        for segment in segments:
            raw_segments.append(segment)
            if progress_callback:
                progress_callback(min(95, 15 + int(80 * max(0, segment.end) / duration)))
        results = []
        previous_end = 0.0
        for item in self._resegment(raw_segments, duration, language):
            item["start"] = max(item["start"], previous_end)
            if item["end"] <= item["start"]:
                logger.warning("Omitiendo intervalo ASR sin duración: %s", item["text"])
                continue
            for word in item["words"]:
                word["start"] = max(item["start"], word["start"])
                word["end"] = max(word["start"], word["end"])
            item["language"] = language
            item["language_probability"] = probability
            results.append(item)
            previous_end = item["end"]
            logger.info("ASR [%.3f - %.3f]: %s", item["start"], item["end"], item["text"])
        if progress_callback:
            progress_callback(100)
        return results

    def _aligned_segments(self, segment, duration: float) -> List[Dict]:
        return self._resegment([segment], duration, self.detected_language or "en")

    def _resegment(self, segments, duration: float, language: str) -> List[Dict]:
        """Adapt pyvideotrans/_stt_utils._resegment to the existing seconds schema.

        Group words across Whisper windows, with 3/5-second targets and an
        8-second limit. Preserve this app's sentence endings and real pauses;
        unlike the reference, never merge a short tail across a speech gap.
        """
        no_spaces = language in {"zh", "zh-cn", "zh-tw", "ja", "th", "yue", "ko", "km"}
        end_punc = (".", "?", "!", "。", "？", "！", "\n")
        weak_punc = (",", ";", ":", "，", "；", "：", "、")
        groups, current, output = [], [], []

        def flush():
            if current:
                groups.append(current[:])
                current.clear()

        for segment in segments:
            valid_words = []
            for word in segment.words or []:
                if not word.word.strip() or not all(math.isfinite(t) for t in (word.start, word.end)):
                    continue
                start, end = max(0.0, word.start), min(duration, word.end)
                if end < start:
                    continue
                valid_words.append(dict(start=start, end=end, word=word.word,
                                        probability=word.probability,
                                        _logprob=segment.avg_logprob))
            if not valid_words:
                # Keep text even when just one Whisper window lacks word times.
                flush()
                text = segment.text.strip()
                start, end = max(0.0, segment.start), min(duration, segment.end)
                if text and all(math.isfinite(t) for t in (start, end)) and end > start:
                    output.append(dict(start=start, end=end, text=text,
                                       prob=segment.avg_logprob, words=[]))
                continue
            for index, word in enumerate(valid_words):
                if current:
                    previous = current[-1]
                    gap = word["start"] - previous["end"]
                    current_duration = previous["end"] - current[0]["start"]
                    future_duration = word["end"] - current[0]["start"]
                    text = previous["word"].rstrip()
                    decimal = bool(re.search(r"\d\.?$", text) and
                                   re.match(r"^\.?\d", word["word"].strip()))
                    strong = text.endswith(end_punc) and not decimal
                    weak = text.endswith(weak_punc) or previous.get("_segment_end", False)
                    split = gap >= self.pause_threshold or strong or future_duration >= 8.0
                    if not split and current_duration >= 3.0:
                        split = (gap >= 0.1 and weak) or (weak and current_duration >= 4.0)
                    if split:
                        flush()
                word["_segment_end"] = index == len(valid_words) - 1
                current.append(word)
        flush()
        for group in groups:
            text = ("" if no_spaces else " ").join(w["word"].strip() for w in group)
            if not no_spaces:
                text = re.sub(r"\s+([.,?!:;])", r"\1", text)
            output.append(dict(
                start=group[0]["start"], end=max(w["end"] for w in group), text=text.strip(),
                prob=sum(w["_logprob"] for w in group) / len(group),
                words=[{k: v for k, v in w.items() if not k.startswith("_")} for w in group],
            ))
        return sorted(output, key=lambda item: item["start"])
