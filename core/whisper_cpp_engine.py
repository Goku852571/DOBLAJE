"""Offline whisper.cpp CLI adapter; timestamps stay on the original audio timeline."""
import json
import math
import re
import subprocess
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class WhisperCppEngine:
    def __init__(self, executable="tools/whisper_cpp/whisper-cli.exe",
                 model_path="models/whisper_cpp/ggml-large-v3-turbo.bin",
                 vad_path="tools/whisper_cpp/ggml-silero-v6.2.0.bin", threads=4):
        self.executable = local_path(executable)
        self.model_path = local_path(model_path)
        self.vad_path = local_path(vad_path)
        self.threads = max(1, int(threads))
        self.detected_language = None
        self.language_probability = None

    @staticmethod
    def parse_vad_mapping(line):
        # This binary writes token offsets on the filtered timeline, while its
        # segment offsets are already mapped. Capture the actual map, not an
        # approximation based on subtitle text or on a second VAD execution.
        match = re.search(r"vad_segment_info: orig_start: ([\d.]+), orig_end: ([\d.]+), "
                          r"vad_start: ([\d.]+), vad_end: ([\d.]+)", line)
        return tuple(map(float, match.groups())) if match else None

    @staticmethod
    def original_time(seconds, mapping):
        for index, (original_start, original_end, filtered_start, filtered_end) in enumerate(mapping):
            if filtered_start <= seconds <= filtered_end:
                return original_start + (seconds - filtered_start) * (
                    original_end - original_start) / (filtered_end - filtered_start)
            if seconds < filtered_start:
                if index == 0:
                    return original_start
                previous = mapping[index - 1]
                # A token in the artificial gap snaps to a real speech edge.
                return previous[1] if seconds - previous[3] <= filtered_start - seconds else original_start
        return mapping[-1][1]

    @classmethod
    def original_interval(cls, start, end, mapping):
        # A word crossing the artificial inter-chunk silence belongs to the
        # speech chunk containing most of its tokens, not to the real gap.
        region = max(mapping, key=lambda m: max(0.0, min(end, m[3]) - max(start, m[2])))
        if min(end, region[3]) <= max(start, region[2]):
            point = cls.original_time((start + end) / 2, mapping)
            region = min(mapping, key=lambda m: min(abs(point - m[0]), abs(point - m[1])))
            point = min(max(point, region[0]), region[1] - 0.01)
            return point, point + 0.01
        original_start, original_end, filtered_start, filtered_end = region
        scale = (original_end - original_start) / (filtered_end - filtered_start)
        return (original_start + (max(start, filtered_start) - filtered_start) * scale,
                original_start + (min(end, filtered_end) - filtered_start) * scale)

    @classmethod
    def parse_result(cls, payload, duration, vad_mapping=None):
        results = []
        language = payload.get("result", {}).get("language")
        previous_end = 0.0
        for item in payload.get("transcription", []):
            text = item.get("text", "").strip()
            if not text or text.startswith("[_"):
                continue
            if vad_mapping is not None:
                if not vad_mapping:
                    raise RuntimeError("Whisper CPP no proporcionó el mapa temporal VAD para alinear el habla.")
                aligned = cls.aligned_token_groups(item, duration, vad_mapping)
                for group in aligned:
                    group['start'] = max(previous_end, group['start'])
                    if group['end'] <= group['start']:
                        raise RuntimeError("Whisper CPP devolvió una frase sin intervalo de habla válido.")
                    group.update(language=language, language_probability=None)
                    for word in group['words']:
                        word['start'] = max(group['start'], word['start'])
                        word['end'] = max(word['start'], word['end'])
                    results.append(group)
                    previous_end = group['end']
                continue
            offsets = item["offsets"]
            start, end = float(offsets["from"]) / 1000, float(offsets["to"]) / 1000
            if not math.isfinite(start) or not math.isfinite(end):
                raise RuntimeError("Whisper CPP devolvió tiempos inválidos.")
            start = min(duration, max(0.0, previous_end, start))
            end = min(duration, end)
            if end <= start:
                continue
            results.append(dict(start=start, end=end, text=text, words=[],
                                language=language, language_probability=None))
            previous_end = end
        if vad_mapping is not None:
            cls.merge_short_groups(results, vad_mapping)
        return results

    @staticmethod
    def phrase_speech_group(item, duration, mapping):
        """Use native segment times when the decoder cannot align its tokens."""
        try:
            offsets = item["offsets"]
            start, end = float(offsets["from"]) / 1000, float(offsets["to"]) / 1000
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Whisper CPP no proporcionó un intervalo de frase válido.") from exc
        if not math.isfinite(duration) or duration <= 0 or not all(
                math.isfinite(t) for t in (start, end)) or end <= start:
            raise RuntimeError("Whisper CPP no proporcionó un intervalo de frase válido.")
        regions = [m for m in mapping if len(m) == 4 and all(math.isfinite(t) for t in m)
                   and m[1] > m[0] and m[3] > m[2]]
        if not regions:
            raise RuntimeError("Whisper CPP no proporcionó una región de habla VAD válida.")
        primary = max(regions, key=lambda m: max(0.0, min(end, m[1], duration) - max(start, m[0], 0.0)))
        start, end = max(0.0, start, primary[0]), min(end, primary[1], duration)
        if end <= start:
            raise RuntimeError("No se pudo alinear la frase con un intervalo de habla real.")
        return [dict(start=start, end=end, text=item["text"].strip(),
                     words=[], timing_source="vad_phrase")]

    @classmethod
    def aligned_token_groups(cls, item, duration, mapping):
        words = []
        # Keep punctuation in the text, but never let its synthetic duration
        # prolong a spoken word through silence. Join subword tokens first.
        for token in item.get('tokens', []):
            piece = token.get('text', '')
            if not piece or piece.startswith('[_'):
                continue
            lexical = any(char.isalnum() for char in piece)
            if not lexical:
                if words:
                    words[-1]['word'] += piece
                continue
            offsets = token.get('offsets') or {}
            try:
                start, end = float(offsets.get('from', -1)) / 1000, float(offsets.get('to', -1)) / 1000
            except (TypeError, ValueError, OverflowError):
                return cls.phrase_speech_group(item, duration, mapping)
            if not all(math.isfinite(t) and t >= 0 for t in (start, end)) or end < start:
                return cls.phrase_speech_group(item, duration, mapping)
            no_spaces = any('\u3400' <= char <= '\u9fff' or '\u3040' <= char <= '\u30ff' for char in piece)
            if words and not piece[0].isspace() and not no_spaces:
                words[-1]['word'] += piece
                words[-1]['end'] = max(words[-1]['end'], end)
            else:
                words.append(dict(word=piece, start=start, end=end, probability=token.get('p')))
        if not words:
            return cls.phrase_speech_group(item, duration, mapping)
        collapsed = sum(word['end'] - word['start'] <= .03 for word in words)
        for word in words:
            start, end = cls.original_interval(word['start'], word['end'], mapping)
            word['start'] = max(0.0, min(duration, start))
            word['end'] = max(word['start'], min(duration, end))
        # Do not silently lose quotes or other prefix punctuation.
        reconstructed = ''.join(word['word'] for word in words).strip()
        text = item['text'].strip()
        if ''.join(reconstructed.split()) != ''.join(text.split()):
            prefix = text[:text.find(words[0]['word'].strip())] if words[0]['word'].strip() in text else ''
            words[0]['word'] = prefix + words[0]['word'].lstrip()
            if ''.join(''.join(w['word'] for w in words).split()) != ''.join(text.split()):
                raise RuntimeError("Los tokens alineados no conservan el texto reconocido por Whisper CPP.")
        offsets = item['offsets']
        native_start, native_end = float(offsets['from']) / 1000, float(offsets['to']) / 1000
        primary = max(mapping, key=lambda m: max(0.0, min(native_end, m[1]) - max(native_start, m[0])))
        # Some decoder windows assign identical times to several spoken tokens.
        # In that case retain the complete phrase in its actual speech window;
        # never emit a 10 ms multi-word phrase or invent word-level precision.
        if collapsed >= max(1, math.ceil(len(words) * .2)):
            return cls.phrase_speech_group(item, duration, mapping)
        # Keep a leading boundary word with the resumed phrase. Token heuristics
        # sometimes leave its onset in the final 100 ms of the previous chunk.
        leading = []
        for word in words:
            if word['end'] <= primary[0] and word['start'] < primary[0]:
                leading.append(word)
            else:
                break
        if leading:
            following = words[len(leading)] if len(leading) < len(words) else None
            limit = min(primary[1], max(primary[0] + .01, following['start']) if following else primary[1])
            if limit > primary[0]:
                for word in leading:
                    word['start'], word['end'] = primary[0], limit
        groups, current = [], []
        def flush():
            if current:
                groups.append(dict(start=current[0]['start'], end=max(w['end'] for w in current),
                                   text=''.join(w['word'] for w in current).strip(),
                                   words=[dict(w, word=w['word'].strip()) for w in current]))
                current.clear()
        for word in words:
            if current and (word['start'] - current[-1]['end'] >= 0.35 or
                            word['end'] - current[0]['start'] > 8.0):
                flush()
            current.append(word)
        flush()
        cls.merge_short_groups(groups, mapping)
        return groups

    @staticmethod
    def merge_short_groups(groups, mapping):
        # A degenerate one-word tail must not get its own un-dubbable interval.
        # Merge only inside the same VAD speech region, never across real silence.
        for index in range(len(groups) - 1, -1, -1):
            group = groups[index]
            if group['end'] - group['start'] >= .25 or len(groups) < 2:
                continue
            neighbor_index = index + 1 if index + 1 < len(groups) else index - 1
            neighbor = groups[neighbor_index]
            lo, hi = min(group['start'], neighbor['start']), max(group['end'], neighbor['end'])
            if hi - lo > 8 or not any(m[0] <= lo and hi <= m[1] for m in mapping):
                continue
            first, second = (group, neighbor) if index < neighbor_index else (neighbor, group)
            neighbor.update(start=lo, end=hi, text=first['text'] + ' ' + second['text'],
                            words=first['words'] + second['words'])
            groups.pop(index)

    def extract_audio_text(self, audio_path, progress_callback=None, language="auto"):
        self.detected_language = None
        self.language_probability = None
        for path in (self.executable, self.model_path, self.vad_path):
            if not path.is_file():
                raise RuntimeError(f"Archivo local de Whisper CPP no encontrado: {path}")
        if self.model_path.name.endswith(".en.bin") and language not in ("en", "auto", None, ""):
            raise RuntimeError("El modelo .en solo admite audio en inglés.")
        with wave.open(str(audio_path), "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                raise RuntimeError("Whisper CPP requiere WAV PCM mono de 16 bits a 16 kHz.")
            duration = audio.getnframes() / 16000
        if duration == 0:
            if progress_callback:
                progress_callback(100)
            return []
        with tempfile.TemporaryDirectory(prefix="videodub_cpp_") as folder:
            output = Path(folder) / "transcript"
            command = [str(self.executable), "--model", str(self.model_path),
                       "--file", str(Path(audio_path).resolve()), "--language", language or "auto",
                       "--threads", str(self.threads), "--no-gpu", "--max-context", "0",
                       "--beam-size", "5", "--best-of", "5", "--temperature", "0",
                       "--max-len", "80", "--split-on-word", "--output-json-full",
                       "--output-file", str(output), "--print-progress", "--vad",
                       "--vad-model", str(self.vad_path), "--vad-threshold", "0.45",
                       "--vad-min-silence-duration-ms", "2000", "--vad-min-speech-duration-ms", "0"]
            tail, vad_mapping = [], []
            with subprocess.Popen(command, cwd=str(self.executable.parent),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace",
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
                for line in process.stdout:
                    entry = self.parse_vad_mapping(line)
                    if entry:
                        vad_mapping.append(entry)
                    tail.append(line.rstrip())
                    tail = tail[-30:]
                    match = re.search(r"progress\s*=\s*(\d+)%", line)
                    if match and progress_callback:
                        progress_callback(min(99, int(match.group(1))))
                code = process.wait()
            if code:
                raise RuntimeError("Whisper CPP falló: " + "\n".join(tail)[-3000:])
            result_file = output.with_suffix(".json")
            if not result_file.is_file():
                raise RuntimeError("Whisper CPP no generó la transcripción JSON.")
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            self.detected_language = payload.get("result", {}).get("language")
            results = self.parse_result(payload, duration, vad_mapping=vad_mapping)
        if progress_callback:
            progress_callback(100)
        return results
