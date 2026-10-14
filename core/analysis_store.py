"""Persistencia local y validación del análisis de cada video."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'asr'


def video_identity(video):
    path = Path(video).resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        digest.update(handle.read(65536))
        handle.seek(max(0, stat.st_size-65536))
        digest.update(handle.read(65536))
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sample_sha256': digest.hexdigest()}


def cache_path(video):
    key = hashlib.sha256(os.path.normcase(str(Path(video).resolve())).encode('utf-8')).hexdigest()
    return CACHE_DIR / (key + '.json')


def clean_segments(segments):
    if not isinstance(segments, list) or not segments:
        raise ValueError('El análisis no contiene segmentos.')
    clean = []
    for index, source in enumerate(segments, 1):
        if not isinstance(source, dict):
            raise ValueError(f'Segmento {index} inválido.')
        item = {key: value for key, value in source.items() if not key.startswith('tts_')}
        for key in ('start', 'end'):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'Tiempo {key} inválido en segmento {index}.')
        if not 0 <= item['start'] < item['end']:
            raise ValueError(f'Intervalo inválido en segmento {index}.')
        for key in ('asr_text', 'source_text', 'final_text'):
            if key not in item:
                item[key] = item.get('asr_text', '') if key == 'source_text' else ''
            if not isinstance(item[key], str):
                raise ValueError(f'Texto inválido en segmento {index}.')
        words = item.get('words', [])
        if not isinstance(words, list):
            raise ValueError(f'Palabras inválidas en segmento {index}.')
        for word in words:
            if not isinstance(word, dict):
                raise ValueError(f'Palabra inválida en segmento {index}.')
            for key in ('start', 'end'):
                value = word.get(key)
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f'Tiempo de palabra inválido en segmento {index}.')
            if not 0 <= word['start'] <= word['end']:
                raise ValueError(f'Intervalo de palabra inválido en segmento {index}.')
        clean.append(item)
    return sorted(clean, key=lambda segment: (segment['start'], segment['end']))


def save_analysis(video, segments, destination=None):
    payload = {'format': 'video_sub_analysis', 'version': 1, 'video_name': Path(video).name,
               'video': video_identity(video), 'segments': clean_segments(segments)}
    # Validar todo antes de tocar el archivo anterior.
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    destination = Path(destination) if destination else cache_path(video)
    if destination.resolve() == Path(video).resolve():
        raise ValueError('El análisis no puede sobrescribir el video original.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.tmp',
                                         dir=destination.parent, delete=False) as handle:
            pending = Path(handle.name)
            handle.write(encoded)
        os.replace(pending, destination)
    finally:
        if pending and pending.exists():
            pending.unlink()
    return destination


def load_analysis(video, source=None):
    source = Path(source) if source else cache_path(video)
    if source.stat().st_size > 100 * 1024 * 1024:
        raise ValueError('El archivo de análisis supera 100 MB.')
    try:
        payload = json.loads(source.read_text(encoding='utf-8-sig'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('El archivo de análisis no contiene JSON válido.') from exc
    if not isinstance(payload, dict) or payload.get('format') != 'video_sub_analysis' or payload.get('version') != 1:
        raise ValueError('Formato de análisis no compatible. Use un archivo de Guardar análisis.')
    if payload.get('video') != video_identity(video):
        raise ValueError('Este análisis pertenece a otro video o el video fue modificado. Analícelo nuevamente.')
    return clean_segments(payload.get('segments'))
