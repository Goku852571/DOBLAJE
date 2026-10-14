"""Recuperación de voces validada contra el guion ASR cargado."""
import json
from pathlib import Path
from core.analysis_store import clean_segments
from core.media import clear_segment_tts, get_audio_duration


def load_tts_progress(current_segments, source):
    path = Path(source)
    if path.stat().st_size > 100*1024*1024:
        raise ValueError('El progreso de voces supera 100 MB.')
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    saved = payload.get('segments') if isinstance(payload, dict) else None
    clean_segments(saved)
    if len(saved) != len(current_segments):
        raise ValueError('El progreso no pertenece al análisis cargado: distinto número de segmentos.')
    for index, (current, recovered) in enumerate(zip(current_segments, saved), 1):
        current_source = current.get('source_text') or current.get('asr_text') or ''
        recovered_source = recovered.get('source_text') or recovered.get('asr_text') or ''
        if (not current_source or current_source != recovered_source or
                current.get('words', []) != recovered.get('words', [])):
            raise ValueError(f'El progreso no coincide con el ASR del segmento {index}.')
        if not current.get('words') and (abs(current['start']-recovered['start']) > .05 or abs(current['end']-recovered['end']) > .05):
            raise ValueError(f'Los tiempos del segmento {index} no coinciden con el análisis.')
    restored = json.loads(json.dumps(saved))
    valid = 0
    for seg in restored:
        audio = seg.get('tts_audio')
        if not audio:
            continue
        try:
            duration = get_audio_duration(audio) if Path(audio).is_file() else 0
            expected = (round(seg['end']*48000)-round(seg['start']*48000))/48000
            good = (duration > 0 and abs(duration-expected) <= 2/48000 and
                    seg.get('tts_text') == seg['final_text'].strip() and
                    seg.get('tts_start') == seg['start'] and seg.get('tts_end') == seg['end'])
        except (OSError, ValueError, TypeError):
            good = False
        if good:
            valid += 1
        else:
            clear_segment_tts(seg)
            seg['tts_error'] = 'Audio ausente o inválido al recuperar; regenere este segmento.'
    return restored, valid
