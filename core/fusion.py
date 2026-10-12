from typing import List, Dict


def fuse_segments(asr_results: List[Dict]) -> List[Dict]:
    """Adapt audio transcriptions to the translation and dubbing segment format."""
    segments = []
    for asr in asr_results:
        segments.append({
            "start": asr["start"],
            "end": asr["end"],
            "asr_text": asr["text"],
            "words": asr.get("words", []),
            "language": asr.get("language"),
            "language_probability": asr.get("language_probability", 0.0),
            "final_text": asr["text"],
            "source_text": asr["text"],
        })
    return sorted(segments, key=lambda segment: segment["start"])
