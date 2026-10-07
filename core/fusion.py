from typing import List, Dict

def calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate the overlap duration between two time segments."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0.0, overlap_end - overlap_start)

def fuse_segments(ocr_results: List[Dict], asr_results: List[Dict]) -> List[Dict]:
    """
    Merge OCR and ASR segments by time overlap.
    Returns a unified list of segments to be displayed in the UI.
    """
    fused = []
    
    # We will iterate mainly over ASR segments, and find overlapping OCR texts.
    # ASR is usually more reliable for audio boundaries.
    
    # Track used OCR segments so we can append unused ones later
    used_ocr = [False] * len(ocr_results)
    
    for asr in asr_results:
        asr_start = asr['start']
        asr_end = asr['end']
        asr_text = asr['text']
        
        overlapping_ocr_texts = []
        
        for i, ocr in enumerate(ocr_results):
            ocr_start = ocr['start']
            ocr_end = ocr['end']
            
            overlap = calculate_overlap(asr_start, asr_end, ocr_start, ocr_end)
            # Match if there is any temporal overlap or timestamp alignment
            if overlap > 0.05 or (asr_start <= ocr_start <= asr_end) or (ocr_start <= asr_start <= ocr_end):
                overlapping_ocr_texts.append(ocr['text'])
                used_ocr[i] = True
                
        ocr_combined = " | ".join(overlapping_ocr_texts) if overlapping_ocr_texts else ""
        
        fused.append({
            "start": asr_start,
            "end": asr_end,
            "ocr_text": ocr_combined,
            "asr_text": asr_text,
            "words": asr.get("words", []),
            "language": asr.get("language"),
            "language_probability": asr.get("language_probability", 0.0),
            "final_text": asr_text if asr_text else ocr_combined, # Default to ASR text
            "source_text": asr_text if asr_text else ocr_combined  # Texto original para traducción
        })
        
    # Add OCR segments that didn't overlap with any ASR (maybe silent reading)
    for i, ocr in enumerate(ocr_results):
        if not used_ocr[i]:
            fused.append({
                "start": ocr['start'],
                "end": ocr['end'],
                "ocr_text": ocr['text'],
                "asr_text": "",
                "final_text": ocr['text'],
                "source_text": ocr['text']  # Texto original para traducción
            })
            
    # Sort by start time
    fused.sort(key=lambda x: x['start'])
    
    return fused
