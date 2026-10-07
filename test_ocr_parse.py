import cv2
import numpy as np
import pprint
from paddleocr import PaddleOCR

def _parse_ocr_lines(ocr_result, confidence_threshold: float):
    filtered = []
    if not ocr_result:
        return filtered

    items_to_process = ocr_result if isinstance(ocr_result, list) else [ocr_result]
    for item in items_to_process:
        if item is None:
            continue
        
        # Case 1: Dict format (PaddleOCR 3.x / Paddlex / PP-OCRv6)
        if isinstance(item, dict):
            texts = item.get('rec_text', item.get('rec_texts', item.get('transcription', [])))
            scores = item.get('rec_score', item.get('rec_scores', item.get('confidence', [])))
            polys = item.get('dt_polys', item.get('points', item.get('bbox', [])))
            
            if isinstance(texts, str):
                texts = [texts]
            if isinstance(scores, (int, float)):
                scores = [scores]
            
            if isinstance(texts, list):
                for idx, txt in enumerate(texts):
                    txt_str = str(txt).strip()
                    sc = float(scores[idx]) if (isinstance(scores, list) and idx < len(scores)) else 1.0
                    poly = polys[idx] if (isinstance(polys, list) and idx < len(polys)) else []
                    if sc >= confidence_threshold and txt_str:
                        filtered.append({'bbox': poly, 'text': txt_str, 'prob': sc})
            elif 'text' in item or 'transcription' in item:
                txt = str(item.get('text', item.get('transcription', ''))).strip()
                sc = float(item.get('score', item.get('confidence', 1.0)))
                poly = item.get('bbox', item.get('points', []))
                if sc >= confidence_threshold and txt:
                    filtered.append({'bbox': poly, 'text': txt, 'prob': sc})

        # Case 2: List of tuples/dicts [[bbox, (text, score)], ...]
        elif isinstance(item, (list, tuple)):
            for line in item:
                if isinstance(line, dict):
                    txt = str(line.get('transcription', line.get('text', ''))).strip()
                    sc = float(line.get('score', line.get('confidence', 0.0)))
                    poly = line.get('points', line.get('bbox', []))
                    if sc >= confidence_threshold and txt:
                        filtered.append({'bbox': poly, 'text': txt, 'prob': sc})
                elif isinstance(line, (list, tuple)) and len(line) >= 2:
                    bbox = line[0]
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        txt = str(text_info[0]).strip()
                        sc = float(text_info[1])
                    else:
                        txt = str(text_info).strip()
                        sc = 1.0
                    if sc >= confidence_threshold and txt:
                        filtered.append({'bbox': bbox, 'text': txt, 'prob': sc})

    return filtered

img = np.ones((150, 600, 3), dtype=np.uint8) * 255
cv2.putText(img, "TESTING OCR", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

try:
    ocr = PaddleOCR(lang='en', use_angle_cls=True, ocr_version="PP-OCRv4", enable_mkldnn=False)
    res = ocr.ocr(img)
    print("RAW OCR RESULT TYPE:", type(res))
    if isinstance(res, list) and len(res) > 0:
        print("RAW OCR RESULT[0] TYPE:", type(res[0]))
    print("RAW OCR RESULT:")
    pprint.pprint(res)
    
    parsed = _parse_ocr_lines(res, 0.2)
    print("\nPARSED RESULT:")
    pprint.pprint(parsed)
except Exception as e:
    print("Error:", e)
