import cv2
from paddleocr import PaddleOCR
import numpy as np
from rapidfuzz import fuzz
from typing import List, Dict, Callable, Optional, Tuple
from utils.logger import logger
class OCREngine:
    def __init__(self, languages: Optional[List[str]] = None):
        self.languages = list(languages) if languages is not None else ['ch', 'en']
        self.reader = None

    def _map_language(self, langs: List[str]) -> str:
        """Map input language codes to PaddleOCR's recognized language identifiers."""
        lang_map = {
            'ch_sim': 'ch',
            'ch': 'ch',
            'ch_tra': 'chinese_cht',
            'en': 'en',
            'es': 'es',
            'ja': 'japan',
            'japan': 'japan',
            'ko': 'korean',
            'korean': 'korean',
            'fr': 'fr',
            'de': 'german',
            'ru': 'ru',
        }
        for l in langs:
            if l in lang_map:
                return lang_map[l]
        return 'ch'

    def initialize(self):
        if self.reader is None:
            lang = self._map_language(self.languages)
            logger.info(f"Initializing PaddleOCR with language: {lang}")
            # Try PaddleOCR 3.x parameters first
            try:
                self.reader = PaddleOCR(
                    lang=lang,
                    use_textline_orientation=True,
                    text_det_thresh=0.3,
                    text_det_box_thresh=0.5,
                    text_rec_score_thresh=0.3,
                    ocr_version="PP-OCRv4",
                    enable_mkldnn=False
                )
            except Exception as e1:
                logger.debug(f"PaddleOCR 3.x init failed ({e1}), attempting 2.x parameters...")
                try:
                    self.reader = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                        show_log=False,
                        det_db_thresh=0.3,
                        det_db_box_thresh=0.5,
                        ocr_version="PP-OCRv4",
                        enable_mkldnn=False
                    )
                except Exception as e2:
                    logger.warning(f"PaddleOCR parameter init warning ({e2}), falling back to minimal init.")
                    self.reader = PaddleOCR(lang=lang)

    def _preprocess_subtitle_roi(self, roi: np.ndarray) -> List[np.ndarray]:
        """Provide original color, enhanced contrast, and binary representations for optimal OCR."""
        if roi.size == 0:
            return []
        
        height, width = roi.shape[:2]
        scale = max(1.0, min(2.5, 1920.0 / max(width, 1)))
        if scale > 1.0:
            resized = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            resized = roi

        # 1. Direct BGR image (optimal for colored/outlined video subtitles)
        candidates = [resized]

        # 2. Enhanced Grayscale CLAHE (optimal for low contrast subtitles)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
        enhanced_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        candidates.append(cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR))

        # 3. Adaptive Threshold Binary (optimal for stylized background noise)
        binary = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 7,
        )
        candidates.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

        return candidates

    def _parse_ocr_lines(self, ocr_result, confidence_threshold: float) -> List[Dict]:
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

    def extract_subtitles(
        self,
        video_path: str,
        sample_rate: float,
        confidence_threshold: float,
        progress_callback: Callable[[int], None] = None,
        languages: Optional[List[str]] = None,
        roi_bottom: float = 0.5,
    ) -> List[Dict]:
        # roi_bottom is the normalized top edge of the subtitle region (0 = full frame, 0.5 = lower half).
        if not 0 <= roi_bottom < 1:
            raise ValueError("roi_bottom must be in [0, 1).")
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be positive and finite.")
        if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if languages is not None and list(languages) != self.languages:
            self.languages = list(languages)
            self.reader = None
        self.initialize()

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError("No se pudo abrir el video para OCR.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not np.isfinite(fps) or fps <= 0 or total_frames <= 0:
                raise ValueError("El video no tiene FPS o duración válidos.")
            frame_step = max(1, int(fps * sample_rate))
            sample_interval = frame_step / fps
            raw_results = []
            logger.info(f"Iniciando extracción OCR: {total_frames} frames a {fps:.1f} fps (región inferior desde {int(roi_bottom*100)}%)")

            for frame_idx in range(0, total_frames, frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                height = frame.shape[0]
                roi = frame[int(height * roi_bottom):height, :]
                candidates = []
                for processed in self._preprocess_subtitle_roi(roi):
                    if len(processed.shape) == 2:
                        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    try:
                        ocr_result = self.reader.ocr(processed, cls=True)
                    except Exception:
                        try:
                            ocr_result = self.reader.ocr(processed)
                        except Exception:
                            continue
                    
                    filtered = self._parse_ocr_lines(ocr_result, confidence_threshold)
                    filtered = self._reading_order(filtered)
                    if filtered:
                        candidates.append({
                            'time': frame_idx / fps,
                            'text': ' '.join(item['text'] for item in filtered),
                            'prob': float(np.mean([item['prob'] for item in filtered])),
                        })
                if candidates:
                    raw_results.append(max(candidates, key=lambda item: item['prob'] * len(item['text']) ** 0.5))
                if progress_callback:
                    progress_callback(int(frame_idx / total_frames * 100))
            if progress_callback:
                progress_callback(100)
            segments = self._consolidate_results(raw_results, sample_interval)
            for segment in segments:
                segment['end'] = min(segment['end'], total_frames / fps)
            logger.info(f"Extracción OCR finalizada: {len(segments)} segmentos de texto detectados.")
            return segments
        finally:
            cap.release()

    @staticmethod
    def _reading_order(items):
        """Cluster boxes into lines before sorting left to right."""
        lines = []
        for item in sorted(items, key=lambda i: min(p[1] for p in i['bbox'])):
            ys = [p[1] for p in item['bbox']]
            center, height = (min(ys) + max(ys)) / 2, max(ys) - min(ys)
            for line in lines:
                if abs(center - line['center']) <= max(height, line['height']) * 0.5:
                    line['items'].append(item)
                    break
            else:
                lines.append(dict(center=center, height=height, items=[item]))
        return [item for line in lines
                for item in sorted(line['items'], key=lambda i: min(p[0] for p in i['bbox']))]

    @staticmethod
    def _vote_words(readings):
        """Choose a confidence-weighted medoid, preserving complete observed phrases.

        Positional word voting corrupts sentences when a reading drops a word
        and cannot correct unspaced Chinese text.
        """
        return max(readings, key=lambda candidate: (
            sum(fuzz.ratio(candidate['text'], other['text']) * other['prob']
                for other in readings), candidate['prob']))['text']

    def _consolidate_results(self, raw_results: List[Dict], sample_rate: float) -> List[Dict]:
        """Group consecutive similar readings and choose a representative phrase."""
        if not raw_results:
            return []
        consolidated = []
        group = [raw_results[0]]

        def finish():
            consolidated.append({
                'start': group[0]['time'],
                'end': group[-1]['time'] + sample_rate,
                'text': self._vote_words(group),
                'prob': float(np.mean([item['prob'] for item in group])),
            })

        for item in raw_results[1:]:
            # A missed sample splits groups, including identical text appearing later.
            consecutive = item['time'] - group[-1]['time'] <= sample_rate * 1.5
            similarity = fuzz.ratio(self._vote_words(group), item['text'])
            if consecutive and similarity >= 85:
                group.append(item)
            else:
                finish()
                group = [item]
        finish()
        return consolidated
