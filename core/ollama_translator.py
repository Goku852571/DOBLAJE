"""Motor de traducción usando Ollama API vía HTTP directo (httpx)."""
import json
import re
import time
from typing import Dict, List, Optional

import httpx

from utils.logger import logger


OLLAMA_URL = "http://localhost:11434/api/generate"

# Modelos y sus temperaturas según especificación
TRANSLATION_MODEL = "demonbyron/HY-MT1.5-1.8B:bf16"
TRANSLATION_TEMPERATURE = 0.2
ANALYSIS_MODEL = "qwen2.5:7b"
ANALYSIS_TEMPERATURE = 0.0
DUBBING_TEMPERATURE = 0.25

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Pausas crecientes entre intentos

ENGLISH_MARKERS = {
    " the ", " and ", " you ", " your ", " this ", " that ", " what ",
    " have ", " with ", " for ", " are ", " is ", " not ", " can ",
    " will ", " just ", " don't ", " i'm ", " it's ", " to ", " of "
}


class OllamaTranslator:
    """Traductor usando Ollama API local (conexión directa httpx)."""

    def __init__(self, host: str = "http://localhost:11434",
                 target_language: str = "español", timeout: int = 120):
        self.host = host.rstrip('/')
        self.generate_url = f"{self.host}/api/generate"
        self.tags_url = f"{self.host}/api/tags"
        self.target_language = target_language
        self.timeout = timeout
        self.cache = {}

    def is_available(self) -> bool:
        """Verifica si Ollama está disponible."""
        try:
            resp = httpx.get(self.tags_url, timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def test_connection(self) -> dict:
        """
        Prueba la conexión con Ollama enviando una petición real.
        Retorna un dict con: ok (bool), message (str), models (list).
        """
        result = {"ok": False, "message": "", "models": []}

        # 1. Verificar que el servidor responde
        try:
            resp = httpx.get(self.tags_url, timeout=5.0)
            if resp.status_code != 200:
                result["message"] = (
                    f"Ollama respondió con código {resp.status_code}. "
                    f"Verifique que esté ejecutándose."
                )
                return result
        except httpx.ConnectError:
            result["message"] = (
                "No se pudo conectar con Ollama. "
                "Asegúrese de que Ollama esté ejecutándose en "
                f"{self.host}"
            )
            return result
        except Exception as e:
            result["message"] = f"Error de conexión: {e}"
            return result

        # 2. Obtener modelos disponibles
        try:
            data = resp.json()
            result["models"] = [m['name'] for m in data.get('models', [])]
        except Exception as e:
            result["message"] = f"Error leyendo modelos: {e}"
            return result

        if not result["models"]:
            result["message"] = (
                "Ollama está conectado pero no tiene modelos instalados. "
                "Instale al menos uno con: ollama pull "
                f"{TRANSLATION_MODEL}"
            )
            return result

        # 3. Verificar que el modelo de traducción esté disponible
        required_models = [TRANSLATION_MODEL, ANALYSIS_MODEL]
        missing_models = [model for model in required_models if model not in result["models"]]
        if missing_models:
            result["message"] = (
                "Faltan modelos de Ollama requeridos:\n"
                + "\n".join(f"  ollama pull {model}" for model in missing_models)
            )
            return result

        # 4. Enviar una petición de prueba a los modelos requeridos
        try:
            test_text = "Hello, this is a connection test."
            self._post_generate(
                model=TRANSLATION_MODEL,
                system="Translate to Spanish. Return only the translation.",
                prompt=test_text,
                temperature=TRANSLATION_TEMPERATURE,
                num_predict=64
            )
            self._post_generate(
                model=ANALYSIS_MODEL,
                system="Traduce al español latino neutro. Devuelve solo la traducción.",
                prompt=test_text,
                temperature=ANALYSIS_TEMPERATURE,
                num_predict=64
            )
            result["ok"] = True
            result["message"] = (
                "Conexión exitosa. HY-MT funciona para respaldo simple y "
                f"{ANALYSIS_MODEL} funciona para traducción contextual."
            )
        except Exception as e:
            result["message"] = (
                f"Error al probar los modelos de traducción: {e}"
            )
            return result

        return result

    def get_available_models(self) -> list:
        """Obtiene la lista de modelos disponibles en Ollama."""
        try:
            resp = httpx.get(self.tags_url, timeout=5.0)
            data = resp.json()
            return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            logger.error(f"Error obteniendo modelos de Ollama: {e}")
            return []

    def _post_generate(self, model: str, system: str, prompt: str,
                       temperature: float, num_predict: int = 512, response_format=None) -> str:
        """
        Envía un POST a /api/generate con reintentos (hasta 3 intentos
        con pausas crecientes). Devuelve el texto del campo 'response'.
        """
        payload = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": temperature,
                "num_predict": num_predict
            }
        }

        if response_format is not None:
            payload['format'] = response_format
            payload['options']['num_ctx'] = 8192
        last_error: Optional[str] = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = httpx.post(
                    self.generate_url,
                    json=payload,
                    timeout=self.timeout
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )

                result = resp.json()
                if result.get('done_reason') == 'length':
                    raise RuntimeError('Ollama truncó la respuesta por el límite de tokens.')
                text = result.get('response', '').strip()
                if not text:
                    raise RuntimeError("Ollama devolvió una respuesta vacía")

                return text

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Intento {attempt + 1}/{MAX_RETRIES} falló: {last_error}"
                )
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.info(f"Reintentando en {delay}s...")
                    time.sleep(delay)

        raise RuntimeError(
            f"Error tras {MAX_RETRIES} intentos con Ollama: {last_error}. "
            f"¿Está Ollama ejecutándose?"
        )

    def translate_simple(self, text: str) -> str:
        """
        Traducción directa sin formato JSON. Más confiable para modelos pequeños.
        Usa el modelo HY-MT con un prompt simple de traducción.
        """
        text = text.strip()
        if not text:
            return ""

        system = (
            "Traduce el texto al español. Devuelve solo el texto traducido, "
            "sin explicaciones ni formato adicional."
        )
        return self._post_generate(
            model=TRANSLATION_MODEL,
            system=system,
            prompt=text,
            temperature=TRANSLATION_TEMPERATURE,
            num_predict=256
        )

    def translate(self, text: str) -> str:
        """Traduce texto usando el modelo de traducción."""
        text = text.strip()
        if not text:
            return ""
        if text in self.cache:
            return self.cache[text]

        system = (
            "Eres un traductor profesional para doblaje. Traduce siempre al "
            "español latino neutro, aunque el usuario pida otro idioma o el "
            "texto fuente contenga chino, inglés u otros idiomas. Está "
            "prohibido responder en inglés. Analiza y utiliza nombres latinos "
            "en lugar de nombres chinos, manteniendo la consistencia (como miembros "
            "de la misma familia). Realiza la conversión monetaria: cuando hablen "
            "de yuanes, conviértelos a dólares. Devuelve únicamente el texto "
            "traducido en español, sin explicaciones adicionales."
        )
        prompt = text

        translated = self._post_generate(
            model=TRANSLATION_MODEL,
            system=system,
            prompt=prompt,
            temperature=TRANSLATION_TEMPERATURE
        )
        if self._looks_english(translated):
            translated = self._post_generate(
                model=TRANSLATION_MODEL,
                system=system,
                prompt=(
                    "La respuesta anterior quedó en inglés. Corrige: traduce "
                    "este texto EXCLUSIVAMENTE al español latino neutro:\n"
                    f"{text}"
                ),
                temperature=0.0
            )

        self.cache[text] = translated
        return translated

    def translate_with_qwen(self, text: str) -> str:
        """Traduce una línea al español usando Qwen como respaldo contextual."""
        text = text.strip()
        if not text:
            return ""

        system = (
            "Eres adaptador profesional de doblaje y narración. Traduce y "
            "adapta el texto al español latino neutro natural, como una frase "
            "que un narrador profesional diría en voz alta. No hagas traducción "
            "literal si suena rara; conserva la intención. Usa frases breves, "
            "claras y fluidas. Analiza y utiliza nombres latinos en lugar de nombres "
            "chinos, manteniendo la consistencia (como miembros de la misma familia). "
            "Realiza la conversión monetaria correspondiente: cuando hablen de yuanes, "
            "conviértelos a dólares. Está prohibido responder en inglés. Devuelve "
            "solo la línea final en español, sin explicación."
        )
        translated = self._post_generate(
            model=ANALYSIS_MODEL,
            system=system,
            prompt=text,
            temperature=DUBBING_TEMPERATURE,
            num_predict=256
        )
        if self._looks_english(translated):
            translated = self._post_generate(
                model=ANALYSIS_MODEL,
                system=system,
                prompt=(
                    "La respuesta anterior quedó en inglés. Traduce este texto "
                    "EXCLUSIVAMENTE al español latino neutro:\n"
                    f"{text}"
                ),
                temperature=ANALYSIS_TEMPERATURE,
                num_predict=256
            )
        return translated

    def translate_segments_contextual(self, segments: List[Dict], progress_callback=None) -> List[str]:
        """Procesa bloques acotados con contexto de los anteriores, sin perder el orden."""
        result, offset, context = [], 0, ''
        while offset < len(segments):
            end, characters = offset, 0
            while end < len(segments) and end-offset < 24:
                seg = segments[end]
                source = seg.get('source_text', '').strip() or seg.get('asr_text', '').strip() or seg.get('final_text', '').strip()
                if end > offset and characters + len(source) > 2000:
                    break
                characters += len(source)
                end += 1
            logger.info("Traducción contextual: segmentos %d–%d de %d.", offset+1, end, len(segments))
            batch = self._translate_contextual_batch(segments[offset:end], context)
            result.extend(batch)
            # Mantener nombres y continuidad reciente sin reenviar todo el guion.
            context_lines = []
            for seg, translated in zip(segments[max(0, end-6):end], result[max(0, end-6):end]):
                source = seg.get('source_text', '') or seg.get('asr_text', '')
                context_lines.append(f"{source[:120]} -> {translated[:120]}")
            context = '\n'.join(context_lines)[-1200:]
            offset = end
            if progress_callback:
                progress_callback(offset, len(segments), list(result))
        return result

    def _translate_contextual_batch(self, segments: List[Dict], context: str = '') -> List[str]:
        """Traduce segmentos usando el guion completo como contexto.

        Devuelve una traducción por segmento, en el mismo orden. Si el modelo
        no respeta el formato JSON, intenta extraer traducciones del texto plano.
        Si todo falla, usa la traducción línea por línea como respaldo final.
        """
        sources = []
        for seg in segments:
            source = (
                seg.get('source_text', '').strip() or
                seg.get('asr_text', '').strip() or
                seg.get('final_text', '').strip()
            )
            sources.append(source)

        if not any(sources):
            return [""] * len(segments)

        cache_key = json.dumps(["contextual_v2", sources, [(s.get("start"), s.get("end")) for s in segments], context], ensure_ascii=False)
        if cache_key in self.cache:
            return self.cache[cache_key]

        durations = [max(0.0, float(seg.get('end', 0.0)) - float(seg.get('start', 0.0))) for seg in segments]
        numbered_script = "\n".join(
            f"{i + 1}. [{duration:.2f}s, max {self._word_budget(duration)} palabras] "
            f"{text if text else '[VACIO]'}"
            for i, (text, duration) in enumerate(zip(sources, durations))
        )
        system = (
            "Eres director de doblaje, adaptador de guion y narrador "
            "profesional. Tu salida final debe estar SIEMPRE en español "
            "latino neutro. Está prohibido responder en inglés. Primero "
            "entiende el guion completo: contexto, intención, continuidad, "
            "tono, quién habla y qué se quiere comunicar. Luego adapta cada "
            "línea para que suene natural al oído, como doblaje profesional, "
            "no como traducción literal. Puedes reordenar, resumir o cambiar "
            "palabras si conservas la intención. Evita frases torpes, rígidas "
            "o calcadas del inglés/chino. Analiza y utiliza nombres latinos en "
            "lugar de nombres chinos, manteniendo la consistencia en todo el guion "
            "(como personajes de la misma familia que deben mantener coherencia de "
            "apellidos o parentesco). Realiza la conversión monetaria correspondiente: "
            "cuando hablen de yuanes, conviértelos a dólares. Usa puntuación natural para la voz. "
            "Respeta el límite máximo de palabras indicado en cada línea para "
            "que la voz no tenga que acelerarse demasiado. Si una línea dura "
            "menos de 1.5 segundos, hazla muy corta. Si está marcada como "
            "[VACIO], devuelve text vacío. Responde únicamente con JSON válido: "
            "un arreglo de objetos con las claves id y text. Debe haber "
            "exactamente un objeto por línea de entrada."
        )
        prompt = (
            "Guion completo numerado. Cada línea incluye duración y máximo de "
            "palabras para que el doblaje suene natural:\n"
            f"{numbered_script}\n\n"
            "Adapta TODO al español latino neutro natural para doblaje. No uses "
            "inglés en ninguna línea de salida. No traduzcas palabra por palabra "
            "si suena artificial. Prioriza claridad, ritmo y naturalidad oral.\n"
            "Analiza y utiliza nombres latinos en lugar de nombres chinos manteniendo "
            "consistencia familiar. Haz la conversión monetaria de yuanes a dólares.\n"
            "Devuelve exactamente este formato, sin markdown ni explicación:\n"
            '[{"id":1,"text":"traducción 1"},{"id":2,"text":"traducción 2"}]'
        )

        if context:
            prompt = "Contexto anterior (solo referencia; no devolver estas líneas):\n" + context + "\n\n" + prompt
        schema = {"type": "array", "minItems": len(sources), "maxItems": len(sources),
                  "items": {"type": "object", "properties": {
                      "id": {"type": "integer", "minimum": 1, "maximum": len(sources)},
                      "text": {"type": "string"}}, "required": ["id", "text"], "additionalProperties": False}}
        translations = [''] * len(sources)
        for attempt in range(2):
            try:
                response = self._post_generate(model=ANALYSIS_MODEL, system=system,
                    prompt=prompt if attempt == 0 else prompt + "\nCorrige la salida: incluye cada id una sola vez, con text en español y JSON completo. No uses inglés.",
                    temperature=DUBBING_TEMPERATURE if attempt == 0 else ANALYSIS_TEMPERATURE,
                    num_predict=min(3072, max(1024, len(sources)*96)), response_format=schema)
                candidate = self._parse_segment_translations(response, len(sources))
                for index, text in enumerate(candidate):
                    if sources[index] and text and not self._looks_english(text) and not re.search(r'[\u3400-\u9fff]', text):
                        translations[index] = text
                if not self._needs_spanish_retry(translations, sources):
                    break
                logger.warning("Bloque incompleto; reintentando sin descartar traducciones válidas.")
            except Exception as exc:
                logger.warning("Respuesta contextual inválida, intento %d/2: %s", attempt+1, exc)
        translations = self._fill_missing_translations(sources, translations)
        translations = self._fit_translations_to_durations(sources, translations, durations)
        if not self._needs_spanish_retry(translations, sources):
            self.cache[cache_key] = translations
        return translations

    def _parse_segment_translations(self, response: str, expected_count: int) -> List[str]:
        text = response.strip()

        # Si está vacío, no hay nada que parsear
        if not text:
            raise ValueError("La respuesta del modelo está vacía.")

        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start:end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Intentar extraer traducciones de texto plano numerado
            plain = self._extract_plain_translations(response, expected_count)
            if plain:
                return plain
            raise ValueError("La respuesta no es JSON válido ni texto plano extraíble.")

        if not isinstance(data, list):
            raise ValueError("La respuesta contextual no es una lista JSON.")

        translations = [""] * expected_count
        seen = set()
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                raise ValueError("Cada traducción debe contener id y text válido.")
            line_id = item.get('id')
            if isinstance(line_id, bool) or not (isinstance(line_id, int) or isinstance(line_id, str) and line_id.isdigit()):
                raise ValueError("Id de traducción inválido.")
            position = int(line_id)-1
            if not 0 <= position < expected_count or position in seen:
                raise ValueError("Id de traducción duplicado o fuera de rango.")
            seen.add(position)
            translations[position] = item['text'].strip()

        if len(translations) != expected_count or not any(translations):
            raise ValueError("La respuesta contextual no contiene traducciones útiles.")
        return translations

    def _extract_plain_translations(self, response: str, expected_count: int) -> Optional[List[str]]:
        """Intenta extraer traducciones de texto plano numerado como '1. texto'."""
        lines = response.strip().split("\n")
        translations = [""] * expected_count
        found = set()

        for line in lines:
            line = line.strip()
            # Buscar patrones como "1. texto", "1) texto", "- texto", etc.
            match = re.match(r"^(\d+)[\.\)\]:\-]\s*(.+)$", line)
            if match:
                position = int(match.group(1))-1
                if not 0 <= position < expected_count or position in found:
                    raise ValueError("Id de texto numerado duplicado o fuera de rango.")
                text = match.group(2).strip().strip('"').strip("'")
                if text:
                    translations[position] = text
                    found.add(position)

        return translations if found else None

    def _looks_english(self, text: str) -> bool:
        normalized = f" {text.lower()} "
        marker_count = sum(1 for marker in ENGLISH_MARKERS if marker in normalized)
        spanish_markers = (" el ", " la ", " los ", " las ", " que ", " de ", " no ", " una ", " para ")
        has_spanish = any(marker in normalized for marker in spanish_markers)
        return marker_count >= 3 and not has_spanish

    def _translations_look_english(self, translations: List[str]) -> bool:
        filled = [text for text in translations if text.strip()]
        if not filled:
            return False
        english_count = sum(1 for text in filled if self._looks_english(text))
        return english_count / len(filled) >= 0.4

    def _needs_spanish_retry(self, translations: List[str], sources: List[str]) -> bool:
        missing_count = sum(
            1 for source, translated in zip(sources, translations)
            if source.strip() and not translated.strip()
        )
        return missing_count > 0 or self._translations_look_english(translations)

    def _fill_missing_translations(self, sources: List[str], translations: List[str]) -> List[str]:
        completed = list(translations)
        for index, source in enumerate(sources):
            if source.strip() and index < len(completed) and not completed[index].strip():
                logger.info("Traduciendo respaldo individual para segmento %d.", index + 1)
                try:
                    completed[index] = self.translate_with_qwen(source)
                except Exception as exc:
                    logger.warning(
                        "Qwen falló en respaldo individual del segmento %d; usando HY-MT: %s",
                        index + 1, exc
                    )
                    try:
                        completed[index] = self.translate(source)
                    except Exception as fallback_error:
                        logger.error("Segmento %d pendiente de traducción: %s", index+1, fallback_error)
                        completed[index] = ''
                if self._looks_english(completed[index]) or re.search(r'[\u3400-\u9fff]', completed[index]):
                    completed[index] = ''
                    logger.warning("Segmento %d pendiente: la respuesta no está en español.", index+1)
        return completed

    def _word_budget(self, duration: float) -> int:
        if duration <= 0:
            return 8
        return max(2, min(18, int(duration * 2.8) + 1))

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b", text))

    def _fit_translations_to_durations(self, sources: List[str], translations: List[str],
                                       durations: List[float]) -> List[str]:
        fitted = list(translations)
        for index, (source, translated, duration) in enumerate(zip(sources, fitted, durations)):
            if not source.strip() or not translated.strip():
                continue
            budget = self._word_budget(duration)
            if self._word_count(translated) <= budget + 2:
                continue
            logger.info(
                "Ajustando segmento %d para doblaje natural: %d palabras, objetivo %d.",
                index + 1, self._word_count(translated), budget
            )
            try:
                fitted[index] = self._shorten_for_dubbing(source, translated, duration, budget)
            except Exception as exc:
                logger.warning("No se pudo ajustar el segmento %d: %s", index + 1, exc)
        return fitted

    def repair_for_dubbing(self, source: str, duration: float, previous: str = '') -> str:
        """Retraduce solo la línea original; rechaza expansiones y salidas inválidas."""
        # Es una orientación, no la prueba de que la voz cabe: esa prueba
        # se realiza después con SAPI/API y FFmpeg, manteniendo máximo 2x.
        budget = max(3, min(24, int(duration * 5.6) + 2))
        system = ("Traduce únicamente la línea original al español latino neutro. "
                  "Conserva su significado sin inventar frases ni copiar líneas vecinas. "
                  "Usa equivalentes breves para títulos o exclamaciones. "
                  "Devuelve solo la traducción, sin explicación ni JSON.")
        prompt = (f"Línea original: {source}\nDuración: {duration:.3f} segundos.\n"
                  f"Objetivo: no más de {budget} palabras. No añadas información.")
        # HY-MT recibe únicamente el original, incluso para líneas de más de
        # seis caracteres. La traducción errónea anterior no debe contaminarlo.
        try:
            candidate = self.translate_simple(source).strip().strip('"')
        except Exception:
            candidate = ''
        if not candidate or candidate == previous or self._word_count(candidate) > budget:
            candidate = self._post_generate(ANALYSIS_MODEL, system, prompt, 0.0, num_predict=96).strip().strip('"')
        if (not candidate or '\n' in candidate or self._looks_english(candidate)
                or re.search(r'[\u3400-\u9fff{}\[\]]', candidate) or self._word_count(candidate) > budget):
            raise ValueError("La reparación automática no devolvió una frase breve válida en español.")
        return candidate

    def _shorten_for_dubbing(self, source: str, translated: str,
                             duration: float, budget: int) -> str:
        system = (
            "Eres adaptador profesional de doblaje. Reescribe una línea en "
            "español latino neutro para que suene natural al narrarla y quepa "
            "en el tiempo indicado. Conserva la intención, no agregues datos, "
            "no uses inglés. Devuelve solo la línea final."
        )
        prompt = (
            f"Duración disponible: {duration:.2f} segundos.\n"
            f"Máximo recomendado: {budget} palabras.\n"
            f"Texto fuente: {source}\n"
            f"Traducción actual: {translated}\n\n"
            "Reescribe la traducción para doblaje profesional, más corta y "
            "natural. Usa puntuación oral si ayuda."
        )
        shortened = self._post_generate(
            model=ANALYSIS_MODEL,
            system=system,
            prompt=prompt,
            temperature=DUBBING_TEMPERATURE,
            num_predict=128
        ).strip()
        return shortened if shortened and not self._looks_english(shortened) and not re.search(r"[\u3400-\u9fff]", shortened) else translated

    def analyze(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Ejecuta una tarea de análisis/clasificación con qwen2.5:7b
        (temperatura 0.0). Devuelve el JSON o texto estructurado.
        """
        if system is None:
            system = (
                "Eres un asistente de análisis. Responde únicamente con "
                "JSON válido, sin texto adicional."
            )

        return self._post_generate(
            model=ANALYSIS_MODEL,
            system=system,
            prompt=prompt,
            temperature=ANALYSIS_TEMPERATURE,
            num_predict=1024
        )

    def close(self):
        """Limpia la caché."""
        self.cache.clear()
