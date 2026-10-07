# Video Dubbing Studio

Aplicación PySide6: extrae subtítulos con EasyOCR, transcribe audio con Whisper,
traduce a español con **tencent/HY-MT1.5-1.8B ejecutado localmente** y genera doblaje.

## Instalación y ejecución

Usa Python 3.11 o 3.12. El proyecto incluye `imageio-ffmpeg` como alternativa
a una instalación de FFmpeg en PATH:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

En este equipo, tras la instalación preparada por Codex, inicia la aplicación con:

```powershell
.\run.ps1
```

La primera traducción descarga los pesos de Hugging Face (varios GB). Las siguientes
cargas usan la caché. También puedes indicar una carpeta con los pesos completos y
el tokenizer en el campo Modelo. Activa «Usar solo modelo descargado» para impedir
descargas del traductor. EasyOCR y Whisper necesitan sus propios modelos en caché.
No se envían textos a servicios de traducción. La síntesis de voz (TTS) usa la voz local **"Vocalizer Expressive Jorge Harpo 22kHz"** (Speech2Go Voice Package VE) mediante SAPI5, sin conexión a APIs externas. Si la voz generada dura más del tiempo del subtítulo, se acelera automáticamente con FFmpeg para encajar perfectamente.

En `auto` se usa CUDA si PyTorch la detecta; en otro caso CPU. CPU carga pesos en
float32 (aproximadamente 7–8 GB solo para pesos, más memoria de ejecución) y puede
ser lenta. Para CUDA instala una distribución de PyTorch compatible con tu equipo.
No se cambia automáticamente a otro modelo si HY-MT falla.

## Ajustes

- Selecciona el idioma **del texto del video**: chino simplificado/inglés por defecto.
- Intervalo OCR: 0.25 segundos; reducirlo captura subtítulos más breves con mayor coste.
- Inicio de región OCR: 0.65 analiza el 35 % inferior. Usa 0 para toda la imagen.
- `ocr_confidence_threshold` en `config.json`: 0.4 por defecto, ajustable entre 0 y 1.
- Idioma del audio (ASR): **Automático** usa la detección de idioma de la misma
  llamada de transcripción. Puede fijarse en **Chino** o **Inglés**.
- Modelo ASR: **Large v3 Turbo** por defecto, con Faster Whisper y Silero VAD,
  usando el modelo y los parámetros locales comprobados en pyvideotrans.
  `asr_model_path` apunta a `models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo`;
  los pesos copiados ocupan aproximadamente 1.62 GB. `asr_compute_type` es `default`.
  El ASR no descarga modelos ni cambia silenciosamente a Base ante un error.
  Base, Small y Medium siguen disponibles si sus pesos ya están en caché.
- El ASR transcribe en el idioma original y alinea los tiempos por palabra.
  Separa frases y pausas de al menos 0.35 s (`asr_pause_threshold`). Los tiempos
  se conservan sobre el audio original, incluidos los silencios entre intervenciones.
  Reagrupa palabras entre ventanas del modelo, con cortes por duración y puntuación,
  sin fusionar fragmentos a través de pausas reales. La traducción recibe el mismo
  formato de segmentos que antes.
- El ajuste de voz elimina silencio inicial/final, conserva pausas internas y
  acelera la voz hablada cuando no cabe. La mezcla coloca cada voz en su muestra
  de inicio y termina en su intervalo, con una única codificación AAC en el MP4.
  Después de cambiar la traducción hay que generar las voces de nuevo.
- `translation_target`: `Spanish`; `translation_max_new_tokens`: 512.

Los controles OCR/modelo se guardan al analizar o al pulsar Guardar Configuración.
La fusión conserva la preferencia por ASR donde existe audio y usa OCR en huecos;
las columnas originales siguen disponibles para comparar y editar la traducción.
Los videos sin audio pueden analizarse por OCR; el render de doblaje actual requiere audio original.

## Comprobaciones

### Traducción manual con JSON

1. Selecciona **Manual: exportar e importar JSON** en el campo Traducción.
2. Selecciona el video y pulsa **Analizar audio (ASR)**. Este modo no necesita
   Ollama. Al terminar se abre el diálogo para guardar el guion; también puedes
   usar **Exportar Guion JSON** después.
3. El archivo contiene `source_script` en orden temporal, con `id`, `start`,
   `end`, `duration` y `source_text`, y una plantilla `translated_script`.
4. Completa únicamente el `text` de cada entrada de `translated_script`.
   Conserva `script_id` y todos los `id`. También se acepta un JSON que contenga
   solo `script_id` y `translated_script`.
5. Pulsa **Importar JSON Traducido** y selecciona tu archivo. Debe incluir todos
   los segmentos, sin identificadores repetidos ni traducciones vacías. Una
   importación inválida no cambia el guion actual.
6. Las traducciones aparecerán en **Texto Final**. Pulsa **Generar Voces** y
   después **Crear Video Final**. El programa usa tus textos importados y conserva
   los tiempos originales; no los vuelve a traducir con Ollama.

Ejemplo de JSON de entrada (para un guion de dos segmentos):

```json
{
  "translated_script": [
    {"id": 1, "text": "¿Qué vamos a comer hoy?"},
    {"id": 2, "text": "Vamos a buscar comida."}
  ]
}
```

Si vuelves a analizar el video, se obtiene un guion nuevo y debes importar de
nuevo sus traducciones. Puedes editar manualmente Texto Final antes de generar
las voces. Al importar otra traducción se invalida el audio generado anteriormente.

```powershell
python -m unittest discover -s tests -v
```

Pruebas de contratos de orden de lectura, separación temporal, selección de frases,
fusión sin audio y caché. No evalúan precisión neuronal: esa validación necesita
videos representativos, modelos descargados y las dependencias instaladas.

Referencia de integración: https://huggingface.co/tencent/HY-MT1.5-1.8B
