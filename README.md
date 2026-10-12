# Video Dubbing Studio

Aplicación PySide6: transcribe audio con Faster Whisper y Silero VAD,
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
descargas del traductor. Faster Whisper necesita sus propios modelos locales.
No se envían textos a servicios de traducción. La síntesis de voz (TTS) usa la voz local **"Vocalizer Expressive Jorge Harpo 22kHz"** (Speech2Go Voice Package VE) mediante SAPI5, sin conexión a APIs externas. Si la voz generada dura más del tiempo del subtítulo, se acelera automáticamente con FFmpeg para encajar perfectamente.

En `auto` se usa CUDA si PyTorch la detecta; en otro caso CPU. CPU carga pesos en
float32 (aproximadamente 7–8 GB solo para pesos, más memoria de ejecución) y puede
ser lenta. Para CUDA instala una distribución de PyTorch compatible con tu equipo.
No se cambia automáticamente a otro modelo si HY-MT falla.

## Ajustes

- **Motor de voz a texto**: Whisper CPP (CPU, local) o Faster Whisper (local).
  Whisper CPP usa `tools/whisper_cpp/whisper-cli.exe`, sus DLL, Silero
  `ggml-silero-v6.2.0.bin` y `models/whisper_cpp/ggml-large-v3-turbo.bin`.
  No descarga archivos durante el análisis ni cambia de motor ante un error.
  `whisper_cpp_threads` permite ajustar los hilos (4 por defecto).
  Ambos motores transcriben el idioma original; la traducción sigue siendo
  manual mediante JSON o automática con Ollama. El modelo GGML no es
  intercambiable con la carpeta CTranslate2 utilizada por Faster Whisper.
  Whisper CPP alinea los tokens al audio original utilizando el mapa temporal
  real de Silero. Conserva el texto, une subpalabras y separa pausas de 0.35 s
  e intervalos de hasta 8 s. La puntuación no prolonga artificialmente la voz.
  Si falta el mapa necesario, detiene el análisis para evitar tiempos incorrectos.
  Los tiempos por palabra son estimaciones del motor, no una garantía fonética.


- **Eliminar voz original (conservar música y efectos)**: al activar el check,
  se separan voz y acompañamiento con UVR-MDX-NET-Inst_HQ_4 ONNX local antes del render.
  El audio original con diálogo no se incluye en la mezcla: se usa únicamente
  el fondo separado a ganancia 1.0 y las voces de doblaje. El control de volumen
  original queda desactivado. Se conserva estéreo; el control de picos reduce
  únicamente el doblaje cuando no hay margen, sin atenuar ni normalizar el fondo.
  Si la separación falla, el programa cancela el render; no reutiliza la voz original.
  Sin el check se mantiene la mezcla habitual con volumen original configurable.
  El modelo debe estar en `models/onnx/UVR-MDX-NET-Inst_HQ_4.onnx`; no se
  descarga automáticamente. Se añade contexto reflejado al separar y se retira
  después para evitar picos artificiales en los extremos, conservando duración
  y nivel. `separation_model: spleeter` mantiene disponible el motor anterior
  con `vocals.fp16.onnx` y `accompaniment.fp16.onnx`, sin cambio automático.
  La separación puede dejar restos de voz o alterar efectos; conviene escuchar
  el fondo separado antes de dar por definitivo el resultado.
- Sincronización de doblaje: se elimina silencio inicial/final, se ajusta la
  duración hablada y se colocan las voces en posiciones absolutas de muestras
  a 48 kHz. Las frases cortas pueden ralentizarse hasta 0.85x; las largas se
  aceleran como máximo a 2x. Si no caben, acorta la traducción del segmento.
  Los silencios entre segmentos se mantienen y una voz fallida impide el render.

- Idioma del audio (ASR): **Automático** usa la detección de idioma de la misma
  llamada de transcripción. Puede fijarse en **Chino** o **Inglés**.
- Modelo ASR de Faster Whisper: **Large v3 Turbo** por defecto, con Silero VAD,
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

Los ajustes del modelo se guardan al analizar o al pulsar Guardar Configuración.
La tabla muestra Inicio, Fin, Texto ASR y Texto Final editable.
El análisis requiere un video con una pista de audio.

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

Pruebas de contratos de transcripción, tiempos, traducción manual, mezcla y caché. No evalúan precisión neuronal: esa validación necesita
videos representativos, modelos descargados y las dependencias instaladas.

Referencia de integración: https://huggingface.co/tencent/HY-MT1.5-1.8B

## Voces con OpenSpeaker

En Generación de voces, selecciona **OpenSpeaker API**, introduce tu clave y el
ID de voz con prefijo de proveedor, y pulsa **Generar Voces**. La clave permanece
solo en memoria durante la sesión; también puedes definir OPENSPEAKER_API_KEY
antes de iniciar la aplicación. El motor, ID y velocidad se guardan en config.json.

Obtén los IDs mediante GET https://api.ai33.pro/v3/voices?provider=edge
con la cabecera xi-api-key. Proveedores: elevenlabs, minimax, clone, edge,
kokoro, vbee y fishaudio. Usa el voice_id devuelto sin modificar.
El proveedor Python también ofrece list_voices(provider, search, page).

La integración envía multipart a /v3/text-to-speech, consulta /v1/task/{task_id},
descarga metadata.audio_url y convierte a WAV PCM16 mono 48 kHz con FFmpeg.
Cada segmento conserva el ajuste temporal existente. Velocidad API: 0.5–1.5.
Los errores 429/503 se reintentan con espera y el proceso tiene un límite de
10 minutos por segmento. La generación consume los créditos de tu cuenta.
