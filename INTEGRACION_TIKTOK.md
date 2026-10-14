# TikTok integrado en Video_Sub

## Instalación en Windows

Abre PowerShell en la carpeta de Video_Sub. Instala Python 3.12 de 64 bits y
crea un entorno propio. No copies un entorno virtual de otro equipo:

    py -3.12 -m venv .venv312
    & '.\.venv312\Scripts\python.exe' -m pip install --upgrade pip
    & '.\.venv312\Scripts\python.exe' -m pip install -r '.\requirements.txt'
    .\run.ps1

También puedes iniciar directamente:

    & '.\.venv312\Scripts\python.exe' '.\main.py'

Las nuevas dependencias son Playwright, Playwright Stealth y Rich. Requests
e Imageio FFmpeg ya formaban parte de Video_Sub. Patchright no se incorpora:
los módulos funcionales utilizan Playwright y no importan Patchright.
No se cambia la selección de ASR, Ollama, TTS ni los modelos configurados.

Se usa Google Chrome instalado por defecto. Para utilizar Chromium de
Playwright, instálalo dentro del proyecto y selecciona ese navegador en el diálogo:

    $env:PLAYWRIGHT_BROWSERS_PATH = Join-Path (Get-Location) 'data/browsers'
    & '.\.venv312\Scripts\python.exe' -m playwright install chromium

El programa también utiliza esa ruta interna para localizar Chromium.
FFmpeg usa la misma estrategia de Video_Sub: ejecutable del PATH si existe
o el binario distribuido por imageio-ffmpeg.

## Uso

La selección manual de un video local conserva su comportamiento.

En **Descargar videos de TikTok y doblar**, elige:

- URL inicial de una serie: extrae capítulos con Playwright.
- Archivo .txt: una URL por línea, orden de primera aparición, sin duplicados.
- Video individual: descarga una sola URL de TikTok.

Se aceptan archivos de texto UTF-8 con o sin BOM, comentarios # y líneas vacías.
No se admiten plataformas diferentes de TikTok.

Para iniciar sesión, pulsa **Iniciar sesión en TikTok**. Completa el acceso
en el navegador y cierra sus ventanas cuando termines. Las cookies se guardan
en el perfil interno. No se importa la sesión de otra aplicación.

Las rutas de descargas y perfil se configuran como rutas relativas a Video_Sub.
Se crean automáticamente al iniciar la operación; las rutas absolutas o que
salen del proyecto mediante .. se rechazan.

Con **Continuar automáticamente con análisis, voces y video final** activado,
un MP4 validado pasa a AnalyzeWorker, TTSWorker y RenderWorker. La salida queda
en Video_Doblado.mp4 junto al MP4 de entrada. El modo automático solo se activa
desde TikTok; abrir un video local conserva el control manual.

Se respeta el modo de traducción actual. En modo manual se pausa tras ASR hasta
importar el JSON traducido; después continúa automáticamente. Si faltan
traducciones o falla cualquier worker, no se generan las etapas siguientes.
Ollama, los modelos ASR, las voces locales y la clave OpenSpeaker conservan
sus requisitos habituales. En otro equipo, copia los modelos/recursos internos
de Video_Sub o prepáralos según su documentación y configura los servicios.

Desactiva el avance automático para revisar primero el MP4 y continuar
con el análisis mediante el botón del diálogo. La ruta del MP4 aparece al
completar la validación. La extracción, descarga, reintentos y unión se registran
en el diálogo y logs/tiktok.log.

## Código y recursos internos

    core/tiktok/
        browser.py       Perfil persistente y gestión de modales
        extractor.py     Extracción de series
        downloader.py    TikWM, respaldo Playwright, lista y reintentos
        catalog.py       Manifiestos y catálogo
        unifier.py       Orden natural, normalización, unión y transcodificación
        inputs.py        Validación de URLs y lectura de listas
        media.py         Validación real del audio/video con FFmpeg
        runtime.py       Rutas internas y exclusión entre operaciones
    core/tiktok_adapter.py   Orquestación y resultado estructurado
    gui/tiktok_worker.py     Worker y proceso interno cancelable
    gui/tiktok_dialog.py     Interfaz de descarga y sesión
    data/descargas/          Capítulos, manifiestos y resultados
    data/perfil_chrome_tiktok/
    data/temp/
    data/browsers/           Chromium opcional
    logs/
    config.json

Se incorporaron y adaptaron solamente los módulos funcionales necesarios.
Se excluyeron el menú Rich, inspectores, entrada de consola, entorno virtual,
descargas y perfil del proyecto de origen. No se ejecuta su main ni se importan
módulos desde otra aplicación. El proceso hijo usa sys.executable, ejecuta
core.tiktok_adapter y carga únicamente el código interno de Video_Sub.

Las claves nuevas de config.json son tiktok_download_dir, tiktok_profile_dir,
tiktok_temp_dir y tiktok_browser, con valores relativos predeterminados.
Las claves obsoletas de la integración externa se ignoran al cargar.

Las listas usan carpetas identificadas por nombre y hash del orden de URLs.
La reanudación valida que el video pueda decodificarse antes de omitirlo.
Las descargas nuevas conservan archivos .part hasta validar la transferencia.
Se preservan TikWM, respaldo Playwright y rondas de recuperación. Los manifiestos
y resultados mantienen número, URL, archivo y estado del capítulo.

La unión conserva el video cuando es compatible y normaliza audio a estéreo
48 kHz. Si falla, transcodifica H.264/AAC. El MP4 resultante se decodifica antes
de entregarlo a ASR; una unión pendiente nunca se considera una entrada válida.
Si solo hay un capítulo, se utiliza ese MP4 validado sin una unión artificial.

Un bloqueo del sistema operativo evita operaciones simultáneas dentro del
proyecto. Cancelar detiene el árbol del proceso hijo en Windows y libera el
bloqueo. Los capítulos válidos permanecen para reanudar. Puede quedar un .part
o .pendiente.mp4 que nunca pasa al análisis.

Si el extractor informa un total o paginación pendiente, se bloquean las series
incompletas. Cuando TikTok no informa un total, permanece la limitación de no
poder demostrar que el extractor vio todos los capítulos existentes.

## Verificación reproducible

    & '.\.venv312\Scripts\python.exe' -m unittest discover -s tests -v
    & '.\.venv312\Scripts\python.exe' '.\tests\validate_tiktok_bridge.py'
    & '.\.venv312\Scripts\python.exe' '.\tests\validate_portable_tiktok.py'
    & '.\.venv312\Scripts\python.exe' -m pip check

Las pruebas offline simulan únicamente red/extracción de navegador; ejecutan
las funciones de descarga por lista y el FFmpeg real. Comprueban URL, .txt,
individual, reanudación, corrupción, orden, errores y unión. Otras pruebas
ejercitan el respaldo Playwright, la transcodificación H.264/AAC real, audio
ausente, cancelación, rutas, bloqueo y encadenamiento automático.

La prueba portable copia el código a una ubicación nueva y bloquea mediante
auditoría el acceso a Downloads. Ejecuta allí la integración y comprueba que
los módulos y recursos se resuelvan desde la copia. No renombra ni elimina
la aplicación de origen.

La sesión de navegador se comprueba con Chrome en about:blank; no constituye
una prueba de acceso a TikTok. Falta validar una serie real con la sesión del
usuario: las pruebas no comprueban CAPTCHA, permisos de cuenta ni cambios del sitio.

## Archivos modificados

Se añadieron core/tiktok/, tests/test_tiktok_downloader.py,
tests/test_tiktok_unifier.py y tests/validate_portable_tiktok.py.
Se sustituyeron el adaptador, el worker y el diálogo de la integración externa;
se actualizaron gui/main_window.py, utils/config.py, requirements.txt,
requirements-tiktok.txt, run.ps1, .gitignore, README.md y las pruebas TikTok previas.

data/ y logs/ quedan fuera de Git para excluir videos y cookies. No se modifica
ni elimina la aplicación original. Las evidencias de validación se guardan en
validation_tiktok/ y sus rutas temporales dejan de existir al terminar.
