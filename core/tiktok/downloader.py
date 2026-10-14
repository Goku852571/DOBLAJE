from .media import valid_video
"""Adapted internal TikTok module; console CLI and external paths removed."""
from .runtime import paths
from core.media import run_cmd
import os
import re
import sys
import time
import json
import logging
from pathlib import Path
from urllib.parse import urlparse, quote
from collections import Counter
import requests

# Asegurar codificación UTF-8 en consola de Windows

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn


console = Console()
logger = logging.getLogger(__name__)
BASE_DIR = paths().root


TIKWM_API_URL = "https://www.tikwm.com/api/"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "https://www.tikwm.com/",
}

def sanitizar_nombre_archivo(texto: str, max_longitud: int = 70) -> str:
    """
    Sanitiza una cadena para que sea un nombre de archivo válido en Windows (NTFS).
    Remueve caracteres prohibidos, recorta longitud y limpia espacios superfluos.
    """
    if not texto:
        return "video"
    limpio = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(texto))
    limpio = re.sub(r'[\s_]+', ' ', limpio).strip(' .')
    if not limpio:
        limpio = "video"
    if len(limpio) > max_longitud:
        limpio = limpio[:max_longitud].rstrip(' .')
    return limpio

def _descargar_stream(download_url: str, destino: Path, headers: dict = None, cookies: dict = None) -> bool:
    """
    Descarga un flujo binario mostrando una barra de progreso Rich interactiva.
    Si la descarga se interrumpe o falla, elimina archivos parciales o corruptos.
    """
    headers = headers or DEFAULT_HEADERS.copy()
    parcial = destino.with_suffix(destino.suffix + ".part")
    try:
        with requests.get(download_url, stream=True, headers=headers, cookies=cookies, timeout=60) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=35),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            ) as progress:
                task = progress.add_task(f"[cyan]{escape(destino.name[:35])}[/cyan]", total=total_size or None)
                with open(parcial, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=128 * 1024):
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))

        # Verificar integridad del archivo descargado
        if total_size and parcial.stat().st_size != total_size and not resp.headers.get("content-encoding"):
            raise ValueError("Transferencia truncada: tamaño distinto a Content-Length")
        if parcial.exists() and parcial.stat().st_size > 100 * 1024:
            import subprocess
            from core.media import _ffmpeg_binary
            validacion = run_cmd(
                [_ffmpeg_binary(), "-v", "error", "-xerror", "-i", str(parcial),
                 "-map", "0:v:0", "-map", "0:a:0?", "-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if validacion.returncode:
                raise ValueError("Video no decodificable: " + validacion.stderr[-600:])
            parcial.replace(destino)
            return True
        else:
            console.print(f"[red]Archivo descargado incompleto o corrupto (< 100 KB): {escape(destino.name)}[/red]")
            parcial.unlink(missing_ok=True)
            return False

    except Exception as e:
        console.print(f"[red]Error al transferir video:[/red] {e}")
        logger.warning("Transferencia fallida (%s): %s", destino.name, e)
        parcial.unlink(missing_ok=True)
        return False


_transferir_una_vez = _descargar_stream


def _descargar_stream(download_url: str, destino: Path, headers: dict = None, cookies: dict = None) -> bool:
    for intento in range(3):
        if _transferir_una_vez(download_url, destino, headers, cookies):
            return True
        if intento < 2:
            print(f"VIDEO_SUB_STAGE Reintento de transferencia {intento + 1}/2: {destino.name}", flush=True)
            time.sleep(3 * (2 ** intento))
    return False

def asegurar_compatibilidad_h264(ruta_video: Path) -> Path:
    """
    Si el video está codificado en HEVC/H.265 (que Windows Media Player no reproduce sin pagar),
    lo convierte automáticamente a H.264 universal usando FFmpeg incorporado.
    """
    try:
        import subprocess
        from core.media import _ffmpeg_binary
        ffmpeg_exe = _ffmpeg_binary()
        res = run_cmd([ffmpeg_exe, "-i", str(ruta_video)], capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if "hevc" in res.stderr.lower():
            salida_h264 = ruta_video.with_stem(f"{ruta_video.stem}_h264")
            if not salida_h264.exists() or not valid_video(salida_h264):
                console.print(f"[cyan]Convirtiendo códec HEVC a H.264 universal para reproducción directa en Windows...[/cyan]")
                run_cmd([ffmpeg_exe, "-y", "-i", str(ruta_video), "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-c:a", "copy", str(salida_h264)], check=True, capture_output=True)
                console.print(f"[green]✓ Versión universal generada:[/green] {escape(salida_h264.name)}")
            return salida_h264
    except Exception as e:
        console.print(f"[dim]Nota compatibilidad H.264: {e}[/dim]")
    return ruta_video

def _obtener_info_tikwm(url: str) -> dict:
    """
    Consulta la API pública de TikWM para obtener metadatos y enlace sin marca de agua.
    Retorna diccionario con play_url, autor, titulo, id o None si falla.
    """
    try:
        # Intento 1: POST
        resp = requests.post(
            TIKWM_API_URL,
            data={"url": url, "hd": 1},
            headers=DEFAULT_HEADERS,
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") != 0:
                logger.warning("TikWM POST: código=%s mensaje=%s", data.get("code"), data.get("msg"))
            if data.get("code") == 0 and "data" in data:
                d = data["data"]
                play_url = d.get("hdplay") or d.get("play")
                if play_url and play_url.startswith("/"):
                    play_url = f"https://www.tikwm.com{play_url}"
                author_obj = d.get("author") or {}
                author = author_obj.get("unique_id") or author_obj.get("nickname") or "tiktok_user"
                title = d.get("title") or ""
                vid_id = str(d.get("id") or "")
                return {
                    "success": True,
                    "play_url": play_url,
                    "autor": author,
                    "titulo": title,
                    "id": vid_id,
                    "motor": "TikWM API"
                }

        logger.info("TikWM POST HTTP %s", resp.status_code)
        time.sleep(3)
        # Intento 2: GET fallback
        resp_get = requests.get(
            TIKWM_API_URL,
            params={"url": url, "hd": 1},
            headers=DEFAULT_HEADERS,
            timeout=15
        )
        if resp_get.status_code == 200:
            data_get = resp_get.json()
            if data_get.get("code") != 0:
                logger.warning("TikWM GET: código=%s mensaje=%s", data_get.get("code"), data_get.get("msg"))
            if data_get.get("code") == 0 and "data" in data_get:
                d = data_get["data"]
                play_url = d.get("hdplay") or d.get("play")
                if play_url and play_url.startswith("/"):
                    play_url = f"https://www.tikwm.com{play_url}"
                author_obj = d.get("author") or {}
                author = author_obj.get("unique_id") or author_obj.get("nickname") or "tiktok_user"
                title = d.get("title") or ""
                vid_id = str(d.get("id") or "")
                return {
                    "success": True,
                    "play_url": play_url,
                    "autor": author,
                    "titulo": title,
                    "id": vid_id,
                    "motor": "TikWM API"
                }
    except Exception as e:
        logger.warning("Consulta TikWM fallida: %s", e)
        console.print(f"[dim]Nota TikWM: {e}[/dim]")

    return {"success": False}

def _navegar_video(page, url: str):
    """Reintenta fallos transitorios sin continuar sobre una página de error."""
    estado = {"http": None}

    def registrar_documento(response):
        if response.request.is_navigation_request() and response.frame == page.main_frame:
            estado["http"] = response.status
            logger.info("Navegación HTTP %s URL=%s", response.status, response.url)

    page.on("response", registrar_documento)
    try:
        for intento in range(3):
            estado["http"] = None
            try:
                respuesta = page.goto(url, wait_until="domcontentloaded", timeout=45000)
                if respuesta is not None:
                    estado["http"] = respuesta.status
                if estado["http"] is not None and estado["http"] >= 400:
                    raise RuntimeError(f"TikTok respondió HTTP {estado['http']}")
                return respuesta
            except Exception as exc:
                codigo = estado["http"]
                logger.warning("Navegación intento=%s HTTP=%s URL=%s error=%s", intento + 1, codigo, url, exc)
                if codigo in (401, 403, 404, 410) or intento == 2:
                    raise RuntimeError(f"No se pudo abrir el capítulo (HTTP {codigo if codigo is not None else 'no recibido'}): {exc}") from exc
                espera = (30 if codigo == 429 else 5) * (2 ** intento)
                console.print(f"[yellow]Fallo de navegación. Reintento {intento + 2}/3 en {espera} segundos.[/yellow]")
                page.wait_for_timeout(espera * 1000)
    finally:
        page.remove_listener("response", registrar_documento)


def _descargar_con_playwright(url: str, output_dir: Path, headless: bool = False, nombre_archivo_custom: str = None) -> dict:
    """
    Estrategia de respaldo mediante Playwright y perfil persistente de Chrome.
    Inspecciona hidratación (__UNIVERSAL_DATA_FOR_REHYDRATION__) o intercepta respuestas multimedia.
    """
    from .browser import obtener_contexto_persistente, manejar_validaciones_y_modales
    console.print("[yellow]Activando motor de respaldo Playwright con perfil persistente...[/yellow]")
    media_captured = []

    def on_response(response):
        ct = response.headers.get("content-type", "").lower()
        r_url = response.url
        if "video/mp4" in ct or ".mp4" in r_url or ("mime_type=video_mp4" in r_url and "tiktokcdn" in r_url):
            if r_url not in media_captured:
                media_captured.append(r_url)

    p, context, page = obtener_contexto_persistente(headless=headless)
    page.on("response", on_response)

    author = "tiktok_user"
    title = ""
    play_addr = None
    vid_id = ""

    try:
        console.print(f"[cyan]Navegando al video con Chrome:[/cyan] {url}")
        _navegar_video(page, url)
        time.sleep(2)
        manejar_validaciones_y_modales(page)

        # 1. Inspeccionar JSON de hidratación
        rehydration_script = """() => {
            const el = document.getElementById('__UNIVERSAL_DATA_FOR_REHYDRATION__');
            if (el) {
                try { return JSON.parse(el.textContent); } catch(e) { return null; }
            }
            return null;
        }"""
        rehydration = page.evaluate(rehydration_script)
        if rehydration:
            try:
                default_scope = rehydration.get("__DEFAULT_SCOPE__", {})
                video_detail = default_scope.get("webapp.video-detail", {})
                item_struct = video_detail.get("itemInfo", {}).get("itemStruct", {})
                if item_struct:
                    vid_id = str(item_struct.get("id") or "")
                    author_obj = item_struct.get("author") or {}
                    author = author_obj.get("uniqueId") or author_obj.get("nickname") or author
                    title = item_struct.get("desc") or ""
                    video_info = item_struct.get("video") or {}
                    play_addr = video_info.get("playAddr") or video_info.get("downloadAddr")
            except Exception:
                pass

        # 2. Si no se encontró playAddr en hidratación, usar respuestas multimedia interceptadas o DOM
        if not play_addr and media_captured:
            play_addr = media_captured[0]

        if not play_addr:
            dom_src = page.evaluate("""() => {
                const v = document.querySelector('video');
                if (v) return v.src || (v.querySelector('source') ? v.querySelector('source').src : '');
                return '';
            }""")
            if dom_src:
                play_addr = dom_src

        if not title:
            raw_title = page.title()
            title = re.sub(r'\s*\|\s*TikTok.*$', '', raw_title).strip()

        if not vid_id:
            id_match = re.search(r'/video/(\d+)', url)
            if id_match:
                vid_id = id_match.group(1)

        # Preparar nombre de archivo
        autor_limpio = sanitizar_nombre_archivo(author, max_longitud=25)
        titulo_limpio = sanitizar_nombre_archivo(title, max_longitud=45) if title else f"video_{vid_id or 'tiktok'}"
        nombre_archivo = nombre_archivo_custom if nombre_archivo_custom else f"[{autor_limpio}] {titulo_limpio}.mp4"
        destino = output_dir / nombre_archivo
        salida_h264 = destino.with_stem(f"{destino.stem}_h264")

        # Comprobar si ya existe con tamaño válido (> 100 KB) - Idempotencia / Reanudación
        archivo_existente = None
        if salida_h264.exists() and salida_h264.stat().st_size > 100 * 1024 and valid_video(salida_h264):
            archivo_existente = salida_h264
        elif destino.exists() and destino.stat().st_size > 100 * 1024 and valid_video(destino):
            archivo_existente = destino

        if archivo_existente:
            tam_mb = round(archivo_existente.stat().st_size / (1024 * 1024), 2)
            console.print(f"[yellow]✓ Capítulo ya descargado previamente (Omitiendo): {escape(archivo_existente.name)}[/yellow]")
            return {
                "success": True,
                "url": url,
                "titulo": title or titulo_limpio,
                "autor": author,
                "archivo": str(archivo_existente),
                "tamano_mb": tam_mb,
                "motor": "Playwright Fallback (Omitido)"
            }

        if not play_addr:
            return {
                "success": False,
                "error": "No se detectó dirección de reproducción MP4 en la página",
                "motor": "Playwright Fallback"
            }

        # Obtener cookies y headers de la sesión del navegador
        cookies_dict = {c["name"]: c["value"] for c in context.cookies()}
        user_agent = page.evaluate("() => navigator.userAgent")
        download_headers = {
            "User-Agent": user_agent,
            "Referer": "https://www.tiktok.com/",
            "Accept": "*/*",
        }

        console.print(f"[cyan]Descargando flujo directo desde CDN de TikTok...[/cyan]")
        ok = _descargar_stream(play_addr, destino, headers=download_headers, cookies=cookies_dict)
        if ok:
            tam_mb = round(destino.stat().st_size / (1024 * 1024), 2)
            archivo_final = asegurar_compatibilidad_h264(destino)
            return {
                "success": True,
                "url": url,
                "titulo": title or titulo_limpio,
                "autor": author,
                "archivo": str(archivo_final),
                "tamano_mb": round(archivo_final.stat().st_size / (1024 * 1024), 2),
                "motor": "Playwright Fallback"
            }
        else:
            return {
                "success": False,
                "error": "Fallo al escribir flujo de video en disco",
                "motor": "Playwright Fallback"
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "motor": "Playwright Fallback"
        }
    finally:
        context.close()
        p.stop()

def descargar_video(url: str, output_dir: Path = None, nombre_archivo_custom: str = None) -> dict:
    """
    Descarga un video individual de TikTok sin marca de agua usando estrategia dual:
    1. Primario: TikWM API (rápido, sin watermark, metadatos enriquecidos).
    2. Respaldo: Playwright con perfil persistente de Chrome.
    Implementa verificación de idempotencia: omite archivos ya descargados con tamaño > 100 KB.
    """
    from .inputs import validate_url
    validate_url(url)
    output_dir = Path(output_dir) if output_dir else paths().downloads
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reanudar antes de consultar servicios externos.
    if nombre_archivo_custom:
        destino_previo = output_dir / nombre_archivo_custom
        for previo in (destino_previo.with_stem(destino_previo.stem + "_h264"), destino_previo):
            if previo.is_file() and previo.stat().st_size > 100 * 1024 and valid_video(previo):
                return {"success": True, "url": url, "archivo": str(previo),
                        "tamano_mb": round(previo.stat().st_size / 1048576, 2), "motor": "Local (Omitido)"}

    console.print(f"\n[bold cyan]Iniciando descarga de:[/bold cyan] [yellow]{url}[/yellow]")

    # 1. Estrategia Primaria: TikWM API
    info = _obtener_info_tikwm(url)
    if info.get("success") and info.get("play_url"):
        autor = info.get("autor", "tiktok_user")
        titulo = info.get("titulo", "")
        vid_id = info.get("id", "")
        autor_limpio = sanitizar_nombre_archivo(autor, max_longitud=25)
        titulo_limpio = sanitizar_nombre_archivo(titulo, max_longitud=45) if titulo else f"video_{vid_id or 'tiktok'}"
        nombre_archivo = nombre_archivo_custom if nombre_archivo_custom else f"[{autor_limpio}] {titulo_limpio}.mp4"
        destino = output_dir / nombre_archivo
        salida_h264 = destino.with_stem(f"{destino.stem}_h264")

        # Comprobar si ya existe con tamaño válido (> 100 KB) - Idempotencia / Reanudación
        archivo_existente = None
        if salida_h264.exists() and salida_h264.stat().st_size > 100 * 1024 and valid_video(salida_h264):
            archivo_existente = salida_h264
        elif destino.exists() and destino.stat().st_size > 100 * 1024 and valid_video(destino):
            archivo_existente = destino

        if archivo_existente:
            tam_mb = round(archivo_existente.stat().st_size / (1024 * 1024), 2)
            console.print(f"[yellow]✓ Capítulo ya descargado previamente (Omitiendo): {escape(archivo_existente.name)}[/yellow]")
            return {
                "success": True,
                "url": url,
                "titulo": titulo or titulo_limpio,
                "autor": autor,
                "archivo": str(archivo_existente),
                "tamano_mb": tam_mb,
                "motor": "TikWM API (Omitido)"
            }

        # Si había un archivo incompleto residual, eliminarlo antes de escribir
        if destino.exists() and destino.stat().st_size <= 100 * 1024:
            destino.unlink(missing_ok=True)

        console.print(f"[green]Video resuelto vía TikWM API (Sin marca de agua):[/green] {escape(nombre_archivo)}")
        ok = _descargar_stream(info["play_url"], destino)
        if ok:
            tam_mb = round(destino.stat().st_size / (1024 * 1024), 2)
            console.print(f"[bold green]✓ Descarga completada:[/bold green] [white]{escape(destino.name)}[/white] ({tam_mb} MB)")
            archivo_final = asegurar_compatibilidad_h264(destino)
            return {
                "success": True,
                "url": url,
                "titulo": titulo or titulo_limpio,
                "autor": autor,
                "archivo": str(archivo_final),
                "tamano_mb": round(archivo_final.stat().st_size / (1024 * 1024), 2),
                "motor": "TikWM API"
            }
        else:
            console.print("[yellow]Fallo en la descarga por TikWM. Intentando con motor de respaldo...[/yellow]")

    # 2. Estrategia de Respaldo: Playwright
    res_fallback = _descargar_con_playwright(url, output_dir=output_dir, nombre_archivo_custom=nombre_archivo_custom)
    if res_fallback.get("success"):
        console.print(f"[bold green]✓ Descarga completada con motor Playwright:[/bold green] {escape(str(res_fallback.get('archivo')))}")
        return res_fallback
    else:
        err = res_fallback.get("error", "No se pudo descargar el video con ninguna estrategia")
        console.print(f"[bold red]✗ Error en descarga:[/bold red] {err}")
        return {
            "success": False,
            "url": url,
            "error": err,
            "motor": None
        }

def _reintentar_fallidos(resultados: list, urls: list, carpeta_capitulos: Path, guardar):
    """Rondas limitadas para rechazos intermitentes observados en el historial."""
    for ronda, espera in enumerate((60, 120), 1):
        pendientes = [i for i, res in enumerate(resultados)
                      if not res.get("success") and (
                          "HTTP 403" in (res.get("error") or "") or
                          "HTTP 429" in (res.get("error") or "") or
                          "ERR_HTTP_RESPONSE_CODE_FAILURE" in (res.get("error") or ""))
                      and not any(f"HTTP {codigo}" in (res.get("error") or "")
                                  for codigo in (401, 404, 410))]
        if not pendientes:
            break
        console.print(f"[yellow]Ronda de recuperación {ronda}/2: {len(pendientes)} capítulos. Esperando {espera} segundos.[/yellow]")
        # Esperas cortas permiten interrumpir con Ctrl+C.
        for _ in range(espera):
            time.sleep(1)
        for i in pendientes:
            try:
                res = descargar_video(urls[i], output_dir=carpeta_capitulos,
                                      nombre_archivo_custom=f"Capitulo_{i + 1:02d}.mp4")
            except Exception as exc:
                res = {"success": False, "url": urls[i], "error": str(exc), "motor": None}
            resultados[i] = res
            guardar(i + 1, urls[i], res, ronda)


def descargar_desde_archivo(archivo_txt: str = "enlaces_serie.txt", output_dir: Path = None, auto_unir: bool = False, serie_dir: Path = None) -> list:
    """
    Lee URLs de un archivo de texto línea por línea y descarga cada video en lote.
    - Idempotente y tolerante a fallos: omite videos ya existentes (> 100 KB).
    - Guarda los capítulos organizados en 'descargas/[{autor}] {titulo}/capitulos/Capitulo_01.mp4'...
    - Crea y actualiza los metadatos info_serie.json e info_serie.md.
    - Si auto_unir=True, une los capítulos en 'Pelicula_Completa_{titulo}.mp4'.
    """
    ruta_archivo = Path(archivo_txt)
    if not ruta_archivo.is_absolute():
        ruta_archivo = BASE_DIR / archivo_txt

    if not ruta_archivo.exists():
        console.print(f"[bold red]Error:[/bold red] No se encontró el archivo de enlaces en: [yellow]{ruta_archivo}[/yellow]")
        return []

    with open(ruta_archivo, "r", encoding="utf-8-sig") as f:
        lineas = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Deduplicar manteniendo orden
    urls = []
    for l in lineas:
        if l not in urls:
            urls.append(l)

    from .inputs import validate_url
    for url in urls:
        validate_url(url)
    if not urls:
        console.print(f"[yellow]El archivo {ruta_archivo.name} no contiene URLs válidas.[/yellow]")
        return []

    from .catalog import sanitizar_nombre_archivo, crear_o_actualizar_manifiesto, extraer_metadatos_de_carpeta

    # Determinar la carpeta dedicada de la serie
    if serie_dir:
        carpeta_serie = Path(serie_dir)
    elif ruta_archivo.parent != BASE_DIR and ruta_archivo.parent != paths().downloads:
        carpeta_serie = ruta_archivo.parent
    elif output_dir and Path(output_dir) != paths().downloads:
        p_out = Path(output_dir)
        carpeta_serie = p_out.parent if p_out.name.lower() == "capitulos" else p_out
    else:
        autor = "tiktok_user"
        titulo = "Serie"

        # 1. Intentar desde archivo .md vecino
        md_vecino = ruta_archivo.with_suffix(".md")
        if md_vecino.exists():
            try:
                with open(md_vecino, "r", encoding="utf-8") as f:
                    for line in f:
                        m = re.search(r'Serie:\s*(.*)', line)
                        if m and m.group(1).strip() and m.group(1).strip() != "Serie_TikTok":
                            titulo = m.group(1).strip()
                            break
            except Exception:
                pass

        # 2. Extraer autor de la primera URL
        match_aut = re.search(r"@([^/?#]+)", urls[0])
        if match_aut:
            autor = match_aut.group(1)

        # 3. Si título sigue siendo genérico, consultar API
        if titulo == "Serie":
            info_1 = _obtener_info_tikwm(urls[0])
            if info_1.get("titulo"):
                titulo = info_1["titulo"]
            if info_1.get("autor"):
                autor = info_1["autor"]

        autor_limpio = sanitizar_nombre_archivo(autor, max_longitud=30, separador="_")
        titulo_limpio = sanitizar_nombre_archivo(titulo, max_longitud=50, separador=" ")
        carpeta_serie = paths().downloads / f"[{autor_limpio}] {titulo_limpio}"

    carpeta_serie.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(carpeta_serie / "descarga.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    carpeta_capitulos = carpeta_serie / "capitulos"
    carpeta_capitulos.mkdir(parents=True, exist_ok=True)

    # Copiar o asegurar enlaces_serie.txt en carpeta_serie
    txt_interno = carpeta_serie / "enlaces_serie.txt"
    if not txt_interno.exists() or txt_interno.resolve() != ruta_archivo.resolve():
        try:
            with open(txt_interno, "w", encoding="utf-8") as f:
                for u in urls:
                    f.write(f"{u}\n")
        except Exception:
            pass

    # Inicializar manifiesto
    autor_final, titulo_final = extraer_metadatos_de_carpeta(carpeta_serie)
    crear_o_actualizar_manifiesto(carpeta_serie, titulo=titulo_final, autor=autor_final, capitulos=urls)

    console.print(Panel.fit(
        f"[bold cyan]Descarga Masiva y Reanudable de Serie[/bold cyan]\n"
        f"Archivo de origen: [yellow]{escape(str(ruta_archivo))}[/yellow]\n"
        f"Carpeta de la serie: [green]{escape(str(carpeta_serie))}[/green]\n"
        f"Total de capítulos en lista: [green]{len(urls)}[/green]\n"
        f"Subcarpeta de capítulos: [cyan]{escape(str(carpeta_capitulos))}[/cyan]\n"
        f"Unión automática al finalizar: {'[bold green]Activada[/]' if auto_unir else '[bold dim]Desactivada[/]'}",
        title="Procesando Lote RPA Tolerante a Fallos"
    ))

    resultados = []

    def guardar_resultado(numero, url, resultado, ronda=0):
        logger.info("Capítulo %s URL=%s resultado=%s ronda=%s", numero, url,
                    json.dumps(resultado, ensure_ascii=False), ronda)
        (carpeta_serie / "resultados_descarga.json").write_text(
            json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

    for i, url in enumerate(urls, 1):
        print(f"VIDEO_SUB_STAGE Descarga: capítulo {i}/{len(urls)}", flush=True)
        nombre_cap = f"Capitulo_{i:02d}.mp4"
        console.print(f"\n[bold magenta]═══════ (Capítulo {i} de {len(urls)}: {nombre_cap}) ═══════[/bold magenta]")
        try:
            res = descargar_video(url, output_dir=carpeta_capitulos, nombre_archivo_custom=nombre_cap)
            resultados.append(res)
        except Exception as e:
            console.print(f"[bold red]✗ Error inesperado en capítulo {i}:[/bold red] {e}")
            resultados.append({
                "success": False,
                "url": url,
                "error": str(e),
                "motor": None
            })
        guardar_resultado(i, url, resultados[-1])
        if i < len(urls):
            time.sleep(1.0) # Pausa preventiva para balancear peticiones

    _reintentar_fallidos(resultados, urls, carpeta_capitulos, guardar_resultado)

    # Actualizar manifiesto tras finalizar todas las descargas
    crear_o_actualizar_manifiesto(carpeta_serie, titulo=titulo_final, autor=autor_final, capitulos=urls)

    # Mostrar tabla resumen final
    tabla = Table(title="[bold yellow]Resumen de Descarga y Reanudación de Serie[/bold yellow]", show_header=True)
    tabla.add_column("#", justify="center", style="dim", width=4)
    tabla.add_column("Autor", style="cyan")
    tabla.add_column("Título", style="white")
    tabla.add_column("Tamaño", justify="right", style="magenta")
    tabla.add_column("Motor", style="yellow")
    tabla.add_column("Estado", justify="center")

    exitosos = 0
    omitidos = 0
    fallidos = 0

    for idx, r in enumerate(resultados, 1):
        if r.get("success"):
            exitosos += 1
            motor_str = r.get("motor", "TikWM")
            if "Omitido" in motor_str or "Existente" in motor_str:
                omitidos += 1
                estado_str = "[bold yellow]Omitido (Existente)[/bold yellow]"
            else:
                estado_str = "[bold green]Descargado[/bold green]"
            titulo_recortado = (r.get("titulo") or f"Capitulo_{idx:02d}")[:35]
            tam = f"{r.get('tamano_mb', 0)} MB"
            tabla.add_row(
                str(idx),
                escape(r.get("autor", autor_final)),
                escape(titulo_recortado),
                tam,
                motor_str,
                estado_str
            )
        else:
            fallidos += 1
            estado_str = "[bold red]Falló[/bold red]"
            tabla.add_row(
                str(idx),
                "N/A",
                escape((r.get("error") or "Error")[:35]),
                "-",
                "-",
                estado_str
            )

    console.print("\n")
    console.print(tabla)
    console.print(Panel(
        f"Capítulos totales en lista: [bold]{len(urls)}[/bold]\n"
        f"Completados con éxito: [bold green]{exitosos}[/bold green] (Omitidos previamente: [bold yellow]{omitidos}[/bold yellow] • Nuevos: [bold green]{exitosos - omitidos}[/bold green])\n"
        f"Fallidos: [bold red]{fallidos}[/bold red]\n"
        f"Carpeta de la serie: [bold cyan]{escape(str(carpeta_serie))}[/bold cyan]",
        title="Fin de Procesamiento Lote"
    ))

    # Integración con TASK-8 y TASK-9: Unir videos si auto_unir está activado
    if auto_unir and not fallidos:
        console.print("\n[bold cyan]Iniciando unificación automática de capítulos descargados...[/bold cyan]")
        try:
            from .unifier import unir_videos_serie
            titulo_slug = re.sub(r'[\s]+', '_', sanitizar_nombre_archivo(titulo_final, max_longitud=45))
            nombre_pelicula = f"Pelicula_Completa_{titulo_slug}.mp4"
            ruta_pelicula = carpeta_serie / nombre_pelicula

            res_u = unir_videos_serie(
                patron_o_carpeta=str(carpeta_capitulos),
                ruta_salida=ruta_pelicula
            )
            if res_u.get("success"):
                crear_o_actualizar_manifiesto(
                    carpeta_serie,
                    titulo=titulo_final,
                    autor=autor_final,
                    video_unido=nombre_pelicula
                )
        except Exception as e:
            console.print(f"[bold red]Error en unificación automática de videos:[/bold red] {e}")

    if auto_unir and fallidos:
        console.print("[yellow]Unión aplazada: faltan capítulos. Reanude la descarga antes de generar la película.[/yellow]")
    logger.removeHandler(handler)
    handler.close()
    return resultados

