from .media import valid_video
"""Adapted internal TikTok module; console CLI and external paths removed."""
from .runtime import paths
from core.media import run_cmd
import os
import sys
import re
import json
import time
import tempfile
import subprocess
from pathlib import Path
from collections import Counter

# Asegurar codificación UTF-8 en consola de Windows

from core.media import _ffmpeg_binary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape

console = Console()
BASE_DIR = paths().root


def sanitizar_nombre_archivo(texto: str, max_longitud: int = 50) -> str:
    """
    Sanitiza una cadena para que sea válida en nombres de archivo en Windows NTFS.
    """
    if not texto:
        return "serie"
    limpio = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(texto))
    limpio = re.sub(r'[\s_]+', ' ', limpio).strip(' .')
    return limpio[:max_longitud].rstrip(' .') or "serie"

def clave_orden_natural(ruta_o_nombre) -> list:
    """
    Permite ordenar archivos por número de capítulo en orden natural:
    1, 2, ..., 9, 10, 11, ... 24 (en lugar de orden alfabético 1, 10, 11, 2...).
    """
    nombre = ruta_o_nombre.name if hasattr(ruta_o_nombre, "name") else str(ruta_o_nombre)
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r'(\d+)', nombre)]

def buscar_videos_serie(carpeta: Path, patron: str = None) -> list[Path]:
    """
    Encuentra y ordena naturalmente los videos de una serie en la carpeta especificada.
    Busca preferentemente en carpeta / 'capitulos' si existe, o en la carpeta raíz.
    Descarta videos parciales (< 100 KB) y archivos maestros de 'Pelicula_Completa_' y 'Serie_Completa_'.
    Valida contra info_serie.json para limitar estrictamente a los capítulos de la serie.
    """
    if not carpeta.exists():
        return []

    # Si existe subdirectorio 'capitulos', buscar allí preferentemente
    carpeta_caps = carpeta / "capitulos"
    directorio_busqueda = carpeta_caps if carpeta_caps.is_dir() else carpeta

    todos = [f for f in directorio_busqueda.glob("*.mp4") if f.is_file()]
    
    # Filtrar archivos especiales de unión previa y archivos corruptos (< 100 KB)
    candidatos = [
        f for f in todos 
        if not f.name.lower().startswith("serie_completa_") 
        and not f.name.lower().startswith("pelicula_completa_")
        and f.stat().st_size > 100 * 1024
        and (not f.stem.endswith("_h264") or valid_video(f))
    ]

    if not candidatos:
        return []

    if patron:
        patron_limpio = patron.strip().lower()
        patron_limpio = patron_limpio.strip("[]")
        filtrados = [f for f in candidatos if patron_limpio in f.name.lower()]
    else:
        # Auto-detección: agrupar por etiqueta [autor] si existe
        grupos = Counter()
        for f in candidatos:
            match = re.search(r'\[(.*?)\]', f.name)
            if match:
                grupos[match.group(1).strip()] += 1

        if grupos:
            serie_principal = grupos.most_common(1)[0][0]
            filtrados = [f for f in candidatos if f"[{serie_principal}]" in f.name]
        else:
            filtrados = candidatos

    # Descartar duplicados si existen versiones normales y _h264 del mismo video
    videos_unicos = {}
    for f in filtrados:
        stem_base = f.stem
        if stem_base.endswith("_h264"):
            stem_base = stem_base[:-5]
        
        if stem_base in videos_unicos:
            if f.stem.endswith("_h264"):
                videos_unicos[stem_base] = f
        else:
            videos_unicos[stem_base] = f

    lista_final = list(videos_unicos.values())
    # Orden natural por número de capítulo
    lista_final.sort(key=clave_orden_natural)

    # Validar contra el manifiesto info_serie.json si existe en carpeta o en su carpeta contenedora
    info_json_path = None
    if (carpeta / "info_serie.json").is_file():
        info_json_path = carpeta / "info_serie.json"
    elif carpeta.parent and (carpeta.parent / "info_serie.json").is_file():
        info_json_path = carpeta.parent / "info_serie.json"

    if info_json_path:
        try:
            with open(info_json_path, "r", encoding="utf-8") as f:
                info_data = json.load(f)
                total_capitulos = info_data.get("total_capitulos")
                capitulos_meta = info_data.get("capitulos")
                if not total_capitulos and capitulos_meta and isinstance(capitulos_meta, list):
                    total_capitulos = len(capitulos_meta)

                if total_capitulos and total_capitulos > 0:
                    # Descartar videos con número de capítulo mayor al total definido
                    filtrados_manifiesto = []
                    for v in lista_final:
                        num_m = re.search(r'(\d+)', v.stem)
                        if num_m:
                            num = int(num_m.group(1))
                            if num <= total_capitulos:
                                filtrados_manifiesto.append(v)
                        else:
                            filtrados_manifiesto.append(v)
                    lista_final = filtrados_manifiesto

                    # Limitar estrictamente al total de capítulos del manifiesto
                    if len(lista_final) > total_capitulos:
                        lista_final = lista_final[:total_capitulos]
        except Exception:
            pass

    return lista_final

def obtener_duracion_video(ruta: Path, ffmpeg_exe: str) -> str:
    """
    Obtiene la duración formateada (HH:MM:SS) de un archivo MP4 usando FFmpeg.
    """
    try:
        res = run_cmd(
            [ffmpeg_exe, "-i", str(ruta)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )
        match = re.search(r'Duration:\s*(\d{2}:\d{2}:\d{2})', res.stderr)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "N/D"

def unir_videos_serie(patron_o_carpeta: str = None, ruta_salida: Path = None, output_dir: Path = None) -> dict:
    """
    Une todos los capítulos de una serie en un único video consolidado:
    - Destino por defecto: descargas/Serie_Completa_[Nombre].mp4
    - Genera archivo de concatenación temporal para el demuxer concat de FFmpeg.
    - Normaliza el audio de cada capítulo a PCM estéreo de 48 kHz antes de concatenar.
    - Primer intento: copia video y codifica audio AAC-LC a 192 kb/s.
    - Segundo intento (fallback por stream mismatch): transcodificación rápida (-c:v libx264 -crf 20 -preset veryfast -c:a aac).
    - Muestra retroalimentación estilizada en consola Rich y tabla resumen con tamaño y duración.
    """
    inicio_tiempo = time.time()
    
    # Resolver directorio de trabajo
    carpeta = paths().downloads
    patron = None

    if patron_o_carpeta:
        posible_path = Path(patron_o_carpeta)
        if posible_path.is_dir():
            carpeta = posible_path
        else:
            patron = str(patron_o_carpeta)

    if output_dir:
        carpeta = Path(output_dir)

    carpeta.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        f"[bold cyan]Módulo Unificador de Series TikTok (FFmpeg)[/bold cyan]\n"
        f"Directorio de origen: [yellow]{carpeta}[/yellow]\n"
        f"Filtro/Patrón: [green]{patron or '(Auto-detección)'}[/green]",
        title="Unificador de Videos"
    ))

    videos = buscar_videos_serie(carpeta, patron=patron)

    if not videos:
        console.print(f"[bold red]✗ No se encontraron videos válidos para unir en:[/bold red] {carpeta}")
        return {
            "success": False,
            "error": "No se encontraron videos válidos para concatenar"
        }

    if len(videos) < 2:
        console.print(f"[yellow]Aviso: Solo se encontró 1 video ({escape(videos[0].name)}). No se requiere unión.[/yellow]")
        return {
            "success": True,
            "archivo_salida": str(videos[0]),
            "total_capitulos": 1,
            "tamano_mb": round(videos[0].stat().st_size / (1024 * 1024), 2),
            "metodo": "Individual (No concatenado)"
        }

    # Determinar carpeta de la serie y nombre maestro
    carpeta_serie = carpeta.parent if carpeta.name.lower() == "capitulos" else carpeta
    
    # 1. Intentar obtener título y autor desde info_serie.json
    titulo_serie = None
    autor_serie = None
    json_path = carpeta_serie / "info_serie.json"
    if json_path.exists():
        try:
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                info_j = json.load(f)
                titulo_serie = info_j.get("titulo")
                autor_serie = info_j.get("autor")
        except Exception:
            pass

    # 2. Si no, extraer de la carpeta [autor] Titulo
    if not titulo_serie and carpeta_serie != paths().downloads:
        match_folder = re.match(r'\[(.*?)\]\s*(.*)', carpeta_serie.name)
        if match_folder:
            autor_serie = match_folder.group(1).strip()
            titulo_serie = match_folder.group(2).strip() or match_folder.group(1).strip()

    # 3. Si no, extraer de patron o del primer video
    if not titulo_serie:
        primer_archivo = videos[0].name
        match_tag = re.search(r'\[(.*?)\]', primer_archivo)
        if match_tag:
            titulo_serie = match_tag.group(1).strip()
            autor_serie = match_tag.group(1).strip()
        elif patron:
            titulo_serie = patron.strip().strip("[]")
        else:
            titulo_serie = "Serie"

    nombre_serie = titulo_serie
    nombre_serie_limpio = sanitizar_nombre_archivo(titulo_serie, max_longitud=45)
    nombre_slug = re.sub(r'[\s]+', '_', nombre_serie_limpio)
    nombre_archivo_pelicula = f"Pelicula_Completa_{nombre_slug}.mp4"
    
    if ruta_salida:
        destino = Path(ruta_salida)
        if not destino.is_absolute():
            destino = carpeta_serie / destino
    else:
        destino = carpeta_serie / nombre_archivo_pelicula

    # Mostrar lista de capítulos detectados en orden natural
    tabla_capitulos = Table(title=f"Capítulos a Unir ({len(videos)} videos ordenados cronológicamente)", show_header=True)
    tabla_capitulos.add_column("#", justify="center", style="dim", width=4)
    tabla_capitulos.add_column("Archivo de Capítulo", style="cyan")
    tabla_capitulos.add_column("Tamaño", justify="right", style="magenta")

    for idx, v in enumerate(videos, 1):
        tam = f"{round(v.stat().st_size / (1024 * 1024), 2)} MB"
        tabla_capitulos.add_row(str(idx), escape(v.name), tam)

    console.print(tabla_capitulos)

    # Crear archivo temporal para concat demuxer
    concat_temp = None
    normalizados_temp = tempfile.TemporaryDirectory(prefix="union_audio_", dir=paths().temp)
    try:
        ffmpeg_exe = _ffmpeg_binary()
        normalizados = []
        for idx, video in enumerate(videos):
            temporal = Path(normalizados_temp.name) / f"cap_{idx:04d}.mkv"
            console.print(f"[cyan]Normalizando audio: {escape(video.name)}[/cyan]")
            proceso = run_cmd(
                [ffmpeg_exe, "-y", "-v", "error", "-xerror", "-i", str(video),
                 "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
                 "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
                 "-af", "aresample=async=1:first_pts=0", "-nostdin", str(temporal)],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proceso.returncode:
                return {"success": False, "error": f"Audio inválido o ausente en {video.name}: {proceso.stderr}"}
            normalizados.append(temporal)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", dir=paths().temp) as f:
            concat_temp = Path(f.name)
            for v in normalizados:
                # Rutas con barras normales y escape de apóstrofes para FFmpeg
                posix_path = v.resolve().as_posix().replace("'", "'\\''")
                f.write(f"file '{posix_path}'\n")

        ffmpeg_exe = _ffmpeg_binary()
        metodo_utilizado = ""

        # Intento 1: Unión Lossless Ultra-rápida (-c copy)
        console.print("\n[cyan]Uniendo video y codificando audio AAC-LC 48 kHz estéreo...[/cyan]")
        cmd_copy = [
            ffmpeg_exe,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_temp),
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "copy", "-c:a", "aac", "-profile:a", "aac_low",
            "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart",
            str(destino)
        ]

        proceso_copy = run_cmd(
            cmd_copy,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        if proceso_copy.returncode == 0 and destino.exists() and destino.stat().st_size > 100 * 1024:
            metodo_utilizado = "Video original + audio normalizado AAC-LC 192k / 48 kHz / estéreo"
            console.print("[bold green]✓ Fusión con audio AAC compatible completada.[/bold green]")
        else:
            # Intento 2: Fallback por stream mismatch (Transcodificación rápida H.264 + AAC)
            console.print(f"[yellow]Aviso: La unión directa por copia detectó discrepancia en parámetros de stream.[/yellow]")
            console.print(f"[cyan]Paso 2: Ejecutando transcodificación rápida H.264 (-c:v libx264 -crf 20 -preset veryfast -c:a aac)...[/cyan]")

            cmd_transcode = [
                ffmpeg_exe,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_temp),
                "-c:v", "libx264",
                "-crf", "20",
                "-preset", "veryfast",
                "-c:a", "aac",
                "-profile:a", "aac_low", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(destino)
            ]

            proceso_transcode = run_cmd(
                cmd_transcode,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )

            if proceso_transcode.returncode == 0 and destino.exists() and destino.stat().st_size > 100 * 1024:
                metodo_utilizado = "Transcodificación Universal (libx264 + aac)"
                console.print(f"[bold green]✓ Fusión y normalización de video completada exitosamente.[/bold green]")
            else:
                err_msg = proceso_transcode.stderr or proceso_copy.stderr
                console.print(f"[bold red]✗ Error al concatenar videos con FFmpeg:[/bold red]\n{err_msg[-400:]}")
                return {
                    "success": False,
                    "error": err_msg
                }

    finally:
        normalizados_temp.cleanup()
        # Limpieza de archivo temporal concat
        if concat_temp and concat_temp.exists():
            try:
                concat_temp.unlink(missing_ok=True)
            except Exception:
                pass

    duracion_seg = round(time.time() - inicio_tiempo, 1)
    tam_mb_final = round(destino.stat().st_size / (1024 * 1024), 2)
    duracion_video_str = obtener_duracion_video(destino, ffmpeg_exe)

    # Tabla resumen final
    tabla_resumen = Table(title="[bold yellow]Resumen de Unificación de Serie[/bold yellow]", show_header=True)
    tabla_resumen.add_column("Propiedad", style="cyan", width=22)
    tabla_resumen.add_column("Detalle", style="white")

    tabla_resumen.add_row("Serie / Autor", nombre_serie)
    tabla_resumen.add_row("Capítulos Unidos", f"[green]{len(videos)} videos[/green]")
    tabla_resumen.add_row("Archivo Maestro", f"[bold white]{escape(destino.name)}[/bold white]")
    tabla_resumen.add_row("Ruta Completa", escape(str(destino)))
    tabla_resumen.add_row("Tamaño Total", f"[bold magenta]{tam_mb_final} MB[/bold magenta]")
    tabla_resumen.add_row("Duración Total Video", f"[bold yellow]{duracion_video_str}[/bold yellow]")
    tabla_resumen.add_row("Método FFmpeg", metodo_utilizado)
    tabla_resumen.add_row("Tiempo de Proceso", f"{duracion_seg} segundos")

    console.print("\n")
    console.print(tabla_resumen)
    console.print(Panel(
        f"[bold green]¡Película completa de la serie generada con éxito![/bold green]\n"
        f"Archivo listo para reproducir: [yellow]{escape(str(destino))}[/yellow]\n"
        f"Duración: [cyan]{duracion_video_str}[/cyan] • Tamaño: [magenta]{tam_mb_final} MB[/magenta]",
        title="Operación Exitosa"
    ))

    # Sincronizar manifiesto de la serie si corresponde
    if destino.exists() and carpeta_serie != paths().downloads:
        try:
            from .catalog import crear_o_actualizar_manifiesto
            crear_o_actualizar_manifiesto(
                carpeta_serie=carpeta_serie,
                titulo=titulo_serie,
                autor=autor_serie,
                video_unido=destino.name
            )
        except Exception:
            pass

    return {
        "success": True,
        "serie": nombre_serie,
        "archivo_salida": str(destino),
        "total_capitulos": len(videos),
        "tamano_mb": tam_mb_final,
        "duracion": duracion_video_str,
        "metodo": metodo_utilizado,
        "tiempo_seg": duracion_seg
    }

