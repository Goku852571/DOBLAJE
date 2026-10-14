from .media import valid_video
"""Adapted internal TikTok module; console CLI and external paths removed."""
from .runtime import paths
from core.media import run_cmd
import os
import sys
import re
import json
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

# Asegurar codificación UTF-8 en consola de Windows

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape

console = Console()
BASE_DIR = paths().root


# Estados posibles de una serie
ESTADO_COMPLETA = "COMPLETA"
ESTADO_UNION_PENDIENTE = "UNION_PENDIENTE"
ESTADO_DESCARGA_INCOMPLETA = "DESCARGA_INCOMPLETA"

def sanitizar_nombre_archivo(texto: str, max_longitud: int = 60, separador: str = "_") -> str:
    """
    Sanitiza una cadena para nombres de carpetas o archivos en Windows NTFS.
    """
    if not texto:
        return "Serie"
    limpio = re.sub(r'[<>:"/\\|?*\x00-\x1f]', separador, str(texto))
    limpio = re.sub(r'[\s_]+', separador, limpio).strip(' ._')
    if not limpio:
        limpio = "Serie"
    if len(limpio) > max_longitud:
        limpio = limpio[:max_longitud].rstrip(' ._')
    return limpio

def extraer_metadatos_de_carpeta(carpeta: Path) -> tuple[str, str]:
    """
    Deduce (autor, titulo) a partir del nombre de la carpeta de serie.
    Ejemplo: "[muffindrama_mx] El secreto de la heredera" -> ("muffindrama_mx", "El secreto de la heredera")
    """
    nombre = carpeta.name
    match = re.match(r'\[(.*?)\]\s*(.*)', nombre)
    if match:
        autor = match.group(1).strip()
        titulo = match.group(2).strip() or "Serie"
        return autor, titulo
    return "tiktok_user", nombre

def clave_orden_natural(ruta_o_nombre) -> list:
    """Orden natural para capítulos (1, 2, ..., 10, 11 en vez de 1, 10, 11, 2)."""
    nombre = ruta_o_nombre.name if hasattr(ruta_o_nombre, "name") else str(ruta_o_nombre)
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r'(\d+)', nombre)]

def buscar_capitulos_locales(carpeta_serie: Path) -> list[Path]:
    """
    Encuentra todos los videos de capítulos válidos (> 100 KB) dentro de la carpeta de la serie.
    Busca preferentemente en carpeta_serie / 'capitulos', o directamente en carpeta_serie si no hay subcarpeta.
    """
    carpeta_caps = carpeta_serie / "capitulos"
    directorio_busqueda = carpeta_caps if carpeta_caps.exists() and carpeta_caps.is_dir() else carpeta_serie

    if not directorio_busqueda.exists():
        return []

    candidatos = [
        f for f in directorio_busqueda.glob("*.mp4")
        if f.is_file() 
        and not f.name.lower().startswith("pelicula_completa_") 
        and not f.name.lower().startswith("serie_completa_")
        and f.stat().st_size > 100 * 1024
        and (not f.stem.endswith("_h264") or valid_video(f))
    ]

    # Deduplicar si existen versiones _h264 y normales
    unicos = {}
    for f in candidatos:
        stem_base = f.stem
        if stem_base.endswith("_h264"):
            stem_base = stem_base[:-5]
        if stem_base in unicos:
            if f.stem.endswith("_h264"):
                unicos[stem_base] = f
        else:
            unicos[stem_base] = f

    lista = list(unicos.values())
    lista.sort(key=clave_orden_natural)
    return lista

def buscar_video_unido(carpeta_serie: Path) -> Path | None:
    """
    Busca si existe el video maestro unificado dentro de la carpeta de la serie.
    """
    if not carpeta_serie.exists():
        return None

    posibles = [
        f for f in carpeta_serie.glob("*.mp4")
        if f.is_file() and (
            f.name.lower().startswith("pelicula_completa_") or 
            f.name.lower().startswith("serie_completa_")
        ) and ".pendiente." not in f.name.lower() and f.stat().st_size > 500 * 1024
    ]
    if posibles:
        # Preferir Pelicula_Completa_ sobre Serie_Completa_ si ambas existen
        posibles.sort(key=lambda x: (not x.name.lower().startswith("pelicula_completa_"), -x.stat().st_size))
        return posibles[0]
    return None

def crear_o_actualizar_manifiesto(
    carpeta_serie: Path,
    titulo: str,
    autor: str,
    descripcion: str = "",
    capitulos: list = None,
    video_unido: str = None
) -> dict:
    """
    Crea o actualiza el archivo info_serie.json e info_serie.md dentro de la carpeta de la serie.
    - Diagnostica automáticamente el estado de la serie (COMPLETA, UNION_PENDIENTE, DESCARGA_INCOMPLETA).
    - Registra número de capítulo, URL original, archivo local y tamaño en MB.
    """
    carpeta_serie = Path(carpeta_serie)
    carpeta_serie.mkdir(parents=True, exist_ok=True)
    carpeta_caps = carpeta_serie / "capitulos"
    carpeta_caps.mkdir(parents=True, exist_ok=True)

    json_path = carpeta_serie / "info_serie.json"
    md_path = carpeta_serie / "info_serie.md"
    txt_path = carpeta_serie / "enlaces_serie.txt"

    # 1. Leer manifiesto previo si existe
    manifiesto_previo = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                manifiesto_previo = json.load(f)
        except Exception:
            manifiesto_previo = {}

    titulo = titulo or manifiesto_previo.get("titulo") or extraer_metadatos_de_carpeta(carpeta_serie)[1]
    autor = autor or manifiesto_previo.get("autor") or extraer_metadatos_de_carpeta(carpeta_serie)[0]
    descripcion = descripcion or manifiesto_previo.get("descripcion") or f"Minidrama de TikTok por @{autor}"

    # 2. Reconciliar enlaces de capítulos
    urls_capitulos = []
    if capitulos:
        for c in capitulos:
            if isinstance(c, dict):
                urls_capitulos.append(c.get("url", ""))
            elif isinstance(c, str):
                urls_capitulos.append(c.strip())
    elif manifiesto_previo.get("capitulos"):
        urls_capitulos = [c.get("url", "") for c in manifiesto_previo.get("capitulos", [])]
    elif txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            urls_capitulos = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Guardar o sincronizar enlaces_serie.txt en la carpeta
    if urls_capitulos and not txt_path.exists():
        with open(txt_path, "w", encoding="utf-8") as f:
            for u in urls_capitulos:
                f.write(f"{u}\n")

    # 3. Detectar archivos locales descargados
    archivos_locales = buscar_capitulos_locales(carpeta_serie)
    total_esperado = len(urls_capitulos) if urls_capitulos else len(archivos_locales)

    # Mapear qué capítulos existen
    mapa_archivos_por_num = {}
    for f in archivos_locales:
        # Extraer número del nombre (Capitulo_01, [autor] 1.mp4, etc.)
        num_match = re.search(r'(\d+)', f.stem)
        if num_match:
            try:
                num = int(num_match.group(1))
                mapa_archivos_por_num[num] = f
            except Exception:
                pass

    lista_capitulos_data = []
    descargados_count = 0

    max_caps = max(total_esperado, len(archivos_locales))
    for i in range(1, max_caps + 1):
        url_cap = urls_capitulos[i - 1] if i - 1 < len(urls_capitulos) else ""
        archivo_encontrado = mapa_archivos_por_num.get(i)
        
        # Si no se encontró por número, intentar por orden si coinciden las longitudes
        if not mapa_archivos_por_num and i - 1 < len(archivos_locales):
            archivo_encontrado = archivos_locales[i - 1]

        nombre_archivo_estandar = f"Capitulo_{i:02d}.mp4"
        esta_descargado = False
        tam_mb = 0.0
        ruta_relativa = f"capitulos/{nombre_archivo_estandar}"

        if archivo_encontrado and archivo_encontrado.exists() and archivo_encontrado.stat().st_size > 100 * 1024:
            esta_descargado = True
            descargados_count += 1
            tam_mb = round(archivo_encontrado.stat().st_size / (1024 * 1024), 2)
            try:
                ruta_relativa = str(archivo_encontrado.relative_to(carpeta_serie)).replace("\\", "/")
            except Exception:
                ruta_relativa = archivo_encontrado.name

        lista_capitulos_data.append({
            "numero": i,
            "nombre_archivo": nombre_archivo_estandar,
            "ruta_relativa": ruta_relativa,
            "url": url_cap,
            "descargado": esta_descargado,
            "tamano_mb": tam_mb
        })

    # 4. Verificar video unido
    master_video = buscar_video_unido(carpeta_serie)
    nombre_video_unido = None
    if master_video:
        nombre_video_unido = master_video.name
    elif video_unido and ".pendiente." not in str(video_unido).lower():
        posible = carpeta_serie / Path(video_unido).name
        if posible.exists():
            nombre_video_unido = posible.name

    # 5. Determinar estado de la serie
    if total_esperado > 0 and descargados_count >= total_esperado:
        if nombre_video_unido:
            estado = ESTADO_COMPLETA
        else:
            estado = ESTADO_UNION_PENDIENTE
    else:
        estado = ESTADO_DESCARGA_INCOMPLETA

    datos_manifiesto = {
        "titulo": titulo,
        "autor": autor,
        "descripcion": descripcion,
        "total_capitulos": total_esperado,
        "capitulos_descargados": descargados_count,
        "capitulos_faltantes": max(0, total_esperado - descargados_count),
        "estado": estado,
        "video_unido": nombre_video_unido,
        "fecha_actualizacion": datetime.now().isoformat(),
        "capitulos": lista_capitulos_data
    }

    # Guardar info_serie.json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(datos_manifiesto, f, indent=2, ensure_ascii=False)

    # Guardar info_serie.md
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Serie: {titulo}\n\n")
        f.write(f"- **Autor:** @{autor}\n")
        f.write(f"- **Estado:** **{estado}**\n")
        f.write(f"- **Capítulos descargados:** {descargados_count}/{total_esperado}\n")
        f.write(f"- **Película Completa:** {f'`{nombre_video_unido}`' if nombre_video_unido else '*(Pendiente de unión)*'}\n")
        f.write(f"- **Descripción:** {descripcion}\n")
        f.write(f"- **Última actualización:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Lista de Capítulos\n\n")
        f.write("| # | Archivo | Estado | Tamaño | Enlace TikTok |\n")
        f.write("|---|---|---|---|---|\n")
        for c in lista_capitulos_data:
            num = c["numero"]
            nom = c["nombre_archivo"]
            desc = "✓ Descargado" if c["descargado"] else "✗ Pendiente"
            tam = f"{c['tamano_mb']} MB" if c["descargado"] else "-"
            enlace = f"[Ver en TikTok]({c['url']})" if c["url"] else "N/A"
            f.write(f"| {num} | `{nom}` | {desc} | {tam} | {enlace} |\n")

    return datos_manifiesto

def escanear_catalogo(descargas_dir: Path = None) -> list[dict]:
    """
    Escanea la carpeta descargas/ buscando subdirectorios de series.
    Diagnostica el estado de cada serie y devuelve la lista completa analizada.
    """
    descargas_dir = Path(descargas_dir) if descargas_dir else paths().downloads
    if not descargas_dir.exists():
        return []

    catalogo = []
    subcarpetas = [d for d in descargas_dir.iterdir() if d.is_dir() and d.name != "perfil_chrome_tiktok"]

    for carpeta in subcarpetas:
        json_path = carpeta / "info_serie.json"
        txt_path = carpeta / "enlaces_serie.txt"
        
        # Deducción de metadatos
        autor, titulo = extraer_metadatos_de_carpeta(carpeta)
        
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Re-validar contra los archivos reales en disco para asegurar frescura
                data = crear_o_actualizar_manifiesto(
                    carpeta_serie=carpeta,
                    titulo=data.get("titulo", titulo),
                    autor=data.get("autor", autor),
                    descripcion=data.get("descripcion", ""),
                    capitulos=data.get("capitulos", []),
                    video_unido=data.get("video_unido")
                )
            except Exception:
                data = crear_o_actualizar_manifiesto(carpeta, titulo=titulo, autor=autor)
        else:
            # Crear manifiesto automáticamente si hay contenido
            caps_locales = buscar_capitulos_locales(carpeta)
            if caps_locales or txt_path.exists():
                data = crear_o_actualizar_manifiesto(carpeta, titulo=titulo, autor=autor)
            else:
                continue

        # Lista de capítulos faltantes
        nums_faltantes = [c["numero"] for c in data.get("capitulos", []) if not c.get("descargado")]
        
        catalogo.append({
            "carpeta": carpeta,
            "titulo": data.get("titulo", titulo),
            "autor": data.get("autor", autor),
            "descripcion": data.get("descripcion", ""),
            "total_capitulos": data.get("total_capitulos", 0),
            "descargados": data.get("capitulos_descargados", 0),
            "faltantes": len(nums_faltantes),
            "nums_faltantes": nums_faltantes,
            "video_unido": data.get("video_unido"),
            "estado": data.get("estado", ESTADO_DESCARGA_INCOMPLETA),
            "manifiesto": data
        })

    catalogo.sort(key=lambda s: s["titulo"].lower())
    return catalogo




