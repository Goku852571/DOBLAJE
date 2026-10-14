"""Adapted internal TikTok module; console CLI and external paths removed."""
from .runtime import paths
from core.media import run_cmd
import sys
import re
import json
import time
from pathlib import Path
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse

# Asegurar codificación UTF-8 en consola de Windows

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape

from .browser import obtener_contexto_persistente, manejar_validaciones_y_modales

console = Console()

class TikTokSeriesExtractor:
    def __init__(self, initial_url: str, output_dir: Path = None):
        self.initial_url = initial_url
        self.output_dir = Path(output_dir) if output_dir else paths().downloads
        self.captured_video_urls = []
        self.playlist_network_items = []
        self.series_title = "Serie_TikTok"
        self.author = None
        self.playlist_id = None
        self.playlist_name = None
        self.playlist_total = None
        self.drama_id = None
        self.captured_api_url = None
        self.api_has_more = False
        self.api_cursor = 0

        match_auth = re.search(r"@([^/?#]+)", self.initial_url)
        if match_auth:
            self.author = match_auth.group(1).lower().lstrip('@')
        self.series_dir = None

    def _intercept_response(self, response):
        """Intercepta respuestas de red buscando APIs de playlist/mix/drama de TikTok."""
        url = response.url
        if any(keyword in url for keyword in ["playlist/item_list", "mix/item_list", "item_list", "series/item_list", "drama/episode/item_list"]):
            try:
                if response.status == 200 and "application/json" in response.headers.get("content-type", ""):
                    data = response.json()

                    if any(k in url for k in ["drama/episode/item_list", "playlist/item_list", "mix/item_list"]):
                        self.captured_api_url = url
                        self.api_has_more = bool(data.get("hasMore"))
                        self.api_cursor = data.get("cursor", 0)
                        if "totalEpisodeCount" in data:
                            try:
                                self.playlist_total = int(data.get("totalEpisodeCount"))
                            except Exception:
                                pass

                    # Extraer metadatos de playlistInfo si están presentes y no se conocen
                    playlist_info = data.get("playlistInfo") or {}
                    if playlist_info:
                        if not self.playlist_id and playlist_info.get("id"):
                            self.playlist_id = str(playlist_info.get("id"))
                        if not self.playlist_name and playlist_info.get("name"):
                            self.playlist_name = playlist_info.get("name")
                            self.series_title = self.playlist_name
                        if not self.playlist_total and playlist_info.get("itemTotal"):
                            try:
                                self.playlist_total = int(playlist_info.get("itemTotal"))
                            except Exception:
                                pass

                    itemList = data.get("itemList") or data.get("items") or []
                    for item in itemList:
                        video_id = item.get("id")
                        if not video_id:
                            continue

                        author_obj = item.get("author") or {}
                        item_author = author_obj.get("uniqueId") or author_obj.get("nickname")

                        # Si self.author es conocido y se detecta el autor del item, verificar coincidencia estricta
                        if self.author and item_author:
                            if item_author.lower().lstrip('@') != self.author:
                                continue

                        author_to_use = self.author or (item_author.lower().lstrip('@') if item_author else "user")
                        clean_url = f"https://www.tiktok.com/@{author_to_use}/video/{video_id}"
                        if clean_url not in self.playlist_network_items:
                            self.playlist_network_items.append(clean_url)
            except Exception:
                pass

    def extraer(self):
        console.print(Panel.fit(
            f"[bold cyan]Extractor RPA de Series de TikTok[/bold cyan]\n"
            f"URL Inicial: [yellow]{self.initial_url}[/yellow]\n"
            f"Salida: [green]{self.output_dir}[/green]",
            title="Iniciando RPA"
        ))

        p, context, page = obtener_contexto_persistente(headless=False)
        page.on("response", self._intercept_response)

        try:
            console.print("[cyan]Navegando al capítulo inicial...[/cyan]")
            page.goto(self.initial_url, wait_until="domcontentloaded", timeout=60000)

            # Esperar carga y gestionar modales
            time.sleep(3)
            manejar_validaciones_y_modales(page)

            # Inmediatamente tras goto y redirecciones, extraer autor de page.url si está presente
            match_page_auth = re.search(r"@([^/?#]+)", page.url)
            if match_page_auth:
                self.author = match_page_auth.group(1).lower().lstrip('@')
                console.print(f"[cyan]Autor detectado tras redirección:[/cyan] [bold]@{self.author}[/bold]")

            # Estrategia 1: Extraer del JSON de hidratación (__UNIVERSAL_DATA_FOR_REHYDRATION__)
            console.print("[yellow]Buscando metadatos de playlist en el estado de la página...[/yellow]")
            try:
                rehydration_data = page.evaluate("""() => {
                    const el = document.getElementById('__UNIVERSAL_DATA_FOR_REHYDRATION__');
                    if (el) {
                        try { return JSON.parse(el.textContent); } catch(e) { return null; }
                    }
                    return null;
                }""")
                if rehydration_data:
                    self._parse_rehydration_data(rehydration_data)
            except Exception as e:
                console.print(f"[dim]Nota estado hidratación: {e}[/dim]")

            # Estrategia 1.5: Paginación automática vía API para series multi-parte (1-24, 25-46, ..., N partes)
            self._paginar_api_serie(page)

            # Estrategia 2: Inspección y apertura del contenedor de Playlist / Serie en el DOM
            # Solo si la API no completó los capítulos reportados por TikTok
            if not self.playlist_total or len(self.playlist_network_items) < self.playlist_total:
                console.print("[yellow]Inspeccionando contenedores de lista de reproducción en el DOM...[/yellow]")
                self._extraer_desde_dom(page)

            # Estrategia 3: Si se obtuvieron pocos enlaces, intentar desplegar o iterar navegación
            min_requerido = self.playlist_total if self.playlist_total else 2
            if len(self.playlist_network_items) < min_requerido and len(self.captured_video_urls) <= 1:
                console.print("[yellow]Intentando navegación secuencial de playlist...[/yellow]")
                self._iterar_por_controles(page)

            # Consolidar y deduplicar enlaces
            todos_los_enlaces = []
            target_author = self.author.lower().lstrip('@') if self.author else None

            for u in self.playlist_network_items + self.captured_video_urls:
                u_clean = self._limpiar_url(u)
                if u_clean and u_clean not in todos_los_enlaces:
                    # Solo aceptar URLs donde f"@{self.author}" esté presente
                    if target_author and f"@{target_author}" not in u_clean.lower():
                        continue
                    todos_los_enlaces.append(u_clean)

            # Si self.playlist_total es conocido y len > self.playlist_total, conservar solo los primeros
            if self.playlist_total and len(todos_los_enlaces) > self.playlist_total:
                todos_los_enlaces = todos_los_enlaces[:self.playlist_total]

            # Fallback seguro: si todos_los_enlaces está vacío, usar la URL actual de la página limpia
            if not todos_los_enlaces:
                fallback_url = self._limpiar_url(page.url) or self._limpiar_url(self.initial_url)
                if fallback_url:
                    todos_los_enlaces.append(fallback_url)

            console.print(f"\n[bold green]Extracción finalizada. Se encontraron {len(todos_los_enlaces)} capítulos en total.[/bold green]")

            # Guardar archivos
            return self._guardar_resultados(todos_los_enlaces)

        except Exception as e:
            console.print(f"[bold red]Error durante la extracción:[/bold red] {e}")
            raise
        finally:
            console.print("[dim]Cerrando sesión del navegador...[/dim]")
            context.close()
            p.stop()

    def _parse_rehydration_data(self, data: dict):
        """Busca listas de reproducción o videos relacionados en el JSON embebido."""
        try:
            default_scope = data.get("__DEFAULT_SCOPE__", {})
            video_detail = default_scope.get("webapp.video-detail", {})
            item_info = video_detail.get("itemInfo", {}).get("itemStruct", {})

            # Obtener autor
            author_data = item_info.get("author", {})
            auth_unique = author_data.get("uniqueId") or author_data.get("nickname")
            if auth_unique:
                self.author = str(auth_unique).lower().lstrip('@')
                console.print(f"[cyan]Autor extraído de metadatos:[/cyan] [bold]@{self.author}[/bold]")

            # 1. Extraer dramaInfo (Minidramas oficiales de TikTok)
            drama_info = item_info.get("dramaInfo") or {}
            if drama_info:
                if drama_info.get("dramaID"):
                    self.drama_id = str(drama_info.get("dramaID"))
                    self.playlist_id = self.drama_id
                if drama_info.get("dramaName"):
                    self.series_title = drama_info.get("dramaName")
                    console.print(f"[bold green]Minidrama/Serie detectado: [bold]{self.series_title}[/bold][/bold green]")
                if drama_info.get("numVideos"):
                    try:
                        self.playlist_total = int(drama_info.get("numVideos"))
                        console.print(f"[cyan]Total capítulos del Minidrama (reportado por TikTok): {self.playlist_total}[/cyan]")
                    except Exception:
                        pass

            # 2. Extraer playlistInfo (Playlists clásicas de TikTok)
            playlist_info = item_info.get("playlistInfo", {})
            if playlist_info:
                if playlist_info.get("id"):
                    self.playlist_id = str(playlist_info.get("id"))
                if playlist_info.get("name"):
                    self.playlist_name = playlist_info.get("name")
                    if self.series_title == "Serie_TikTok":
                        self.series_title = self.playlist_name
                    console.print(f"[green]Nombre de serie detectado: [bold]{self.playlist_name}[/bold][/green]")
                item_total = playlist_info.get("itemTotal")
                if item_total and not self.playlist_total:
                    try:
                        self.playlist_total = int(item_total)
                        console.print(f"[cyan]Total capítulos reportados por TikTok: {self.playlist_total}[/cyan]")
                    except Exception:
                        pass

                # Si playlistInfo o itemStruct tiene array de playlist items/videos, parsear y agregar URLs canónicas
                items_candidatos = (
                    playlist_info.get("itemList") or 
                    playlist_info.get("items") or 
                    playlist_info.get("videoList") or 
                    item_info.get("itemList") or 
                    []
                )
                for it in items_candidatos:
                    it_id = it.get("id") if isinstance(it, dict) else (it if isinstance(it, str) and it.isdigit() else None)
                    it_auth = it.get("author", {}).get("uniqueId") if isinstance(it, dict) else None
                    if it_id:
                        if self.author and it_auth and it_auth.lower().lstrip('@') != self.author:
                            continue
                        auth_slug = self.author or (it_auth.lower().lstrip('@') if it_auth else "user")
                        canonical_url = f"https://www.tiktok.com/@{auth_slug}/video/{it_id}"
                        if canonical_url not in self.playlist_network_items:
                            self.playlist_network_items.append(canonical_url)

            elif self.series_title == "Serie_TikTok":
                desc = item_info.get("desc")
                if desc:
                    desc_clean = re.sub(r'#\S+', '', desc).strip()
                    if desc_clean:
                        self.series_title = desc_clean[:45]
                        console.print(f"[green]Título aproximado detectado: [bold]{self.series_title}[/bold][/green]")
        except Exception as e:
            console.print(f"[dim]Nota estado hidratación: {e}[/dim]")

    def _paginar_api_serie(self, page):
        """Pagina automáticamente a través de la API autenticada de TikTok para obtener todos los capítulos de todas las partes (1-24, 25-46, ..., N)."""
        if not self.captured_api_url:
            return

        cursor = self.api_cursor
        has_more = self.api_has_more

        # Si ya tenemos todos los capítulos esperados y has_more es False, nada pendiente
        if self.playlist_total and len(self.playlist_network_items) >= self.playlist_total:
            return

        total_txt = str(self.playlist_total) if self.playlist_total else "desconocido"
        console.print(f"[bold cyan]Paginando serie multi-parte vía API de TikTok (capítulos actuales: {len(self.playlist_network_items)} de {total_txt})...[/bold cyan]")

        intentos_max = 30 # Soporta series de hasta 720 capítulos
        intento = 0

        while has_more and intento < intentos_max:
            intento += 1
            if self.playlist_total and len(self.playlist_network_items) >= self.playlist_total:
                break

            try:
                parsed = urlparse(self.captured_api_url)
                params = parse_qs(parsed.query)
                params_flat = {k: v[0] for k, v in params.items()}
                params_flat["cursor"] = str(cursor)

                query_next = urlencode(params_flat)
                next_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query_next, parsed.fragment))

                console.print(f"[cyan]Consultando parte {intento + 1} (cursor={cursor})...[/cyan]")
                res_data = page.evaluate("""async (fetchUrl) => {
                    try {
                        const resp = await fetch(fetchUrl);
                        return await resp.json();
                    } catch(e) {
                        return null;
                    }
                }""", next_url)

                if not res_data:
                    break

                status = res_data.get("statusCode") if "statusCode" in res_data else res_data.get("status_code", 0)
                if status != 0:
                    break

                item_list = res_data.get("itemList") or res_data.get("items") or []
                if not item_list:
                    break

                nuevos_en_lote = 0
                for item in item_list:
                    vid_id = item.get("id")
                    if not vid_id:
                        continue

                    author_obj = item.get("author") or {}
                    item_author = author_obj.get("uniqueId") or author_obj.get("nickname")
                    if self.author and item_author:
                        if item_author.lower().lstrip('@') != self.author:
                            continue

                    clean_url = f"https://www.tiktok.com/@{self.author or 'user'}/video/{vid_id}"
                    if clean_url not in self.playlist_network_items:
                        self.playlist_network_items.append(clean_url)
                        nuevos_en_lote += 1

                console.print(f"[green]+ Añadidos {nuevos_en_lote} capítulos de la parte {intento + 1} (Total acumulado: {len(self.playlist_network_items)} de {total_txt})[/green]")

                has_more = bool(res_data.get("hasMore"))
                raw_cursor = res_data.get("cursor")
                if raw_cursor is not None:
                    try:
                        cursor = int(raw_cursor)
                    except Exception:
                        cursor = int(cursor) + len(item_list)
                else:
                    cursor = int(cursor) + len(item_list)
                time.sleep(0.5)
            except Exception as e:
                console.print(f"[yellow]Aviso en paginación de serie:[/yellow] {e}")
                break

    def _extraer_desde_dom(self, page):
        """Busca elementos visuales de la playlist y sus links ÚNICAMENTE dentro de contenedores legítimos de playlist."""
        try:
            # Buscar contenedores de playlist o barra lateral de capítulos legítimos
            playlist_selectors = [
                "[data-e2e='playlist-bar']",
                "[data-e2e='playlist-video-item']",
                "div[class*='DivPlaylistContainer']",
                "div[class*='DivPlaylistVideoList']",
                "div[class*='PlaylistVideoList']",
                "aside[class*='playlist']",
                "div[data-e2e='video-playlist-item']",
            ]

            enlaces_raw = []
            for sel in playlist_selectors:
                try:
                    elements = page.locator(sel)
                    count = elements.count()
                    if count > 0:
                        console.print(f"[green]Detectado contenedor con selector: {sel} ({count} elementos)[/green]")
                        for i in range(count):
                            el = elements.nth(i)
                            tag = el.evaluate("e => e.tagName.toLowerCase()", timeout=1000)
                            if tag == "a":
                                href = el.get_attribute("href")
                                if href and "/video/" in href:
                                    enlaces_raw.append(href)
                            # Subenlaces <a> dentro del contenedor
                            sub_links = el.locator("a[href*='/video/']").all()
                            for link in sub_links:
                                href = link.get_attribute("href")
                                if href:
                                    enlaces_raw.append(href)
                except Exception:
                    pass

            target_auth = self.author.lower().lstrip('@') if self.author else None

            # Filtrar enlaces: NUNCA escanear toda la página. Solo los encontrados en contenedores legítimos.
            for href in enlaces_raw:
                try:
                    url_absoluta = urljoin("https://www.tiktok.com", href)
                    clean = self._limpiar_url(url_absoluta)
                    if not clean:
                        continue

                    # Estrictamente requerir que contenga f"@{self.author.lower()}/video/" o "@" si autor es desconocido
                    clean_lower = clean.lower()
                    if target_auth:
                        if f"@{target_auth}/video/" not in clean_lower:
                            continue
                    else:
                        if "/@" not in clean_lower:
                            continue

                    if clean not in self.captured_video_urls:
                        self.captured_video_urls.append(clean)
                except Exception:
                    pass

            console.print(f"[cyan]Enlaces válidos de playlist encontrados en el DOM: {len(self.captured_video_urls)}[/cyan]")
        except Exception as e:
            console.print(f"[dim]Error escaneando DOM: {e}[/dim]")

    def _iterar_por_controles(self, page):
        """Si la lista no cargó todos los capítulos, navega exclusivamente con controles específicos de playlist."""
        min_requerido = self.playlist_total if self.playlist_total else 2
        if len(self.playlist_network_items) >= min_requerido:
            console.print(
                f"[green]Capítulos de la serie ya obtenidos vía API de playlist "
                f"({len(self.playlist_network_items)} de {self.playlist_total or '?'}). "
                f"Omitiendo navegación de feed.[/green]"
            )
            return

        try:
            # Controles EXCLUSIVOS de playlist (se eliminan botones de feed FYP como 'Siguiente video' o 'Next video')
            next_btn_selectors = [
                "[data-e2e='playlist-arrow-right']",
                "[data-e2e='playlist-next']",
                "button[aria-label='Siguiente episodio']",
            ]

            target_auth = self.author.lower().lstrip('@') if self.author else None

            for _ in range(self.playlist_total or 50): # Límite de seguridad
                current_url = self._limpiar_url(page.url)

                # Verificar autor de la URL actual: si difiere de self.author, BREAK inmediatamente
                if target_auth and current_url:
                    if f"@{target_auth}/video/" not in current_url.lower():
                        console.print(f"[yellow]Navegación desvió fuera del autor @{target_auth}. Deteniendo iteración.[/yellow]")
                        break

                if current_url and current_url not in self.captured_video_urls:
                    self.captured_video_urls.append(current_url)

                # Si self.playlist_total es conocido y len >= self.playlist_total, BREAK inmediatamente
                if self.playlist_total and len(self.captured_video_urls) >= self.playlist_total:
                    break

                # Buscar y hacer clic en el botón siguiente exclusivo de playlist
                clicked = False
                for sel in next_btn_selectors:
                    btn = page.locator(sel).first
                    try:
                        if btn.is_visible(timeout=500):
                            btn.click()
                            page.wait_for_timeout(1500)
                            manejar_validaciones_y_modales(page)
                            clicked = True
                            break
                    except Exception:
                        pass

                if not clicked:
                    break
        except Exception as e:
            console.print(f"[dim]Fin de navegación secuencial: {e}[/dim]")

    def _limpiar_url(self, raw_url: str) -> str:
        """Limpia parámetros de tracking como ?is_from_webapp=1&sender_device=pc."""
        if not raw_url:
            return ""
        # Buscar patrón canónico tiktok.com/@user/video/ID o tiktok.com/video/ID
        match = re.search(r"(https?://(?:www\.)?tiktok\.com/@[^/?#]+/video/\d+)", raw_url)
        if match:
            return match.group(1)
        match_alt = re.search(r"(https?://(?:www\.)?tiktok\.com/video/\d+)", raw_url)
        if match_alt:
            return match_alt.group(1)
        # Fallback a remover query params
        parsed = urlparse(raw_url)
        if "/video/" in parsed.path:
            return f"https://www.tiktok.com{parsed.path}"
        return ""

    def _guardar_resultados(self, urls: list) -> Path:
        """Guarda los enlaces en .txt y .md en la carpeta raíz y en la carpeta dedicada de la serie."""
        from .catalog import sanitizar_nombre_archivo, crear_o_actualizar_manifiesto

        descargas_dir = paths().downloads
        descargas_dir.mkdir(parents=True, exist_ok=True)

        autor_limpio = sanitizar_nombre_archivo(self.author or "tiktok_user", max_longitud=30, separador="_")
        titulo_limpio = sanitizar_nombre_archivo(self.series_title, max_longitud=50, separador=" ")
        nombre_carpeta = f"[{autor_limpio}] {titulo_limpio}"

        self.series_dir = descargas_dir / nombre_carpeta
        self.series_dir.mkdir(parents=True, exist_ok=True)
        (self.series_dir / "capitulos").mkdir(exist_ok=True)

        # 1. Guardar enlaces_serie.txt e info_serie en la carpeta dedicada de la serie
        txt_serie = self.series_dir / "enlaces_serie.txt"
        with open(txt_serie, "w", encoding="utf-8") as f:
            for url in urls:
                f.write(f"{url}\n")

        # Crear manifiesto JSON y MD inicial
        crear_o_actualizar_manifiesto(
            carpeta_serie=self.series_dir,
            titulo=self.series_title,
            autor=self.author or "tiktok_user",
            descripcion=f"Serie extraída de TikTok ({len(urls)} capítulos)",
            capitulos=urls
        )

        # 2. Guardar también en self.output_dir (ej: raíz del proyecto) para retrocompatibilidad
        md_path = self.output_dir / "enlaces_serie.md"
        txt_path = self.output_dir / "enlaces_serie.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            for url in urls:
                f.write(f"{url}\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Enlaces de la Serie: {self.series_title}\n\n")
            f.write(f"- **Total de capítulos extraídos:** {len(urls)}\n")
            f.write(f"- **URL inicial proporcionada:** {self.initial_url}\n\n")
            f.write("## Lista de Capítulos\n\n")
            for idx, url in enumerate(urls, 1):
                f.write(f"{idx}. [{url}]({url})\n")

        table = Table(title="Resultados y Manifiesto Guardados con Éxito")
        table.add_column("Elemento", style="cyan")
        table.add_column("Ubicación", style="green")
        table.add_column("Detalle", style="magenta")

        table.add_row("Carpeta de la Serie", escape(str(self.series_dir)), f"{len(urls)} capítulos")
        table.add_row("Manifiesto JSON", escape(str(self.series_dir / "info_serie.json")), "Metadatos y estado")
        table.add_row("Manifiesto Markdown", escape(str(self.series_dir / "info_serie.md")), "Documentación")
        table.add_row("Enlaces Serie (.txt)", escape(str(txt_serie)), f"{len(urls)} links")
        console.print(table)

        return self.series_dir

