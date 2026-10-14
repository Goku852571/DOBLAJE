"""Internal TikTok pipeline, isolated from ASR/TTS and independent of other projects."""
import hashlib
import json
import os
from pathlib import Path
import sys

from core.tiktok.inputs import read_urls, validate_url
from core.tiktok.media import validate_video
from core.tiktok.runtime import configure, acquire_pipeline_lock


def execute(request, root=None):
    result = dict(success=False, status="ERROR", series_dir="", chapters_found=0,
                  chapters_downloaded=0, chapters_failed=[], merged_video="", error="")
    lock = None
    try:
        mode = request["mode"]
        if mode in ("url", "single"):
            validate_url(request["source"])
        elif mode == "txt":
            urls = read_urls(request["source"])
        elif mode != "login":
            raise ValueError("Modo TikTok desconocido.")
        resources = configure(request.get("settings"), root=root)
        lock = acquire_pipeline_lock(resources)
        from core.tiktok import extractor, downloader, unifier, catalog
        if mode == "login":
            from core.tiktok.browser import iniciar_sesion
            print("VIDEO_SUB_STAGE Inicio de sesión en el perfil interno", flush=True)
            iniciar_sesion()
            result.update(success=True, status="SESSION_READY")
            return result
        if mode == "url":
            print("VIDEO_SUB_STAGE Extracción de capítulos", flush=True)
            engine = extractor.TikTokSeriesExtractor(
                request["source"], output_dir=resources.downloads)
            folder = engine.extraer()
            if not folder:
                raise RuntimeError("La extracción no produjo una serie.")
            extracted = read_urls(Path(folder) / "enlaces_serie.txt")
            result.update(series_dir=str(folder), chapters_found=len(extracted))
            total = getattr(engine, "playlist_total", None)
            if (total and len(extracted) < int(total)) or (
                    not total and getattr(engine, "api_has_more", False)):
                raise RuntimeError("Extracción incompleta: %s capítulos de %s." % (
                    len(extracted), total or "un total pendiente"))
            if len(extracted) < 2:
                raise RuntimeError("No se confirmó una serie. Usa Video individual para descargar un solo video.")
            urls = extracted
            title = engine.series_title
            author = engine.author or "tiktok_user"
        else:
            if mode == "single":
                urls = [request["source"]]
            title = Path(request["source"]).stem if mode == "txt" else "Video_TikTok"
            author = "tiktok_user"
        # Stable URL-list identity prevents resuming a different episode into
        # the same chapter number after an extractor reorders/changes a series.
        digest = hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest()[:16]
        name = catalog.sanitizar_nombre_archivo(title, max_longitud=40)
        serie = resources.downloads / (name + "_" + digest)
        serie.mkdir(parents=True, exist_ok=True)
        links = serie / "enlaces_serie.txt"
        links.write_text("\n".join(urls) + "\n", encoding="utf-8")
        result.update(series_dir=str(serie), chapters_found=len(urls))
        print("VIDEO_SUB_STAGE Descarga y reintentos", flush=True)
        downloads = downloader.descargar_desde_archivo(str(links), serie_dir=serie, auto_unir=False)
        catalog.crear_o_actualizar_manifiesto(serie, titulo=title, autor=author, capitulos=urls)
        expected = []
        print("VIDEO_SUB_STAGE Validación de capítulos", flush=True)
        for index, url in enumerate(urls, 1):
            base = serie / "capitulos" / ("Capitulo_%02d.mp4" % index)
            variants = [base.with_name(base.stem + "_h264.mp4"), base]
            record = downloads[index - 1] if index <= len(downloads) else {}
            reported = Path(record.get("archivo") or "")
            path = next((p for p in variants if p.resolve() == reported.resolve()
                         and p.is_file() and p.stat().st_size > 100 * 1024), None)
            error = record.get("error") or ""
            if path and record.get("success"):
                try:
                    validate_video(path)
                except Exception as exc:
                    error = str(exc)
            else:
                error = error or "Capítulo ausente o inválido."
            if error:
                result["chapters_failed"].append(dict(number=index, url=url, error=error))
            else:
                expected.append(path)
        result["chapters_downloaded"] = len(expected)
        if result["chapters_failed"]:
            result["status"] = "DOWNLOAD_INCOMPLETE"
            raise RuntimeError("Descarga incompleta: " + "; ".join(
                "capítulo %s: %s" % (c["number"], c["error"]) for c in result["chapters_failed"]))
        actual = unifier.buscar_videos_serie(serie / "capitulos")
        if [p.resolve() for p in actual] != [p.resolve() for p in expected]:
            raise RuntimeError("Los archivos para unir no coinciden con los capítulos esperados en orden.")
        print("VIDEO_SUB_STAGE Unión y normalización de audio", flush=True)
        result["status"] = "MERGE_PENDING"
        destination = serie / "Pelicula_Completa_Video_Sub.pendiente.mp4"
        merged = unifier.unir_videos_serie(
            patron_o_carpeta=str(serie / "capitulos"), ruta_salida=destination)
        if not merged.get("success"):
            raise RuntimeError(merged.get("error") or "La unión quedó pendiente.")
        path = Path(merged.get("archivo_salida") or "")
        if not path.is_file() or merged.get("total_capitulos") != len(urls):
            raise RuntimeError("La unión no devolvió todos los capítulos.")
        print("VIDEO_SUB_STAGE Verificación del MP4", flush=True)
        validate_video(path)
        if path == destination:
            final = serie / "Pelicula_Completa_Video_Sub.mp4"
            os.replace(path, final)
            path = final
        catalog.crear_o_actualizar_manifiesto(
            serie, titulo=title, autor=author, capitulos=urls, video_unido=path.name)
        result.update(success=True, status="READY", merged_video=str(path.resolve()))
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if lock:
            lock.close()
    return result


if __name__ == "__main__":
    request_file, result_file = map(Path, sys.argv[1:3])
    request = json.loads(request_file.read_text(encoding="utf-8"))
    response = execute(request)
    result_file.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(0 if response["success"] else 1)
