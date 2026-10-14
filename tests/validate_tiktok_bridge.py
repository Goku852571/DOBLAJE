"""Offline integration checks: mock network/browser only, use real downloader/FFmpeg."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio_ffmpeg
from core.tiktok import extractor as extractor_series
from core.tiktok import downloader as descargador_video
from core.tiktok import unifier as unificador_videos
from core.tiktok import browser as browser_manager
from core.tiktok.runtime import paths
from core.tiktok_adapter import execute

urls = ["https://www.tiktok.com/@prueba/video/100",
        "https://www.tiktok.com/@prueba/video/200"]
reports = []
original_profile = paths().profile
with tempfile.TemporaryDirectory(prefix="Video Sub á ñ [serie] ' ") as folder:
    root = Path(folder).resolve()
    fixture = root / "muestra.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4",
                    "-c:v", "mpeg4", "-q:v", "2", "-c:a", "aac", str(fixture)], check=True)
    assert fixture.stat().st_size > 100 * 1024
    source = root / "enlaces á ñ.txt"
    source.write_text("\n".join(urls + [urls[0]]) + "\n", encoding="utf-8-sig")

    def fake_extract(self):
        self.author = "prueba"
        self.series_title = "Serie de prueba"
        self.playlist_total = 2
        return self._guardar_resultados(urls)

    def transfer(url, destination, **kwargs):
        shutil.copyfile(fixture, destination)
        return True

    def info(url):
        return dict(success=True, play_url="https://fixture.invalid/video",
                    autor="prueba", titulo="Prueba", id=url.rsplit("/", 1)[-1])

    def run(label, mode="txt", **changes):
        request = dict(settings={"tiktok_download_dir": "data/descargas"},
                       mode=mode, source=urls[0] if mode in ("url", "single") else str(source))
        request.update(changes)
        with contextlib.redirect_stdout(io.StringIO()):
            result = execute(request, root=root / label)
        reports.append(dict(case=label, result=result))
        return result

    with patch.object(extractor_series.TikTokSeriesExtractor, "extraer", fake_extract), patch.object(
            descargador_video, "_obtener_info_tikwm", side_effect=info), patch.object(
            descargador_video, "_descargar_stream", side_effect=transfer), patch(
            "time.sleep", return_value=None):
        first = run("txt")
        assert first["success"], first
        assert first["chapters_found"] == first["chapters_downloaded"] == 2
        assert Path(first["merged_video"]).is_file()
        # Existing resume logic must avoid every network request.
        with patch.object(descargador_video, "_obtener_info_tikwm",
                          side_effect=AssertionError("Resume requested network")):
            resumed = run("txt")
            assert resumed["success"], resumed
        series = run("url", mode="url")
        assert series["success"], series
        assert paths().profile == root / "url" / "data" / "perfil_chrome_tiktok"
        individual = run("single", mode="single")
        assert individual["success"] and individual["chapters_found"] == 1, individual
        # Oversized corrupt files must be re-downloaded, not skipped as valid.
        broken = Path(first["series_dir"]) / "capitulos" / "Capitulo_01.mp4"
        broken.write_bytes(b"x" * (150 * 1024))
        repaired = run("txt")
        assert repaired["success"], repaired
        corrupt_conversion = broken.with_name("Capitulo_01_h264.mp4")
        corrupt_conversion.write_bytes(b"x" * (150 * 1024))
        recovered = run("txt")
        assert recovered["success"], recovered
        with patch.object(descargador_video, "_obtener_info_tikwm", return_value={"success": False}), patch.object(
                descargador_video, "_descargar_con_playwright",
                return_value={"success": False, "error": "fallo TikWM/Playwright simulado"}), patch.object(
                unificador_videos, "unir_videos_serie") as merge:
            failed = run("download_failure")
            assert not failed["success"] and failed["chapters_failed"]
            assert "fallo TikWM/Playwright" in failed["error"]
            merge.assert_not_called()
        with patch.object(unificador_videos, "unir_videos_serie",
                          return_value={"success": False, "error": "fallo FFmpeg simulado"}):
            failed = run("merge_failure")
            assert not failed["success"] and failed["status"] == "MERGE_PENDING"
            assert failed["merged_video"] == ""
        def partial_extract(self):
            self.author = "prueba"
            self.series_title = "Incompleta"
            self.playlist_total = 3
            return self._guardar_resultados(urls)
        with patch.object(extractor_series.TikTokSeriesExtractor, "extraer", partial_extract), patch.object(
                descargador_video, "descargar_desde_archivo") as download:
            failed = run("incomplete_extraction", mode="url")
            assert not failed["success"] and "incompleta" in failed["error"].lower()
            download.assert_not_called()
    # Chapter order is visible in the original manifest and download result file.
    manifest = json.loads((Path(first["series_dir"]) / "info_serie.json").read_text(encoding="utf-8"))
    assert [c["url"] for c in manifest["capitulos"]] == urls
    assert [c["numero"] for c in manifest["capitulos"]] == [1, 2]
(ROOT / "validation_tiktok").mkdir(exist_ok=True)
(ROOT / "validation_tiktok/offline_results.json").write_text(
    json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
print("PASS: TXT, series URL, resume, chapter order, real MP4 merge, download failure, merge failure, incomplete extraction, internal Chrome profile, single video, corrupt-resume repair.")
