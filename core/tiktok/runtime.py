"""Project-relative TikTok resources; no resources from another application."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TikTokPaths:
    root: Path
    downloads: Path
    profile: Path
    temp: Path
    logs: Path
    browser: str = "chrome"


_current = None


def relative_path(root, value):
    value = Path(value)
    if value.is_absolute():
        raise ValueError("Usa rutas relativas a Video_Sub.")
    resolved = (root / value).resolve()
    if root not in resolved.parents:
        raise ValueError("La ruta debe quedar dentro de Video_Sub.")
    return resolved


def configure(settings=None, root=None):
    global _current
    settings = settings or {}
    root = Path(root or ROOT).resolve()
    browser = settings.get("tiktok_browser", "chrome")
    if browser not in ("chrome", "chromium"):
        raise ValueError("Navegador TikTok no válido; elige chrome o chromium.")
    _current = TikTokPaths(
        root, relative_path(root, settings.get("tiktok_download_dir", "data/descargas")),
        relative_path(root, settings.get("tiktok_profile_dir", "data/perfil_chrome_tiktok")),
        relative_path(root, settings.get("tiktok_temp_dir", "data/temp")),
        relative_path(root, "logs"), browser)
    for directory in (_current.downloads, _current.profile, _current.temp, _current.logs):
        directory.mkdir(parents=True, exist_ok=True)
    return _current


def paths():
    return _current or configure()


def acquire_pipeline_lock(resources):
    """OS-held lock: a killed child releases it, so cancellation can resume."""
    handle = (resources.logs / "tiktok.lock").open("a+b")
    try:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        import os
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except OSError as exc:
        handle.close()
        raise RuntimeError("Ya hay una descarga o sesión TikTok activa en este proyecto.") from exc
