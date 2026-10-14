from pathlib import Path
from subprocess import TimeoutExpired
from core.media import _ffmpeg_binary, run_cmd


def validate_video(path):
    if not Path(path).is_file():
        raise RuntimeError("No se encontró el archivo MP4.")
    result = run_cmd([_ffmpeg_binary(), "-v", "error", "-xerror", "-i", str(path),
                      "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                     capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    if result.returncode:
        raise RuntimeError("MP4 inválido o sin audio/video: " + result.stderr[-3000:])


def valid_video(path):
    try:
        validate_video(path)
        return True
    except (OSError, RuntimeError, TimeoutError, TimeoutExpired):
        return False
