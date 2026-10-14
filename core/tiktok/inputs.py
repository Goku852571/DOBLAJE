from pathlib import Path
from urllib.parse import urlparse

def validate_url(value):
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not (
            host == "tiktok.com" or host.endswith(".tiktok.com")):
        raise ValueError("Solo se admiten URLs de TikTok.")
    return value


def read_urls(path):
    path = Path(path)
    if path.suffix.lower() != ".txt":
        raise ValueError("Selecciona un archivo .txt de URLs de TikTok.")
    urls = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            validate_url(value)
            if value not in urls:
                urls.append(value)
    if not urls:
        raise ValueError("El archivo no contiene URLs de TikTok.")
    return urls

