"""Local voice/background separation using pyvideotrans's UVR and Spleeter paths."""
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "onnx"


def separate_original_voice(input_file: str, vocal_file: str, background_file: str, log=None,
                            model_name="UVR-MDX-NET-Inst_HQ_4"):
    """Write separate stereo WAV stems without downloading anything at runtime."""
    if model_name not in ("spleeter", "UVR-MDX-NET-Inst_HQ_4"):
        raise RuntimeError("Modelo de separación no reconocido: " + str(model_name))
    use_uvr = model_name != "spleeter"
    required = [MODEL_DIR / (model_name + '.onnx')] if use_uvr else [
        MODEL_DIR / "vocals.fp16.onnx", MODEL_DIR / "accompaniment.fp16.onnx"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Faltan los modelos locales para eliminar la voz original: " + ", ".join(missing))
    try:
        import numpy as np
        import sherpa_onnx
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "La separación de voz requiere numpy, sherpa-onnx y soundfile instalados en este entorno. "
            f"Módulo ausente: {exc.name}.") from exc
    except ImportError as exc:
        detail = str(exc)
        blocked = any(term in detail.casefold() for term in (
            "control de aplicaciones", "application control", "code integrity", "blocked", "bloque"))
        if blocked:
            raise RuntimeError(
                "Windows bloqueó una biblioteca DLL de la separación de voz mediante Control de aplicaciones. "
                "Los paquetes están instalados; no es un error de dependencias ausentes. "
                "Solicita al administrador revisar el registro Microsoft-Windows-CodeIntegrity/Operational "
                "y autorizar una distribución compatible y de confianza. "
                f"Detalle de Windows: {detail}") from exc
        raise RuntimeError(
            "No se pudo cargar una biblioteca nativa de la separación de voz. "
            f"Detalle: {detail}") from exc

    # Same configurations and distinct stem orders as pyvideotrans's
    # videotrans/process/_audio_separate.py:vocal_bgm and vocal_bgm_spleeter.
    model_config = dict(num_threads=4, debug=False, provider="cpu")
    if use_uvr:
        model_config['uvr'] = sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(required[0]))
    else:
        model_config['spleeter'] = sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(
            vocals=str(required[0]), accompaniment=str(required[1]))
    config = sherpa_onnx.OfflineSourceSeparationConfig(
        model=sherpa_onnx.OfflineSourceSeparationModelConfig(**model_config)
    )
    if not config.validate():
        raise RuntimeError("La configuración local de separación de voz no es válida.")
    samples, rate = sf.read(input_file, dtype="float32", always_2d=True)
    if not len(samples):
        raise RuntimeError("El audio que se va a separar está vacío.")
    original_duration = len(samples) / rate
    # STFT inverse reconstruction at the exposed file edges can create huge
    # numerical spikes. Reflect context around the file and discard only this
    # added context afterward; the original timeline and level stay unchanged.
    pad_samples = rate
    padded = np.pad(samples, ((pad_samples, pad_samples), (0, 0)),
                    mode="reflect" if len(samples) > 1 else "edge")
    if log:
        log(f"Separando voz y fondo con {model_name} ONNX local; puede tardar varios minutos...")
    separator = sherpa_onnx.OfflineSourceSeparation(config)
    output = separator.process(sample_rate=rate, samples=np.ascontiguousarray(padded.T))
    if len(output.stems) != 2 or output.sample_rate <= 0:
        raise RuntimeError("El separador no devolvió las dos pistas esperadas.")
    expected_samples = round(original_duration * output.sample_rate)
    expected_padded = round(len(padded) / rate * output.sample_rate)
    trim_start = round(pad_samples / rate * output.sample_rate)
    # UVR: accompaniment first. Spleeter: vocals first. Swapping them would
    # accidentally mix the original dialogue back into the final video.
    for index, filename in enumerate((background_file, vocal_file) if use_uvr else (vocal_file, background_file)):
        data = np.asarray(output.stems[index].data).T
        if data.ndim != 2 or not np.isfinite(data).all():
            raise RuntimeError("La pista separada contiene datos de audio inválidos.")
        if abs(len(data) - expected_padded) > round(0.05 * output.sample_rate):
            raise RuntimeError("La separación cambió la duración del audio; se canceló para evitar desincronización.")
        data = data[trim_start:trim_start + expected_samples]
        if len(data) < expected_samples:
            data = np.pad(data, ((0, expected_samples - len(data)), (0, 0)))
        sf.write(filename, data, samplerate=output.sample_rate, subtype="FLOAT")
    if log:
        log("Pistas separadas: voz original y fondo. Se mezclará únicamente el fondo con el doblaje.")
