import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_FILE = str(Path(__file__).resolve().parents[1] / "config.json")

DEFAULT_CONFIG = {
 'tiktok_download_dir': 'data/descargas',
 'tiktok_profile_dir': 'data/perfil_chrome_tiktok',
 'tiktok_temp_dir': 'data/temp',
 'tiktok_browser': 'chrome',
 'tts_voice_name': 'Jorge',
 'tts_engine': 'sapi',
 'openspeaker_voice_id': '',
 'openspeaker_speed': 1.0,
 'asr_engine': 'whisper_cpp',
 'whisper_cpp_executable': 'tools/whisper_cpp/whisper-cli.exe',
 'whisper_cpp_model_path': 'models/whisper_cpp/ggml-large-v3-turbo.bin',
 'whisper_cpp_vad_path': 'tools/whisper_cpp/ggml-silero-v6.2.0.bin',
 'whisper_cpp_threads': 4,
 'asr_model': 'large-v3-turbo',
 'asr_model_path': 'models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo',
 'asr_device': 'cpu',
 'asr_compute_type': 'default',
 'asr_language': 'auto',
 'translation_mode': 'automatic',
 'asr_pause_threshold': 0.35,
 'translation_model': 'demonbyron/HY-MT1.5-1.8B:bf16',
 'translation_device': 'auto',
 'translation_offline': False,
 'translation_target': 'español latino neutro',
 'translation_max_new_tokens': 512,
 'ollama_model': 'qwen2.5:7b',
 'original_audio_volume': 0.5,
 'remove_original_voice': False,
 'separation_model': 'UVR-MDX-NET-Inst_HQ_4',
 'tts_volume': 1.0}

class ConfigManager:
    def __init__(self, config_path: str = CONFIG_FILE):
        self.config_path = config_path
        self._config: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            self._save(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            
            # Merge with default config to ensure all keys exist
            config = DEFAULT_CONFIG.copy()
            for key in ("tiktok_project", "tiktok_python", "tiktok_output"):
                loaded.pop(key, None)
            config.update(loaded)
            return config
        except Exception:
            return DEFAULT_CONFIG.copy()

    def _save(self, config: Dict[str, Any]):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        self._config[key] = value
        self._save(self._config)

config_manager = ConfigManager()
