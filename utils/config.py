import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_FILE = str(Path(__file__).resolve().parents[1] / "config.json")

DEFAULT_CONFIG = {
 'tts_voice_name': 'Jorge',
 'ocr_enabled': False,
 'ocr_sample_rate': 0.25,
 'ocr_confidence_threshold': 0.4,
 'ocr_languages': ['ch_sim', 'en'],
 'ocr_roi_bottom': 0.65,
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
