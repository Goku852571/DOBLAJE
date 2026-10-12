import json
import hashlib
import os
import tempfile
from typing import List, Dict
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, 
    QProgressBar, QTextEdit, QHeaderView, QMessageBox, QGroupBox, QFormLayout,
    QComboBox, QDoubleSpinBox, QApplication, QInputDialog, QCheckBox,
    QDialog, QPlainTextEdit, QDialogButtonBox
)
from PySide6.QtCore import Qt
from utils.logger import logger, logger_signals
from utils.config import config_manager
from core.media import clear_segment_tts
from gui.workers import AnalyzeWorker, TTSWorker, RenderWorker, TestConnectionWorker, RetryTranslationWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Dubbing Studio")
        self.resize(1000, 700)
        
        self.video_path = ""
        self.segments: List[Dict] = []
        self.temp_dir = tempfile.mkdtemp(prefix="videodub_")
        
        self.init_ui()
        self.init_logger()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 1. Top Section: Video Selection & Config
        top_layout = QHBoxLayout()
        
        # Video Selector
        video_group = QGroupBox("Video Source")
        video_layout = QVBoxLayout(video_group)
        self.lbl_video_path = QLabel("Ningún video seleccionado")
        btn_select_video = QPushButton("Seleccionar Video")
        btn_select_video.clicked.connect(self.select_video)
        video_layout.addWidget(self.lbl_video_path)
        video_layout.addWidget(btn_select_video)
        
        # Config
        config_group = QGroupBox("Generación de voces")
        config_layout = QFormLayout(config_group)
        self.txt_voice_name = QLineEdit(config_manager.get("tts_voice_name", "Jorge"))
        
        self.tts_engine = QComboBox()
        self.tts_engine.addItem("Local (SAPI5)", "sapi")
        self.tts_engine.addItem("OpenSpeaker API", "openspeaker")
        self.tts_engine.setCurrentIndex(max(0, self.tts_engine.findData(config_manager.get("tts_engine", "sapi"))))
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("Clave de sesión o variable OPENSPEAKER_API_KEY")
        self.api_voice = QLineEdit(config_manager.get("openspeaker_voice_id", ""))
        self.api_voice.setPlaceholderText("voice_id de /v3/voices (incluye prefijo)")
        self.api_speed = QDoubleSpinBox()
        self.api_speed.setRange(0.5, 1.5)
        self.api_speed.setSingleStep(0.05)
        self.api_speed.setValue(config_manager.get("openspeaker_speed", 1.0))
        config_layout.addRow("Motor de voz:", self.tts_engine)
        config_layout.addRow("Clave OpenSpeaker:", self.api_key)
        config_layout.addRow("ID de voz API:", self.api_voice)
        config_layout.addRow("Velocidad API:", self.api_speed)
        btn_save_config = QPushButton("Guardar Configuración")
        btn_save_config.clicked.connect(self.save_config)
        
        config_layout.addRow("Nombre de Voz Local:", self.txt_voice_name)
        config_layout.addRow("", btn_save_config)

        top_layout.addWidget(video_group)
        top_layout.addWidget(config_group)
        layout.addLayout(top_layout)

        recognition_group = QGroupBox("Reconocimiento de audio y traducción")
        settings = QFormLayout(recognition_group)
        self.translation_mode = QComboBox()
        self.translation_mode.addItem("Automática (Ollama)", "automatic")
        self.translation_mode.addItem("Manual (JSON)", "manual")
        self.translation_mode.setCurrentIndex(max(0, self.translation_mode.findData(config_manager.get("translation_mode", "automatic"))))
        settings.addRow("Traducción:", self.translation_mode)
        self.asr_engine = QComboBox()
        self.asr_engine.addItem("Whisper CPP (CPU, local)", "whisper_cpp")
        self.asr_engine.addItem("Faster Whisper (local)", "faster_whisper")
        self.asr_engine.setCurrentIndex(max(0, self.asr_engine.findData(config_manager.get("asr_engine", "whisper_cpp"))))
        self.asr_language = QComboBox()
        for label, code in [("Automático (detectar idioma)", "auto"), ("Chino", "zh"), ("Inglés", "en")]:
            self.asr_language.addItem(label, code)
        language = config_manager.get("asr_language", "auto")
        if self.asr_language.findData(language) < 0:
            self.asr_language.addItem(language, language)
        self.asr_language.setCurrentIndex(self.asr_language.findData(language))
        self.spin_original_volume = QDoubleSpinBox()
        self.spin_original_volume.setRange(0.0, 2.0)
        self.spin_original_volume.setSingleStep(0.05)
        self.spin_original_volume.setValue(config_manager.get("original_audio_volume", 0.5))
        self.chk_remove_original_voice = QCheckBox("Eliminar voz original (conservar música y efectos)")
        self.chk_remove_original_voice.setToolTip("Separación local de voz. Conserva el nivel del fondo; controla los picos solo del doblaje.")
        self.chk_remove_original_voice.setChecked(config_manager.get("remove_original_voice", False))
        self.chk_remove_original_voice.toggled.connect(self.spin_original_volume.setDisabled)
        self.spin_original_volume.setDisabled(self.chk_remove_original_voice.isChecked())
        self.spin_tts_volume = QDoubleSpinBox()
        self.spin_tts_volume.setRange(0.0, 2.0)
        self.spin_tts_volume.setSingleStep(0.05)
        self.spin_tts_volume.setValue(config_manager.get("tts_volume", 1.0))
        settings.addRow("Motor de voz a texto:", self.asr_engine)
        settings.addRow("Idioma del audio (ASR):", self.asr_language)
        settings.addRow("Volumen audio original:", self.spin_original_volume)
        settings.addRow("", self.chk_remove_original_voice)
        settings.addRow("Volumen voces generadas:", self.spin_tts_volume)
        
        # Botón de prueba de conexión
        self.btn_test_connection = QPushButton("Probar Conexión Ollama")
        self.btn_test_connection.clicked.connect(self.test_ollama_connection)
        settings.addRow("", self.btn_test_connection)
        
        layout.addWidget(recognition_group)

        # 2. Controls
        controls_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("1. Analizar audio (ASR)")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        
        self.btn_retry_translation = QPushButton("Reintentar Traducción")
        self.btn_retry_translation.clicked.connect(self.retry_translation)
        self.btn_retry_translation.setEnabled(False)
        
        self.btn_tts = QPushButton("2. Generar Voces")
        self.btn_tts.clicked.connect(self.start_tts)
        self.btn_tts.setEnabled(False)
        
        self.btn_render = QPushButton("3. Crear Video Final")
        self.btn_render.clicked.connect(self.start_render)
        self.btn_render.setEnabled(False)
        
        controls_layout.addWidget(self.btn_analyze)
        controls_layout.addWidget(self.btn_retry_translation)
        controls_layout.addWidget(self.btn_tts)
        controls_layout.addWidget(self.btn_render)
        layout.addLayout(controls_layout)

        json_layout = QHBoxLayout()
        self.btn_copy_script_json = QPushButton("Copiar JSON del Guion")
        self.btn_copy_script_json.clicked.connect(self.copy_script_json)
        self.btn_copy_script_json.setEnabled(False)

        self.btn_import_translation_json = QPushButton("Importar JSON Traducido")
        self.btn_import_translation_json.clicked.connect(self.import_translation_json)
        self.btn_import_translation_json.setEnabled(False)
        json_layout.addWidget(self.btn_copy_script_json)
        json_layout.addWidget(self.btn_import_translation_json)
        layout.addLayout(json_layout)

        # 3. Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Inicio (s)", "Fin (s)", "Texto ASR", "Texto Final (Editable)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemChanged.connect(self.on_table_item_changed)
        layout.addWidget(self.table)

        # 4. Progress and Logs
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

    def init_logger(self):
        logger_signals.log_message.connect(self.append_log)

    def append_log(self, level: str, message: str):
        self.log_area.append(message)
        # Scroll to bottom
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def save_recognition_config(self):
        config_manager.set("translation_mode", self.translation_mode.currentData())
        config_manager.set("asr_engine", self.asr_engine.currentData())
        config_manager.set("asr_language", self.asr_language.currentData())
        config_manager.set("original_audio_volume", self.spin_original_volume.value())
        config_manager.set("remove_original_voice", self.chk_remove_original_voice.isChecked())
        config_manager.set("tts_volume", self.spin_tts_volume.value())

    def save_config(self):
        self.save_recognition_config()
        self.save_tts_config()
        config_manager.set("tts_voice_name", self.txt_voice_name.text().strip() or "Jorge")
        QMessageBox.information(self, "Información", "Configuración guardada exitosamente.")

    def save_tts_config(self):
        config_manager.set("tts_engine", self.tts_engine.currentData())
        config_manager.set("openspeaker_voice_id", self.api_voice.text().strip())
        config_manager.set("openspeaker_speed", self.api_speed.value())
        config_manager.set("tts_voice_name", self.txt_voice_name.text().strip() or "Jorge")

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if file_name:
            self.video_path = file_name
            self.lbl_video_path.setText(os.path.basename(file_name))
            self.btn_analyze.setEnabled(True)
            self.table.setRowCount(0)
            self.segments = []
            self.btn_retry_translation.setEnabled(False)
            self.btn_tts.setEnabled(False)
            self.btn_render.setEnabled(False)
            self.btn_copy_script_json.setEnabled(False)
            self.btn_import_translation_json.setEnabled(False)

    def test_ollama_connection(self):
        """Prueba la conexión con Ollama antes de iniciar el análisis."""
        from core.ollama_translator import OllamaTranslator
        
        host = config_manager.get("ollama_host", "http://localhost:11434")
        translator = OllamaTranslator(host=host)
        
        self.btn_test_connection.setEnabled(False)
        self.btn_test_connection.setText("Probando...")
        
        # Ejecutar en segundo plano para no bloquear la GUI
        self.test_worker = TestConnectionWorker(translator)
        self.test_worker.log.connect(lambda msg: logger.info(msg))
        self.test_worker.finished_connection.connect(self.on_test_connection_finished)
        self.test_worker.start()

    def on_test_connection_finished(self, result: dict):
        self.btn_test_connection.setEnabled(True)
        self.btn_test_connection.setText("Probar Conexión Ollama")
        
        if result.get("ok"):
            QMessageBox.information(
                self, "Conexión Exitosa",
                f"{result.get('message')}\n\nModelos disponibles:\n" +
                "\n".join(f"  • {m}" for m in result.get("models", []))
            )
        else:
            QMessageBox.critical(
                self, "Error de Conexión",
                f"No se pudo conectar con Ollama:\n\n{result.get('message')}"
            )

    def start_analysis(self):
        if not self.video_path:
            return
        
        self.save_recognition_config()
        manual = config_manager.get("translation_mode", "automatic") == "manual"
        self.btn_analyze.setEnabled(False)
        if not manual:
            from core.ollama_translator import OllamaTranslator
            translator = OllamaTranslator(host=config_manager.get("ollama_host", "http://localhost:11434"))
            if not translator.is_available():
                self.btn_analyze.setEnabled(True)
                QMessageBox.critical(self, "Sin Conexión", "Ollama no está disponible. Seleccione traducción manual para trabajar con JSON.")
                return
        
        self.btn_tts.setEnabled(False)
        self.btn_retry_translation.setEnabled(False)
        self.btn_render.setEnabled(False)
        self.btn_copy_script_json.setEnabled(False)
        self.btn_import_translation_json.setEnabled(False)
        self.progress_bar.setValue(0)
        self.table.setRowCount(0)
        
        self.save_recognition_config()
        self.segments = []
        self.worker = AnalyzeWorker(self.video_path, manual_translation=manual)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(lambda msg: logger.info(msg))
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.start()

    def retry_translation(self):
        """Reintenta traducir todos los segmentos con texto original."""
        if config_manager.get("translation_mode", "automatic") == "manual" or any(
            s.get("translation_source") == "manual_json" for s in self.segments
        ):
            QMessageBox.information(self, "Traducción Manual", "Se usará su traducción importada. Puede importar otro JSON para cambiarla.")
            return
        if not self.segments:
            QMessageBox.warning(self, "Advertencia", "No hay segmentos para traducir.")
            return
        
        # Verificar que hay segmentos con texto original
        has_source = any(
            s.get('source_text', '').strip() or s.get('asr_text', '').strip()
            for s in self.segments
        )
        if not has_source:
            QMessageBox.warning(
                self, "Advertencia",
                "No se encontró texto original en los segmentos."
            )
            return
        
        self.btn_retry_translation.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.btn_tts.setEnabled(False)
        self.btn_render.setEnabled(False)
        self.btn_import_translation_json.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = RetryTranslationWorker(
            self.segments,
            host=config_manager.get("ollama_host", "http://localhost:11434"),
            target_language=config_manager.get("translation_target", "español")
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(lambda msg: logger.info(msg))
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_retry_translation_finished)
        self.worker.start()

    def on_retry_translation_finished(self, segments: List[Dict]):
        self.segments = segments
        self.populate_table()
        self.btn_analyze.setEnabled(True)
        self.btn_retry_translation.setEnabled(True)
        self.btn_tts.setEnabled(True)
        
        translated_count = sum(1 for s in segments if s.get('final_text', '').strip())
        QMessageBox.information(
            self, "Traducción Completada",
            f"Traducción exitosa: {translated_count}/{len(segments)} segmentos traducidos."
        )

    def on_analysis_finished(self, segments: List[Dict]):
        self.segments = sorted(segments, key=lambda s: (s['start'], s['end']))
        self.populate_table()
        self.btn_analyze.setEnabled(True)
        if segments:
            if self.worker.manual_translation:
                self.btn_retry_translation.setEnabled(False)
                self.btn_tts.setEnabled(False)
                QMessageBox.information(self, "Guion para traducción manual",
                    "Copie el JSON del guion al portapapeles con el botón correspondiente. Complete cada text de translated_script y use Importar JSON Traducido. Se conservarán los tiempos originales.")
                self.copy_script_json()
                return
            self.btn_retry_translation.setEnabled(True)
            self.btn_tts.setEnabled(True)
            self.btn_copy_script_json.setEnabled(True)
            self.btn_import_translation_json.setEnabled(True)
            # Contar cuántos segmentos tienen traducción
            translated_count = sum(1 for s in segments if s.get('final_text', '').strip())
            QMessageBox.information(
                self, "Análisis Completado",
                f"Análisis finalizado: {len(segments)} segmentos.\n"
                f"Traducidos: {translated_count}/{len(segments)}.\n"
                f"Revise la columna 'Texto Final' y use 'Reintentar Traducción' si es necesario."
            )
        else:
            QMessageBox.warning(self, "Advertencia", "No se encontraron segmentos para traducir.")

    def populate_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.segments))
        for i, seg in enumerate(self.segments):
            item_start = QTableWidgetItem(f"{seg['start']:.2f}")
            item_start.setFlags(item_start.flags() & ~Qt.ItemIsEditable)
            
            item_end = QTableWidgetItem(f"{seg['end']:.2f}")
            item_end.setFlags(item_end.flags() & ~Qt.ItemIsEditable)
            
            
            item_asr = QTableWidgetItem(seg.get('asr_text', ''))
            item_asr.setFlags(item_asr.flags() & ~Qt.ItemIsEditable)
            
            item_final = QTableWidgetItem(seg.get('final_text', ''))
            
            self.table.setItem(i, 0, item_start)
            self.table.setItem(i, 1, item_end)
            self.table.setItem(i, 2, item_asr)
            self.table.setItem(i, 3, item_final)
            
        self.table.blockSignals(False)
        has_segments = bool(self.segments)
        self.btn_copy_script_json.setEnabled(has_segments)
        self.btn_import_translation_json.setEnabled(has_segments)

    def on_table_item_changed(self, item):
        if item.column() == 3:
            row = item.row()
            if row < len(self.segments):
                self.segments[row]['final_text'] = item.text()
                clear_segment_tts(self.segments[row])
                self.btn_render.setEnabled(False)

    def _segment_source_text(self, seg: Dict) -> str:
        return (
            seg.get('source_text', '').strip() or
            seg.get('asr_text', '').strip() or
            seg.get('final_text', '').strip()
        )

    def _script_json_payload(self) -> Dict:
        source_script = []
        translated_template = []
        for index, seg in enumerate(sorted(self.segments, key=lambda s: (s['start'], s['end'])), start=1):
            duration = max(0.0, float(seg.get('end', 0.0)) - float(seg.get('start', 0.0)))
            source_text = self._segment_source_text(seg)
            item = {
                "id": index,
                "start": round(float(seg.get('start', 0.0)), 3),
                "end": round(float(seg.get('end', 0.0)), 3),
                "duration": round(duration, 3),
                "source_text": source_text,
            }
            source_script.append(item)
            translated_template.append({
                "id": index,
                "text": "",
            })
        return {
            "schema_version": 1,
            "script_id": hashlib.sha256(json.dumps(source_script, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest(),
            "instructions": (
                "Traduce y adapta source_script al español latino neutro para doblaje profesional. "
                "Usa el contexto completo del guion, conserva intención y continuidad, y escribe frases "
                "naturales para voz hablada. Respeta cada id. Mantén las frases breves según duration. "
                "Devuelve solo JSON válido con la clave translated_script."
            ),
            "source_script": source_script,
            "translated_script": translated_template,
        }

    def copy_script_json(self):
        if not self.segments:
            QMessageBox.warning(self, "Advertencia", "No hay guion para copiar.")
            return
        payload = self._script_json_payload()
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        QApplication.clipboard().setText(text)
        logger.info("JSON del guion copiado al portapapeles.")
        QMessageBox.information(
            self,
            "JSON Copiado",
            "Se copió al portapapeles el guion sin traducir y la plantilla del JSON traducido."
        )

    def import_translation_json(self):
        if not self.segments:
            QMessageBox.warning(self, "Advertencia", "No hay segmentos donde importar traducción.")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Pegar JSON Traducido")
        dialog.resize(600, 400)
        layout = QVBoxLayout(dialog)
        
        lbl = QLabel("Pegue el contenido del JSON traducido aquí:")
        layout.addWidget(lbl)
        
        text_edit = QPlainTextEdit()
        layout.addWidget(text_edit)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        button_box.button(QDialogButtonBox.Ok).setText("Importar")
        layout.addWidget(button_box)
        
        if dialog.exec() != QDialog.Accepted:
            return
            
        text = text_edit.toPlainText().strip()
        if not text:
            return
            
        try:
            translations = self._parse_translation_json(text)
            applied = self._apply_translation_items(translations)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "JSON Inválido", str(exc))
            return
        self.populate_table()
        if applied:
            self.btn_tts.setEnabled(True)
            self.btn_render.setEnabled(False)
            self.btn_retry_translation.setEnabled(False)
        logger.info("Traducciones importadas desde JSON: %d segmentos.", applied)
        QMessageBox.information(
            self,
            "Traducción Importada",
            f"Se importaron {applied}/{len(self.segments)} segmentos traducidos."
        )

    def _parse_translation_json(self, text: str) -> List[Dict]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"No es JSON válido: {exc}") from exc

        if isinstance(payload, dict):
            if payload.get("script_id") and payload["script_id"] != self._script_json_payload()["script_id"]:
                raise ValueError("El JSON pertenece a otro guion. Exporte el guion actual y traduzca ese archivo.")
            items = payload.get("translated_script") or payload.get("translations") or payload.get("segments")
        elif isinstance(payload, list):
            items = payload
        else:
            items = None

        if not isinstance(items, list):
            raise ValueError(
                "El JSON debe contener una lista en 'translated_script', "
                "o ser directamente una lista de objetos con id y text."
            )
        return items

    def _apply_translation_items(self, items: List[Dict]) -> int:
        by_id = {}
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Cada traducción debe ser un objeto con id y text.")
            raw_id = item.get("id")
            if isinstance(raw_id, bool) or not (isinstance(raw_id, int) or isinstance(raw_id, str) and raw_id.strip().isdigit()):
                raise ValueError("Cada id debe ser un número entero.")
            item_id = int(raw_id)
            if item_id in by_id:
                raise ValueError(f"Id duplicado: {item_id}.")
            text = item.get("text", item.get("translated_text", ""))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Falta la traducción del id {item_id}.")
            text = text.strip()
            by_id[item_id] = text

        if not by_id:
            raise ValueError("No se encontraron entradas traducidas con id y text.")
        expected = set(range(1, len(self.segments) + 1))
        if set(by_id) != expected:
            raise ValueError(f"Los id deben coincidir con el guion completo. Faltantes: {sorted(expected - set(by_id))}. Desconocidos: {sorted(set(by_id) - expected)}.")

        applied = 0
        for index, seg in enumerate(sorted(self.segments, key=lambda s: (s['start'], s['end'])), start=1):
            if index in by_id and by_id[index]:
                seg["final_text"] = by_id[index]
                seg["translation_source"] = "manual_json"
                clear_segment_tts(seg)
                applied += 1
        return applied

    def start_tts(self):
        if not self.segments:
            return
        if config_manager.get("translation_mode", "automatic") == "manual" and any(not s.get('final_text', '').strip() for s in self.segments):
            QMessageBox.warning(self, "Traducción pendiente", "Importe el JSON con todos los segmentos traducidos antes de generar las voces.")
            return

        self.btn_analyze.setEnabled(False)
        self.btn_retry_translation.setEnabled(False)
        self.btn_tts.setEnabled(False)
        self.btn_render.setEnabled(False)
        self.btn_import_translation_json.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.save_tts_config()
        self.worker = TTSWorker(self.segments, self.temp_dir, api_key=self.api_key.text().strip())
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(lambda msg: logger.info(msg))
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_tts_finished)
        self.worker.start()

    def on_tts_finished(self, segments: List[Dict]):
        self.segments = segments
        self.btn_analyze.setEnabled(True)
        self.btn_retry_translation.setEnabled(self._can_auto_translate())
        self.btn_import_translation_json.setEnabled(True)
        self.btn_tts.setEnabled(True)
        self.btn_render.setEnabled(True)
        QMessageBox.information(self, "Voces Generadas", "Voces generadas correctamente. Puede crear el video final.")

    def start_render(self):
        self.save_recognition_config()
        output_path, _ = QFileDialog.getSaveFileName(self, "Guardar Video Final", "", "MP4 Video (*.mp4)")
        if not output_path:
            return

        self.btn_analyze.setEnabled(False)
        self.btn_retry_translation.setEnabled(False)
        self.btn_tts.setEnabled(False)
        self.btn_render.setEnabled(False)
        self.btn_import_translation_json.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = RenderWorker(self.video_path, self.segments, self.temp_dir, output_path)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(lambda msg: logger.info(msg))
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_render_finished)
        self.worker.start()

    def on_render_finished(self, output_path: str):
        self.btn_analyze.setEnabled(True)
        self.btn_retry_translation.setEnabled(self._can_auto_translate())
        self.btn_import_translation_json.setEnabled(bool(self.segments))
        self.btn_tts.setEnabled(True)
        self.btn_render.setEnabled(True)
        QMessageBox.information(self, "Video Creado", f"Video final guardado en:\n{output_path}")

    def on_worker_error(self, error_msg: str):
        self.btn_analyze.setEnabled(True)
        self.btn_retry_translation.setEnabled(self._can_auto_translate())
        self.btn_import_translation_json.setEnabled(bool(self.segments))
        if self.segments:
            self.btn_tts.setEnabled(True)
            self.btn_render.setEnabled(
                any(s.get('tts_audio') for s in self.segments) and
                all(s.get('tts_audio') for s in self.segments if s.get('final_text', '').strip())
            )
        else:
            self.btn_tts.setEnabled(False)
            self.btn_render.setEnabled(False)
                
        QMessageBox.critical(self, "Error", error_msg)
        logger.error(error_msg)

    def _can_auto_translate(self):
        return bool(self.segments) and config_manager.get("translation_mode", "automatic") != "manual" and not any(
            s.get("translation_source") == "manual_json" for s in self.segments
        )
