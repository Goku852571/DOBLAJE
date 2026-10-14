import json
import hashlib
import os
import tempfile
from typing import List, Dict
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, 
    QProgressBar, QTextEdit, QHeaderView, QMessageBox, QGroupBox, QFormLayout,
    QComboBox, QDoubleSpinBox, QApplication, QInputDialog, QCheckBox,
    QDialog, QPlainTextEdit, QDialogButtonBox
)
from PySide6.QtCore import Qt, QTimer
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
        self._auto_dub_output = ""
        self._pipeline_workers = []
        self.segments: List[Dict] = []
        self.temp_dir = tempfile.mkdtemp(prefix="videodub_")
        
        self.init_ui()
        self.init_logger()
        self._analysis_save_timer = QTimer(self)
        self._analysis_save_timer.setSingleShot(True)
        self._analysis_save_timer.setInterval(1000)
        self._analysis_save_timer.timeout.connect(self._auto_save_analysis)

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
        self.btn_tiktok = QPushButton("Descargar videos de TikTok y doblar")
        self.btn_tiktok.clicked.connect(self.download_tiktok)
        video_layout.addWidget(self.btn_tiktok)
        
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
        self.btn_tts.setToolTip("Conserva las voces válidas y reintenta los segmentos pendientes o modificados.")
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
        analysis_controls = QHBoxLayout()
        self.btn_save_analysis = QPushButton("Guardar análisis")
        self.btn_load_analysis = QPushButton("Cargar análisis")
        self.btn_save_analysis.setEnabled(False)
        self.btn_load_analysis.setEnabled(False)
        self.btn_save_analysis.clicked.connect(self.save_analysis_file)
        self.btn_load_analysis.clicked.connect(self.load_analysis_file)
        analysis_controls.addWidget(self.btn_save_analysis)
        analysis_controls.addWidget(self.btn_load_analysis)
        self.btn_load_tts_progress = QPushButton("Cargar progreso de voces")
        self.btn_load_tts_progress.clicked.connect(self.load_tts_progress_file)
        analysis_controls.addWidget(self.btn_load_tts_progress)
        layout.addLayout(analysis_controls)

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
        self.pipeline_status = QLabel("Preparado para seleccionar un video local o descargar TikTok")
        layout.addWidget(self.pipeline_status)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

    def init_logger(self):
        logger_signals.log_message.connect(self.append_log)

    def append_log(self, level: str, message: str):
        self.log_area.append(message)
        lower = message.lower()
        for fragment, stage in (("transcrip", "Análisis ASR"), ("idioma", "Detección de idioma"),
                                ("traduc", "Traducción"), ("generando voz", "Generación de voces"),
                                ("sincroniz", "Sincronización"), ("render", "Renderizado"),
                                ("video final", "Renderizado")):
            if fragment in lower:
                self.pipeline_status.setText(stage)
                break
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

    def download_tiktok(self):
        if getattr(self, "worker", None) and self.worker.isRunning():
            QMessageBox.information(self, "Proceso en curso", "Espera a que termine el proceso actual.")
            return
        from gui.tiktok_dialog import TikTokDialog
        dialog = TikTokDialog(self)
        if dialog.exec() == QDialog.Accepted:
            result = dialog.result or {}
            if result.get("success") and result.get("merged_video"):
                self.load_video(result["merged_video"])
                if result.get("auto_dub"):
                    self._auto_dub_output = str(Path(self.video_path).parent / "Video_Doblado.mp4")
                logger.info("MP4 TikTok validado listo para análisis: %s", self.video_path)
                self.start_analysis()

    def load_video(self, file_name):
        if hasattr(self, 'btn_save_analysis'):
            self._auto_save_analysis()
        self._auto_dub_output = ""
        self.video_path = file_name
        self.lbl_video_path.setText(os.path.basename(file_name))
        self.lbl_video_path.setToolTip(file_name)
        self.btn_analyze.setEnabled(True)
        self.table.setRowCount(0)
        self.segments = []
        self.btn_retry_translation.setEnabled(False)
        self.btn_tts.setEnabled(False)
        self.btn_render.setEnabled(False)
        self.btn_copy_script_json.setEnabled(False)
        self.btn_import_translation_json.setEnabled(False)

        if hasattr(self, 'btn_save_analysis'):
            self.btn_save_analysis.setEnabled(False)
            self.btn_load_analysis.setEnabled(True)
            from core.analysis_store import cache_path
            if cache_path(file_name).is_file():
                self.pipeline_status.setText("Hay un análisis guardado: pulse Cargar análisis o Analizar audio para reutilizarlo.")

    def select_video(self):
        if self._analysis_busy():
            return
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if file_name:
            self.load_video(file_name)

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
        
        if hasattr(self, 'btn_load_analysis') and self._offer_saved_analysis():
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
        if getattr(self, "worker", None):
            self._pipeline_workers.append(self.worker)
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
            target_language=config_manager.get("translation_target", "español"),
            video_path=self.video_path
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(lambda msg: logger.info(msg))
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_retry_translation_finished)
        self.worker.start()

    def on_retry_translation_finished(self, segments: List[Dict]):
        self.segments = segments
        if hasattr(self, 'btn_save_analysis'):
            self._auto_save_analysis()
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
        if hasattr(self, 'btn_save_analysis'):
            self._auto_save_analysis()
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
            if getattr(self, "_auto_dub_output", ""):
                if translated_count == len(segments):
                    QTimer.singleShot(0, self.start_tts)
                else:
                    self.on_worker_error("Faltan traducciones. El doblaje automático se detuvo.")
                return
            QMessageBox.information(
                self, "Análisis Completado",
                f"Análisis finalizado: {len(segments)} segmentos.\n"
                f"Traducidos: {translated_count}/{len(segments)}.\n"
                f"Revise la columna 'Texto Final' y use 'Reintentar Traducción' si es necesario."
            )
        else:
            self._auto_dub_output = ""
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
            if seg.get('tts_error'):
                item_final.setToolTip('Voz pendiente: ' + seg['tts_error'])
            
            self.table.setItem(i, 0, item_start)
            self.table.setItem(i, 1, item_end)
            self.table.setItem(i, 2, item_asr)
            self.table.setItem(i, 3, item_final)
            
        self.table.blockSignals(False)
        has_segments = bool(self.segments)
        if hasattr(self, 'btn_save_analysis'):
            self.btn_save_analysis.setEnabled(has_segments)
        self.btn_copy_script_json.setEnabled(has_segments)
        self.btn_import_translation_json.setEnabled(has_segments)

    def on_table_item_changed(self, item):
        if item.column() == 3:
            row = item.row()
            if row < len(self.segments):
                self.segments[row]['final_text'] = item.text()
                clear_segment_tts(self.segments[row])
                self.btn_render.setEnabled(False)
                self._queue_analysis_save()

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
                "Analiza y utiliza nombres latinos en lugar de nombres chinos, manteniendo la consistencia "
                "a lo largo de todo el guion (por ejemplo, personajes de la misma familia deben mantener coherencia "
                "y compartir apellido). Realiza la conversión monetaria correspondiente: cuando hablen de yuanes, "
                "haz la conversión correspondiente a dólares. "
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
        self._queue_analysis_save()
        logger.info("Traducciones importadas desde JSON: %d segmentos.", applied)
        if getattr(self, "_auto_dub_output", "") and applied == len(self.segments):
            QTimer.singleShot(0, self.start_tts)
            return
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
        if hasattr(self, 'btn_save_analysis'):
            self._auto_save_analysis()
        if not self.segments:
            return
        if any(not s.get('final_text', '').strip() for s in self.segments):
            QMessageBox.warning(self, "Traducción pendiente", "Complete las traducciones pendientes con Reintentar Traducción o importe el JSON antes de generar las voces.")
            return

        self.btn_analyze.setEnabled(False)
        self.btn_retry_translation.setEnabled(False)
        self.btn_tts.setEnabled(False)
        self.btn_render.setEnabled(False)
        self.btn_import_translation_json.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.save_tts_config()
        if getattr(self, "worker", None):
            self._pipeline_workers.append(self.worker)
        self.worker = TTSWorker(self.segments, self.temp_dir, api_key=self.api_key.text().strip())
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(lambda msg: logger.info(msg))
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_tts_finished)
        self.worker.start()

    def on_tts_finished(self, segments: List[Dict]):
        self.segments = segments
        if hasattr(self, 'btn_save_analysis'):
            self.populate_table()
            self._auto_save_analysis()
        self.btn_analyze.setEnabled(True)
        self.btn_retry_translation.setEnabled(self._can_auto_translate())
        self.btn_import_translation_json.setEnabled(True)
        self.btn_tts.setEnabled(True)
        self.btn_render.setEnabled(True)
        if getattr(self, "_auto_dub_output", ""):
            QTimer.singleShot(0, self.start_render)
            return
        QMessageBox.information(self, "Voces Generadas", "Voces generadas correctamente. Puede crear el video final.")

    def start_render(self):
        self.save_recognition_config()
        output_path = getattr(self, "_auto_dub_output", "")
        if not output_path:
            output_path, _ = QFileDialog.getSaveFileName(self, "Guardar Video Final", "", "MP4 Video (*.mp4)")
        if not output_path:
            return

        self.btn_analyze.setEnabled(False)
        self.btn_retry_translation.setEnabled(False)
        self.btn_tts.setEnabled(False)
        self.btn_render.setEnabled(False)
        self.btn_import_translation_json.setEnabled(False)
        self.progress_bar.setValue(0)
        
        if getattr(self, "worker", None):
            self._pipeline_workers.append(self.worker)
        self.worker = RenderWorker(self.video_path, self.segments, self.temp_dir, output_path)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(lambda msg: logger.info(msg))
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_render_finished)
        self.worker.start()

    def on_render_finished(self, output_path: str):
        self._auto_dub_output = ""
        self.btn_analyze.setEnabled(True)
        self.btn_retry_translation.setEnabled(self._can_auto_translate())
        self.btn_import_translation_json.setEnabled(bool(self.segments))
        self.btn_tts.setEnabled(True)
        self.btn_render.setEnabled(True)
        QMessageBox.information(self, "Video Creado", f"Video final guardado en:\n{output_path}")

    def on_worker_error(self, error_msg: str):
        self._auto_dub_output = ""
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
                
        if getattr(getattr(self, "worker", None), "tts_engine", None):
            self.populate_table()
        QMessageBox.critical(self, "Error", error_msg)
        logger.error(error_msg)

    def _can_auto_translate(self):
        return bool(self.segments) and config_manager.get("translation_mode", "automatic") != "manual" and not any(
            s.get("translation_source") == "manual_json" for s in self.segments
        )

    def _queue_analysis_save(self):
        timer = getattr(self, '_analysis_save_timer', None)
        if timer is not None:
            timer.start()

    def _auto_save_analysis(self):
        if not self.video_path or not self.segments:
            return
        try:
            from core.analysis_store import save_analysis
            path = save_analysis(self.video_path, self.segments)
            logger.info("Análisis y traducciones guardados: %s", path)
        except (OSError, ValueError, TypeError) as exc:
            logger.error("No se pudo guardar el análisis: %s", exc)

    def _analysis_busy(self):
        worker = getattr(self, 'worker', None)
        if worker is not None and worker.isRunning():
            QMessageBox.information(self, "Proceso en curso", "Espere a que termine el proceso antes de cambiar el video o el análisis.")
            return True
        return False

    def save_analysis_file(self):
        if not self.video_path or not self.segments or self._analysis_busy():
            return
        suggested = str(Path(self.video_path).with_suffix('.analisis.json'))
        filename, _ = QFileDialog.getSaveFileName(self, "Guardar análisis ASR y traducciones", suggested, "Análisis JSON (*.json)")
        if not filename:
            return
        try:
            from core.analysis_store import save_analysis
            save_analysis(self.video_path, self.segments, filename)
            self.pipeline_status.setText("Análisis exportado: " + filename)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "No se pudo guardar el análisis", str(exc))

    def _apply_saved_analysis(self, segments):
        self.segments = segments
        self.populate_table()
        complete = bool(segments) and all(s.get('final_text', '').strip() for s in segments)
        self.btn_analyze.setEnabled(True)
        self.btn_tts.setEnabled(complete)
        self.btn_render.setEnabled(False)
        self.btn_retry_translation.setEnabled(self._can_auto_translate())
        self.progress_bar.setValue(100)
        self.pipeline_status.setText(f"Análisis recuperado: {len(segments)} segmentos. No se ejecutó ASR.")
        logger.info("Análisis recuperado sin ASR: %d segmentos.", len(segments))
        if getattr(self, '_auto_dub_output', '') and complete:
            QTimer.singleShot(0, self.start_tts)

    def load_analysis_file(self):
        if not self.video_path or self._analysis_busy():
            return
        from core.analysis_store import cache_path, load_analysis
        cached = cache_path(self.video_path)
        filename, _ = QFileDialog.getOpenFileName(self, "Cargar análisis del video seleccionado", str(cached if cached.is_file() else Path(self.video_path).parent), "Análisis JSON (*.json)")
        if not filename:
            return
        try:
            segments = load_analysis(self.video_path, filename)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "No se pudo cargar el análisis", str(exc))
            return
        self._apply_saved_analysis(segments)
        self._auto_save_analysis()

    def _offer_saved_analysis(self):
        from core.analysis_store import cache_path, load_analysis
        if not cache_path(self.video_path).is_file():
            return False
        try:
            segments = load_analysis(self.video_path)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("El análisis guardado no se puede reutilizar: %s", exc)
            return False
        answer = QMessageBox.question(self, "Análisis disponible",
            "Este video ya tiene un análisis guardado. ¿Desea reutilizarlo?\n"
            "Sí: cargar sin ejecutar ASR. No: analizar nuevamente. Cancelar: volver.",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
        if answer == QMessageBox.Cancel:
            return True
        if answer == QMessageBox.Yes:
            self._apply_saved_analysis(segments)
            return True
        return False

    def closeEvent(self, event):
        self._auto_save_analysis()
        super().closeEvent(event)

    def load_tts_progress_file(self):
        if self._analysis_busy():
            return
        if not self.segments:
            QMessageBox.information(self, "Cargue el análisis", "Seleccione el video y cargue su análisis ASR antes de recuperar las voces.")
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Cargar tts_progreso.json", "", "Progreso JSON (*.json)")
        if not filename:
            return
        try:
            from core.tts_recovery import load_tts_progress
            segments, valid = load_tts_progress(self.segments, filename)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            QMessageBox.critical(self, "No se pudo recuperar el progreso", str(exc))
            return
        self.segments = segments
        self.populate_table()
        complete = all(s.get('final_text', '').strip() for s in segments)
        required = sum(bool(s.get('final_text', '').strip()) for s in segments)
        self.btn_tts.setEnabled(complete)
        self.btn_render.setEnabled(complete and valid == required)
        self._auto_save_analysis()
        self.pipeline_status.setText(f"Voces recuperadas: {valid}/{required}. Genere solo las pendientes.")
