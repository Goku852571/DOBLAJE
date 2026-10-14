"""TikTok intake dialog; the existing local-video workflow stays independent."""
import os
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QHBoxLayout, QLineEdit, QComboBox,
    QPushButton, QFileDialog, QLabel, QTextEdit, QMessageBox, QCheckBox)
from utils.config import config_manager
from gui.tiktok_worker import TikTokWorker
from core.tiktok.inputs import validate_url, read_urls
from core.tiktok.runtime import relative_path, ROOT


class TikTokDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Descargar videos de TikTok y doblar")
        self.resize(800, 600)
        self.worker = None
        self.result = None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.output = QLineEdit(config_manager.get("tiktok_download_dir", "data/descargas"))
        self.profile = QLineEdit(config_manager.get("tiktok_profile_dir", "data/perfil_chrome_tiktok"))
        self.browser = QComboBox()
        self.browser.addItem("Google Chrome instalado", "chrome")
        self.browser.addItem("Chromium de Playwright", "chromium")
        self.browser.setCurrentIndex(max(0, self.browser.findData(config_manager.get("tiktok_browser", "chrome"))))
        self.mode = QComboBox()
        self.mode.addItem("URL inicial de una serie", "url")
        self.mode.addItem("Archivo .txt de URLs", "txt")
        self.mode.addItem("Video individual de TikTok", "single")
        self.source = QLineEdit()
        self.source.setPlaceholderText("URL del primer capítulo de TikTok")
        self.browse_source = QPushButton("Seleccionar .txt")
        self.browse_source.clicked.connect(self.choose_txt)
        self.browse_source.setEnabled(False)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        form.addRow("Descargas (ruta relativa):", self.output)
        form.addRow("Perfil (ruta relativa):", self.profile)
        form.addRow("Navegador:", self.browser)
        form.addRow("Origen:", self.mode)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source)
        source_row.addWidget(self.browse_source)
        form.addRow("URL / archivo:", source_row)
        layout.addLayout(form)
        notice = QLabel("Las descargas y el perfil pertenecen a Video_Sub. "
                        "Inicia sesión una vez con el perfil interno. "
                        "Repite el mismo origen para reanudar.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.login = QPushButton("Iniciar sesión en TikTok")
        self.login.clicked.connect(lambda: self.begin(login=True))
        layout.addWidget(self.login)
        self.auto_dub = QCheckBox("Continuar automáticamente con análisis, voces y video final")
        self.auto_dub.setChecked(True)
        layout.addWidget(self.auto_dub)
        self.status = QLabel("Preparado")
        layout.addWidget(self.status)
        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        layout.addWidget(self.logs)
        self.video = QLineEdit()
        self.video.setReadOnly(True)
        self.video.setPlaceholderText("Aquí aparecerá la ruta del MP4 validado")
        layout.addWidget(self.video)
        controls = QHBoxLayout()
        self.start = QPushButton("Extraer / descargar, reanudar y unir")
        self.start.clicked.connect(self.begin)
        self.cancel = QPushButton("Cancelar proceso")
        self.cancel.clicked.connect(self.cancel_process)
        self.cancel.setEnabled(False)
        self.proceed = QPushButton("Continuar con análisis y traducción")
        self.proceed.setEnabled(False)
        self.proceed.clicked.connect(self.accept)
        controls.addWidget(self.start)
        controls.addWidget(self.cancel)
        controls.addWidget(self.proceed)
        layout.addLayout(controls)

    def mode_changed(self):
        self.source.clear()
        self.browse_source.setEnabled(self.mode.currentData() == "txt")
        self.source.setPlaceholderText("Archivo .txt" if self.mode.currentData() == "txt"
                                       else "URL del primer capítulo de TikTok")

    def choose_txt(self):
        path, _ = QFileDialog.getOpenFileName(self, "URLs de TikTok", "", "Texto (*.txt)")
        if path:
            self.source.setText(path)

    def begin(self, checked=False, login=False):
        if self.worker and self.worker.isRunning():
            return
        try:
            source = self.source.text().strip()
            if login:
                pass
            elif self.mode.currentData() in ("url", "single"):
                validate_url(source)
            else:
                read_urls(source)
            relative_path(ROOT, self.output.text().strip())
            relative_path(ROOT, self.profile.text().strip())
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Configuración TikTok", str(exc))
            return
        settings = dict(tiktok_download_dir=self.output.text().strip(),
                        tiktok_profile_dir=self.profile.text().strip(),
                        tiktok_temp_dir=config_manager.get("tiktok_temp_dir", "data/temp"),
                        tiktok_browser=self.browser.currentData())
        for key, value in settings.items():
            config_manager.set(key, value)
        self.result = None
        for control in (self.output, self.profile, self.browser, self.mode, self.source, self.browse_source, self.login, self.auto_dub):
            control.setEnabled(False)
        self.video.clear()
        self.proceed.setEnabled(False)
        self.start.setEnabled(False)
        self.cancel.setEnabled(True)
        self.status.setText("Iniciando descargador...")
        self.worker = TikTokWorker(dict(settings=settings,
                                       mode="login" if login else self.mode.currentData(), source=source))
        self.worker.finished.connect(self.worker_stopped)
        self.worker.log.connect(self.logs.append)
        self.worker.stage.connect(self.status.setText)
        self.worker.completed.connect(self.completed)
        self.worker.start()

    def cancel_process(self):
        if self.worker and self.worker.isRunning():
            self.status.setText("Cancelando extracción, descarga o unión...")
            self.worker.cancel()
            self.cancel.setEnabled(False)

    def completed(self, result):
        self.result = dict(result, auto_dub=self.auto_dub.isChecked())
        self.start.setEnabled(True)
        for control in (self.output, self.profile, self.browser, self.mode, self.source, self.login, self.auto_dub):
            control.setEnabled(True)
        self.browse_source.setEnabled(self.mode.currentData() == "txt")
        self.cancel.setEnabled(False)
        if result.get("status") == "SESSION_READY":
            self.status.setText("Sesión guardada en el perfil interno. Ya puedes descargar.")
            return
        if result.get("success") and result.get("merged_video"):
            self.video.setText(result["merged_video"])
            self.status.setText("MP4 validado: %s/%s capítulos. Revisa la ruta y continúa cuando estés listo." % (
                result["chapters_downloaded"], result["chapters_found"]))
            self.proceed.setEnabled(True)
            if not self.worker or not self.worker.isRunning():
                self.worker_stopped()
        else:
            self.status.setText(result.get("status", "ERROR"))
            self.logs.append(result.get("error", "Error desconocido."))
            QMessageBox.warning(self, "TikTok: proceso sin completar",
                                result.get("error", "No hay un MP4 completo para analizar."))

    def worker_stopped(self):
        if self.result and self.result.get("success") and self.result.get("merged_video") and self.result.get("auto_dub"):
            self.accept()

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.cancel_process()
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.cancel_process()
            event.ignore()
        else:
            event.accept()

    def done(self, code):
        if self.worker and self.worker.isRunning():
            self.cancel_process()
            return
        super().done(code)
