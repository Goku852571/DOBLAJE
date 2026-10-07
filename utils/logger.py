import logging
import sys
from PySide6.QtCore import QObject, Signal

class LoggerSignals(QObject):
    log_message = Signal(str, str) # level, message

class GuiLogger(logging.Handler):
    def __init__(self, signals: LoggerSignals):
        super().__init__()
        self.signals = signals

    def emit(self, record):
        msg = self.format(record)
        self.signals.log_message.emit(record.levelname, msg)

def setup_logger():
    logger = logging.getLogger("VideoDubbing")
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger, None

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    signals = LoggerSignals()
    gui_handler = GuiLogger(signals)
    gui_handler.setLevel(logging.INFO)
    gui_handler.setFormatter(formatter)
    logger.addHandler(gui_handler)

    return logger, signals

logger, logger_signals = setup_logger()
