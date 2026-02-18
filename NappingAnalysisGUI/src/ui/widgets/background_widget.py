from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import QWidget

class BackgroundWidget(QWidget):
    def __init__(self, parent=None, bg_path: str | None = None):
        super().__init__(parent)
        self._pix = QPixmap()
        if bg_path:
            self.set_background(bg_path)

    def set_background(self, bg_path: str):
        self._pix = QPixmap(bg_path)
        if self._pix.isNull():
            print("[BG] ERROR: cannot load:", bg_path)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pix.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        scaled = self._pix.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
