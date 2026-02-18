from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QPixmap
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink

from src.core.utils.paths import gui_path, asset_path


class MainMenuPage(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setContentsMargins(0, 0, 0, 0)

        # ======================================
        # STACK : SUPERPOSITION
        # ======================================
        self.stack = QtWidgets.QStackedLayout(self)
        self.stack.setStackingMode(QtWidgets.QStackedLayout.StackingMode.StackAll)
        self.stack.setContentsMargins(0, 0, 0, 0)

        # ======================================
        # BACKGROUND VIDEO -> QLabel (pas de native window)
        # ======================================
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: black;")
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        # IMPORTANT : laisse passer les clics
        self.video_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._last_pixmap = None

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.0)
        self.player.setAudioOutput(self.audio_output)

        self.video_sink = QVideoSink(self)
        self.player.setVideoOutput(self.video_sink)

        self.video_sink.videoFrameChanged.connect(self.on_video_frame)
        self.player.mediaStatusChanged.connect(self.loop_video)

        video_path = asset_path("videos", "main_menu.mp4")
        self.player.setSource(QUrl.fromLocalFile(video_path))
        self.player.play()

        # ======================================
        # OVERLAY UI (au-dessus)
        # ======================================
        self.overlay_widget = QtWidgets.QWidget()
        self.overlay_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.overlay_widget.setStyleSheet("background: transparent;")

        overlay_layout = QtWidgets.QVBoxLayout(self.overlay_widget)
        overlay_layout.setContentsMargins(0, 0, 0, 0)

        tmp = QtWidgets.QMainWindow()
        uic.loadUi(gui_path("Main_Menu.ui"), tmp)

        content = tmp.centralWidget()
        # ======================================
        # NOUVEAU TITRE RECHERCHE
        # ======================================

        label_titre = content.findChild(QtWidgets.QLabel, "label_titre")

        if label_titre:
            label_titre.setText("""
            <div align="center">
                <div style="font-size:14pt; font-weight:300; letter-spacing:1px;">
                    Institut Lyfe × ENISE
                </div>
                <br>
                <div style="font-size:22pt; font-weight:700; letter-spacing:2px;">
                    PROJECTIVE AUGMENTED REALITY PLATFORM
                </div>
                <br>
                <div style="font-size:14pt; font-style:italic;">
                    for Sensory Research
                </div>
            </div>
            """)

            label_titre.setStyleSheet("""
                QLabel {
                    color: white;
                }
            """)

        # =========================
        # 1) VOILE SOMBRE (overlay)
        # =========================
        veil = QtWidgets.QWidget(self.overlay_widget)
        veil.setObjectName("veil")
        veil.setStyleSheet("""
        #veil {
            background-color: rgba(0, 0, 0, 120);  /* augmente 120 -> 160 si tu veux plus sombre */
        }
        """)
        veil.lower()  # le voile derrière les boutons mais au-dessus de la vidéo

        # IMPORTANT : le voile doit suivre la taille de l'overlay
        def _resize_veil():
            veil.setGeometry(self.overlay_widget.rect())

        _resize_veil()

        content.setParent(self.overlay_widget)
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content.setStyleSheet("background: transparent;")

        overlay_layout.addWidget(content)

        label_titre = content.findChild(QtWidgets.QLabel, "label_titre")
        if label_titre:
            label_titre.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 22px;
                    font-weight: 700;
                }
            """)


        # ======================================
        # AJOUT AU STACK
        # ======================================
        self.stack.addWidget(self.video_label)      # index 0
        self.stack.addWidget(self.overlay_widget)   # index 1

        # CRUCIAL : overlay au-dessus
        self.stack.setCurrentWidget(self.overlay_widget)

        # (double sécurité)
        self.overlay_widget.raise_()

        # ======================================
        # BOUTONS
        # ======================================
        self.pushButton_Record = content.findChild(QtWidgets.QPushButton, "pushButton_Record")
        self.pushButton_2_ARS = content.findChild(QtWidgets.QPushButton, "pushButton_2_ARS")
        self.pushButton_background = content.findChild(QtWidgets.QPushButton, "pushButton_background")
        self.pushButton_Settings = content.findChild(QtWidgets.QPushButton, "pushButton_Settings")
        self.pushButton_Quit = content.findChild(QtWidgets.QPushButton, "pushButton_Quit")

        if not all([self.pushButton_Record, self.pushButton_2_ARS, self.pushButton_background,
                    self.pushButton_Settings, self.pushButton_Quit]):
            print("[UI] ERROR: boutons manquants dans Main_Menu.ui")

        self.pushButton_Record.clicked.connect(self.go_to_record)
        self.pushButton_2_ARS.clicked.connect(self.go_to_RA)
        self.pushButton_background.clicked.connect(self.go_to_background)
        self.pushButton_Settings.clicked.connect(self.go_to_settings)
        self.pushButton_Quit.clicked.connect(self.quit_app)

        btn_style = """
        QPushButton {
            color: white;
            background-color: rgba(255, 255, 255, 40);
            border: 1px solid rgba(255, 255, 255, 90);
            border-radius: 14px;
            padding: 12px;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: rgba(255, 255, 255, 70);
        }
        QPushButton:pressed {
            background-color: rgba(255, 255, 255, 110);
        }
        """

        for b in [self.pushButton_Record, self.pushButton_2_ARS, self.pushButton_background,
                self.pushButton_Settings, self.pushButton_Quit]:
            if b:
                b.setStyleSheet(btn_style)


    # ======================================
    # VIDEO FRAMES -> QLabel
    # ======================================
    def on_video_frame(self, frame):
        if not frame or not frame.isValid():
            return

        img = frame.toImage()
        if img.isNull():
            return

        pix = QPixmap.fromImage(img)
        self._last_pixmap = pix
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self):
        if self._last_pixmap is None:
            return
        scaled = self._last_pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,  # effet "fond d'écran"
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    # ======================================
    # LOOP VIDEO
    # ======================================
    def loop_video(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.setPosition(0)
            self.player.play()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled_pixmap()
        # resize du voile
        for w in self.overlay_widget.findChildren(QtWidgets.QWidget, "veil"):
            w.setGeometry(self.overlay_widget.rect())


    # ======================================
    # RESIZE
    # ======================================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    # ======================================
    # NAVIGATION
    # ======================================
    def go_to_RA(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.RA_window)

    def go_to_record(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.protocol_home)
        self.parent.protocol_home.setFocus()

    def go_to_background(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.Background_Window)
        self.parent.Background_Window.setFocus()

    def go_to_settings(self):
        self.parent.settings_window.exec()

    def quit_app(self):
        QtWidgets.QApplication.quit()
