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
        # VIDEO BACKGROUND -> QLabel
        # ======================================
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: black;")
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
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
        # OVERLAY (UI au-dessus)
        # ======================================
        self.overlay_widget = QtWidgets.QWidget()
        self.overlay_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.overlay_widget.setStyleSheet("background: transparent;")

        overlay_layout = QtWidgets.QVBoxLayout(self.overlay_widget)
        overlay_layout.setContentsMargins(0, 0, 0, 0)

        tmp = QtWidgets.QMainWindow()
        uic.loadUi(gui_path("Main_Menu.ui"), tmp)

        content = tmp.centralWidget()
        content.setParent(self.overlay_widget)
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content.setStyleSheet("background: transparent;")

        overlay_layout.addWidget(content)

        # ======================================
        # MASQUER LA LIGNE NOIRE DU .UI (name="line")
        # ======================================
        line = content.findChild(QtWidgets.QFrame, "line")
        if line:
            line.hide()

        # ======================================
        # TITRE RECHERCHE
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
            label_titre.setStyleSheet("QLabel { color: white; }")

        # ======================================
        # AJOUT AU STACK
        # ======================================
        self.stack.addWidget(self.video_label)      # index 0
        self.stack.addWidget(self.overlay_widget)   # index 1
        self.stack.setCurrentWidget(self.overlay_widget)

        # ======================================
        # BOUTONS (depuis le .ui)
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

        # ---- Labels (texte des boutons) ----
        self.pushButton_Record.setText("Start Session")
        self.pushButton_2_ARS.setText("Augmented Reality")
        self.pushButton_background.setText("Projection Background")
        self.pushButton_Settings.setText("System Settings")
        self.pushButton_Quit.setText("Exit")

        # ======================================
        # LEFT PANEL (film / contraste)
        # ======================================
        self.left_panel = QtWidgets.QFrame(self.overlay_widget)
        self.left_panel.setObjectName("leftPanel")
        self.left_panel.setStyleSheet("""
        #leftPanel {
            /* Teinte très légère (bleu/gris) + gradient => contraste même sur fond noir */
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(20, 24, 30, 245),
                stop:0.65 rgba(20, 24, 30, 220),
                stop:1 rgba(20, 24, 30, 160)
            );
            border-right: 1px solid rgba(255, 255, 255, 35);
        }
        """)

        self.left_glow = QtWidgets.QFrame(self.overlay_widget)
        self.left_glow.setStyleSheet("""
        background: qlineargradient(
            x1:0, y1:0, x2:0, y2:1,
            stop:0 rgba(255, 214, 120, 0),
            stop:0.25 rgba(255, 214, 120, 90),
            stop:0.5 rgba(255, 214, 120, 160),
            stop:0.75 rgba(255, 214, 120, 90),
            stop:1 rgba(255, 214, 120, 0)
        );
        """)
        self.left_glow.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.left_glow.raise_()

        self.left_layout = QtWidgets.QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(24, 24, 24, 24)
        self.left_layout.setSpacing(16)
        self.left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)


        self.left_title = QtWidgets.QLabel("MENU", self.left_panel)
        self.left_title.setStyleSheet("""
        QLabel {
            color: rgba(255, 255, 255, 210);
            font-size: 11pt;
            font-weight: 700;
            letter-spacing: 3px;
        }
        """)

        self.left_layout.addWidget(self.left_title)
        self.left_layout.addSpacing(10)

        # ======================================
        # STYLE PREMIUM BOUTONS (inchangé)
        # ======================================
        # btn_style = """
        # QPushButton {
        #     color: rgba(255, 255, 255, 235);
        #     background-color: rgba(255, 255, 255, 38);
        #     border: 1px solid rgba(255, 255, 255, 90);
        #     border-radius: 18px;
        #     padding: 18px 22px;
        #     font-size: 16px;
        #     font-weight: 600;
        # }
        # QPushButton:hover {
        #     background-color: rgba(255, 255, 255, 58);
        #     border: 1px solid rgba(255, 255, 255, 140);
        # }
        # QPushButton:pressed {
        #     background-color: rgba(255, 255, 255, 80);
        #     border: 1px solid rgba(255, 255, 255, 170);
        # }
        # QPushButton:focus {
        #     outline: none;
        #     border: 2px solid rgba(255, 255, 255, 200);
        # }
        # """
        btn_style = """
        QPushButton {
            color: rgba(255, 255, 255, 235);
            letter-spacing: 0.4px;
            background-color: rgba(255, 214, 120, 18);  /* teinte ambre très légère */
            border: 1px solid rgba(255, 214, 120, 80);  /* bord ambre */
            border-radius: 18px;
            padding: 18px 22px;
            font-size: 16px;
            font-weight: 600;
        }

        QPushButton:hover {
            background-color: rgba(255, 214, 120, 32);
            border: 1px solid rgba(255, 214, 120, 140);
        }

        QPushButton:pressed {
            background-color: rgba(255, 214, 120, 45);
            border: 1px solid rgba(255, 214, 120, 180);
        }

        QPushButton:focus {
            outline: none;
            border: 2px solid rgba(255, 214, 120, 220);
        }
        """

        for b in [self.pushButton_Record, self.pushButton_2_ARS, self.pushButton_background,
                  self.pushButton_Settings, self.pushButton_Quit]:
            if b:
                b.setParent(self.left_panel)
                # b.setMinimumHeight(70)
                # b.setMaximumWidth(360)
                # Taille UNIFORME
                BTN_W = 360
                BTN_H = 60

                b.setMinimumSize(BTN_W, BTN_H)
                b.setMaximumSize(BTN_W, BTN_H)
                b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                QtWidgets.QSizePolicy.Policy.Fixed)
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.setStyleSheet(btn_style)
                self.left_layout.addWidget(b)

        self.left_layout.addStretch(1)
        self.left_panel.raise_()

    # ======================================
    # VIDEO FRAMES -> QLabel
    # ======================================
    def on_video_frame(self, frame):
        if not frame or not frame.isValid():
            return
        img = frame.toImage()
        if img.isNull():
            return
        self._last_pixmap = QPixmap.fromImage(img)
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self):
        if self._last_pixmap is None:
            return
        scaled = self._last_pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
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

    # ======================================
    # RESIZE
    # ======================================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

        # Left panel fixé à gauche
        panel_w = 420
        self.left_panel.setGeometry(0, 0, panel_w, self.height())
        # Ligne glow à droite du panel
        if hasattr(self, "left_glow") and self.left_glow:
            x = self.left_panel.geometry().right()  # bord droit du panel
            self.left_glow.setGeometry(x, 0, 2, self.height())
            self.left_glow.raise_()

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
