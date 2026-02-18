# -*- coding: utf-8 -*-
from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QPixmap
from PyQt6 import QtCore
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink

from src.core.utils.paths import gui_path, asset_path


class MainMenuPage(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.theme = "dark"   # "dark" ou "light" (tu peux mettre "light" si tu veux démarrer en clair)
        self.setContentsMargins(0, 0, 0, 0)

        # =============================
        # STACK : VIDEO + OVERLAY
        # =============================
        self.stack = QtWidgets.QStackedLayout(self)
        self.stack.setStackingMode(QtWidgets.QStackedLayout.StackingMode.StackAll)
        self.stack.setContentsMargins(0, 0, 0, 0)

        # =============================
        # VIDEO BACKGROUND -> QLabel
        # =============================
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setScaledContents(False)
        self.video_label.setStyleSheet("background: black;")
        self.video_label.setScaledContents(False)
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

        # =============================
        # OVERLAY (UI)
        # =============================
        self.overlay_widget = QtWidgets.QWidget()
        self.overlay_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.overlay_widget.setStyleSheet("background: transparent;")

        overlay_layout = QtWidgets.QVBoxLayout(self.overlay_widget)
        overlay_layout.setContentsMargins(0, 0, 0, 0)

        tmp = QtWidgets.QMainWindow()
        uic.loadUi(gui_path("Main_Menu.ui"), tmp)

        content = tmp.centralWidget()
        self.content = content
        content.setParent(self.overlay_widget)
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content.setStyleSheet("background: transparent;")
        overlay_layout.addWidget(content)

        # Masquer la ligne noire du .ui (name="line")
        line = content.findChild(QtWidgets.QFrame, "line")
        if line:
            line.hide()

        # =============================
        # TITRE (garder la ref + restyle via apply_theme)
        # =============================
        self.label_titre = content.findChild(QtWidgets.QLabel, "label_titre")
        if self.label_titre:
            self.label_titre.setText("""
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
            self.label_titre.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.label_titre.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
            self.label_titre.setParent(self.overlay_widget)
            self.label_titre.raise_()

        # =============================
        # AJOUT AU STACK
        # =============================
        self.stack.addWidget(self.video_label)      # index 0
        self.stack.addWidget(self.overlay_widget)   # index 1
        self.stack.setCurrentWidget(self.overlay_widget)

        # =============================
        # BOUTONS depuis le .ui
        # =============================
        self.pushButton_Record = content.findChild(QtWidgets.QPushButton, "pushButton_Record")
        self.pushButton_2_ARS = content.findChild(QtWidgets.QPushButton, "pushButton_2_ARS")
        self.pushButton_background = content.findChild(QtWidgets.QPushButton, "pushButton_background")
        self.pushButton_Settings = content.findChild(QtWidgets.QPushButton, "pushButton_Settings")
        self.pushButton_Quit = content.findChild(QtWidgets.QPushButton, "pushButton_Quit")

        missing = []
        for name, w in [
            ("pushButton_Record", self.pushButton_Record),
            ("pushButton_2_ARS", self.pushButton_2_ARS),
            ("pushButton_background", self.pushButton_background),
            ("pushButton_Settings", self.pushButton_Settings),
            ("pushButton_Quit", self.pushButton_Quit),
        ]:
            if w is None:
                missing.append(name)

        if missing:
            print("[UI] ERROR: boutons manquants dans Main_Menu.ui :", missing)

        # Connexions navigation (si boutons présents)
        if self.pushButton_Record:
            self.pushButton_Record.clicked.connect(self.go_to_record)
        if self.pushButton_2_ARS:
            self.pushButton_2_ARS.clicked.connect(self.go_to_RA)
        if self.pushButton_background:
            self.pushButton_background.clicked.connect(self.go_to_background)
        if self.pushButton_Settings:
            self.pushButton_Settings.clicked.connect(self.go_to_settings)
        if self.pushButton_Quit:
            self.pushButton_Quit.clicked.connect(self.quit_app)

        # Labels (texte)
        if self.pushButton_Record:
            self.pushButton_Record.setText("Start Session")
        if self.pushButton_2_ARS:
            self.pushButton_2_ARS.setText("Augmented Reality")
        if self.pushButton_background:
            self.pushButton_background.setText("Projection Background")
        if self.pushButton_Settings:
            self.pushButton_Settings.setText("System Settings")
        if self.pushButton_Quit:
            self.pushButton_Quit.setText("Exit")

        # =============================
        # LEFT PANEL
        # =============================
        self.left_panel = QtWidgets.QFrame(self.overlay_widget)
        self.left_panel.setObjectName("leftPanel")

        self.left_layout = QtWidgets.QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(24, 24, 24, 24)
        self.left_layout.setSpacing(16)
        self.left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.left_title = QtWidgets.QLabel("MENU", self.left_panel)
        self.left_title.setObjectName("leftTitle")
        self.left_layout.addWidget(self.left_title)
        self.left_layout.addSpacing(10)

        # Glow vertical à droite
        self.left_glow = QtWidgets.QFrame(self.overlay_widget)
        self.left_glow.setObjectName("leftGlow")
        self.left_glow.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.left_glow.raise_()

        # Mettre les boutons dans le panel (SAUF thème)
        for b in [
            self.pushButton_Record,
            self.pushButton_2_ARS,
            self.pushButton_background,
            self.pushButton_Settings,
            self.pushButton_Quit
        ]:
            if b:
                b.setParent(self.left_panel)
                self.left_layout.addWidget(b)

        self.left_layout.addStretch(1)
        self.left_panel.raise_()

        # =============================
        # BOUTON THEME (flottant en haut à droite)
        # =============================
        self.pushButton_Theme = QtWidgets.QPushButton("☀/🌙", self.overlay_widget)
        self.pushButton_Theme.setObjectName("pushButton_ThemeFloating")
        self.pushButton_Theme.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pushButton_Theme.setFixedSize(44, 34)
        self.pushButton_Theme.clicked.connect(self.toggle_theme)
        self.pushButton_Theme.raise_()

        # =============================
        # Appliquer thème initial + lancer vidéo
        # =============================
        self.apply_theme()

    # =============================
    # VIDEO FRAMES -> QLabel
    # =============================
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

        target = self.video_label.size()
        if target.width() <= 0 or target.height() <= 0:
            return

        # ---- réglage zoom ----
        # dark : 1.00 (plein cover)
        # light : 0.88 à 0.95 (moins zoomé mais toujours cover)
        zoom = 1.0 if self.theme == "dark" else 0.92

        # on réduit un peu la taille cible => effet "dézoom"
        scaled_target = target * zoom

        # 1) cover sur la taille réduite
        scaled = self._last_pixmap.scaled(
            scaled_target,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )

        # 2) puis on recadre au centre sur la taille finale (target)
        x = max(0, (scaled.width() - target.width()) // 2)
        y = max(0, (scaled.height() - target.height()) // 2)

        cropped = scaled.copy(x, y, target.width(), target.height())
        self.video_label.setPixmap(cropped)


    # =============================
    # THEME APPLY
    # =============================
    def apply_theme(self):
        # ---- Panel + titre ----
        if self.theme == "light":
            self.left_panel.setStyleSheet("""
            #leftPanel {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255,255,255,240),
                    stop:1 rgba(255,255,255,170)
                );
                border-right: 1px solid rgba(0,0,0,35);
            }
            QLabel#leftTitle {
                color: rgba(20,20,20,200);
                font-size: 11pt;
                font-weight: 800;
                letter-spacing: 3px;
            }
            """)
            self.left_glow.setStyleSheet("""
            QFrame#leftGlow {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,180,60,0),
                    stop:0.25 rgba(255,180,60,60),
                    stop:0.5 rgba(255,180,60,130),
                    stop:0.75 rgba(255,180,60,60),
                    stop:1 rgba(255,180,60,0)
                );
            }
            """)

            btn_style = """
            QPushButton {
                color: rgba(20,20,20,230);
                background-color: rgba(255,255,255,210);
                border: 1px solid rgba(0,0,0,45);
                border-radius: 18px;
                padding: 18px 22px;
                font-size: 16px;
                font-weight: 650;
            }
            QPushButton:hover {
                border: 1px solid rgba(255,180,60,200);
            }
            QPushButton:pressed {
                background-color: rgba(255,255,255,235);
            }
            """
            theme_btn_style = """
            QPushButton#pushButton_ThemeFloating {
                color: rgba(20,20,20,230);
                background-color: rgba(255,255,255,210);
                border: 1px solid rgba(0,0,0,70);
                border-radius: 10px;
                font-weight: 800;
            }
            QPushButton#pushButton_ThemeFloating:hover {
                border: 1px solid rgba(255,180,60,220);
            }
            """

            title_style = "QLabel { color: rgba(15,15,15,235); background: transparent; }"

        else:
            self.left_panel.setStyleSheet("""
            #leftPanel {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(20,24,30,245),
                    stop:0.65 rgba(20,24,30,225),
                    stop:1 rgba(20,24,30,160)
                );
                border-right: 1px solid rgba(255,255,255,35);
            }
            QLabel#leftTitle {
                color: rgba(255,255,255,215);
                font-size: 11pt;
                font-weight: 800;
                letter-spacing: 3px;
            }
            """)
            self.left_glow.setStyleSheet("""
            QFrame#leftGlow {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,214,120,0),
                    stop:0.25 rgba(255,214,120,70),
                    stop:0.5 rgba(255,214,120,150),
                    stop:0.75 rgba(255,214,120,70),
                    stop:1 rgba(255,214,120,0)
                );
            }
            """)

            btn_style = """
            QPushButton {
                color: rgba(255,255,255,235);
                letter-spacing: 0.4px;
                background-color: rgba(255,214,120,18);
                border: 1px solid rgba(255,214,120,85);
                border-radius: 18px;
                padding: 18px 22px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(255,214,120,32);
                border: 1px solid rgba(255,214,120,155);
            }
            QPushButton:pressed {
                background-color: rgba(255,214,120,45);
                border: 1px solid rgba(255,214,120,190);
            }
            """
            theme_btn_style = """
            QPushButton#pushButton_ThemeFloating {
                color: rgba(255,255,255,235);
                background-color: rgba(20,24,30,175);
                border: 1px solid rgba(255,214,120,140);
                border-radius: 10px;
                font-weight: 800;
            }
            QPushButton#pushButton_ThemeFloating:hover {
                border: 1px solid rgba(255,214,120,210);
            }
            """

            title_style = "QLabel { color: rgba(255,255,255,240); background: transparent; }"

        # ---- Uniformiser taille + style boutons panel ----
        BTN_W, BTN_H = 360, 60
        for b in [self.pushButton_Record, self.pushButton_2_ARS, self.pushButton_background,
                  self.pushButton_Settings, self.pushButton_Quit]:
            if b:
                b.setMinimumSize(BTN_W, BTN_H)
                b.setMaximumSize(BTN_W, BTN_H)
                b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.setStyleSheet(btn_style)

        # ---- Style bouton thème flottant ----
        if self.pushButton_Theme:
            self.pushButton_Theme.setStyleSheet(theme_btn_style)

        # ---- Titre couleur (IMPORTANT) ----
        if self.label_titre:
            self.label_titre.setStyleSheet(title_style)

        # ---- VIDEO selon thème ----
        video_file = "main_menu_light.mp4" if self.theme == "light" else "main_menu.mp4"
        video_path = asset_path("videos", video_file)
        self.player.setSource(QUrl.fromLocalFile(video_path))
        self.player.play()

    # =============================
    # LOOP VIDEO
    # =============================
    def loop_video(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.setPosition(0)
            self.player.play()

    # =============================
    # RESIZE
    # =============================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

        # Left panel fixé à gauche
        panel_w = 420
        self.left_panel.setGeometry(0, 0, panel_w, self.height())

        # Glow à droite du panel
        x = self.left_panel.geometry().right()
        self.left_glow.setGeometry(x, 0, 2, self.height())
        self.left_glow.raise_()

        # Bouton thème top-right
        margin = 16
        self.pushButton_Theme.move(self.width() - self.pushButton_Theme.width() - margin, margin)
        self.pushButton_Theme.raise_()
        # =============================
        # TITRE : position fixe (dans le repère de content)
        # =============================
        if self.label_titre and hasattr(self, "content"):
            panel_w = 420
            margin_x = 50
            margin_y = 180

            # position voulue dans le repère de MainMenuPage
            gx = panel_w + margin_x
            gy = margin_y

            # convertit vers le repère du parent réel (content)
            p = self.content.mapFrom(self, self.mapToGlobal(QtCore.QPoint(gx, gy)))
            x = p.x()
            y = p.y()

            w = max(200, self.width() - gx - 60)
            h = 220

            self.label_titre.setGeometry(x, y, w, h)
            self.label_titre.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            self.label_titre.raise_()

    # =============================
    # TOGGLE THEME
    # =============================
    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.apply_theme()

    # =============================
    # NAVIGATION
    # =============================
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
