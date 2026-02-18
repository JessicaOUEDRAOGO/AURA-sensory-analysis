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
        # FOOTER INFO (research)
        # =========================
        self.footer = QtWidgets.QFrame(self.overlay_widget)
        self.footer.setObjectName("footer")
        self.footer.setStyleSheet("""
        #footer {
            background-color: rgba(0, 0, 0, 90);
            border-top: 1px solid rgba(255, 255, 255, 60);
        }
        QLabel#footerText {
            color: rgba(255, 255, 255, 210);
            font-size: 11pt;
            padding: 8px 14px;
        }
        """)

        self.footer_text = QtWidgets.QLabel(self.footer)
        self.footer_text.setObjectName("footerText")
        self.footer_text.setText("Ready • Protocol: PROTO_TEST • Projector: 1 • Camera: 0 • v2.1")
        self.footer_text.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        # placer le footer (sera ajusté au resize)
        self.footer.setGeometry(0, self.height() - 44, self.width(), 44)
        self.footer_text.setGeometry(0, 0, self.footer.width(), self.footer.height())
        self.footer.raise_()


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

        # =========================
        # LEFT PANEL (glass) + layout boutons
        # =========================
        self.left_panel = QtWidgets.QFrame(self.overlay_widget)
        self.left_panel.setObjectName("leftPanel")
        # self.left_panel.setStyleSheet("""
        # #leftPanel {
        #     background-color: rgba(0, 0, 0, 110);   /* FILM sombre */
        #     border-right: 1px solid rgba(255, 255, 255, 60);
        # }
        # """)
        self.left_panel.setStyleSheet("""
        #leftPanel {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(0, 0, 0, 160),
                stop:1 rgba(0, 0, 0, 40)
            );
            border-right: 1px solid rgba(255, 255, 255, 60);
        }
        """)


        # Conteneur interne (marges)
        self.left_layout = QtWidgets.QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(24, 24, 24, 24)
        self.left_layout.setSpacing(16)

        # Petit header dans le panel (optionnel)
        self.left_title = QtWidgets.QLabel("MENU", self.left_panel)
        self.left_title.setStyleSheet("""
        QLabel {
            color: rgba(255, 255, 255, 180);
            font-size: 11pt;
            font-weight: 600;
            letter-spacing: 2px;
        }
        """)
        self.left_layout.addWidget(self.left_title)

        self.left_layout.addSpacing(10)

        # Déplacer les boutons dans le panel gauche
        for b in [self.pushButton_Record, self.pushButton_2_ARS, self.pushButton_background,
                self.pushButton_Settings, self.pushButton_Quit]:
            if b:
                b.setParent(self.left_panel)
                b.setMinimumHeight(70)   # ajuste si tu veux plus grand
                b.setMaximumWidth(360)   # largeur boutons dans le panel
                self.left_layout.addWidget(b)

        # pousser vers le haut (évite que ça se centre verticalement)
        self.left_layout.addStretch(1)

        # Important : le panel doit être au-dessus de la vidéo
        self.left_panel.raise_()

        # =========================
        # STYLE PREMIUM BOUTONS
        # =========================
        btn_style = """
        QPushButton {
            color: rgba(255, 255, 255, 235);
            background-color: rgba(255, 255, 255, 38);
            border: 1px solid rgba(255, 255, 255, 90);
            border-radius: 18px;
            padding: 18px 22px;
            font-size: 16px;
            font-weight: 600;
        }

        QPushButton:hover {
            background-color: rgba(255, 255, 255, 58);
            border: 1px solid rgba(255, 255, 255, 140);
        }

        QPushButton:pressed {
            background-color: rgba(255, 255, 255, 80);
            border: 1px solid rgba(255, 255, 255, 170);
        }

        QPushButton:focus {
            outline: none;
            border: 2px solid rgba(255, 255, 255, 200);
        }
        """

        # Appliquer aux boutons
        for b in [self.pushButton_Record, self.pushButton_2_ARS, self.pushButton_background,
                self.pushButton_Settings, self.pushButton_Quit]:
            if b:
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.setStyleSheet(btn_style)
        
        # =========================
        # PANEL GLASS derrière les boutons
        # =========================
        panel = QtWidgets.QFrame(self.overlay_widget)
        panel.setObjectName("centerPanel")
        panel.setStyleSheet("""
        #centerPanel {
            background-color: rgba(0, 0, 0, 90);
            border: 1px solid rgba(255, 255, 255, 70);
            border-radius: 22px;
        }
        """)
        panel.setGeometry(0, 0, 520, 610)  # sera centré au resize
        panel.lower()  # derrière les boutons, au-dessus du voile
        panel.raise_()  # on le remontera juste sous content après

        # on veut qu'il soit sous le content (boutons)
        panel.lower()


        # btn_style = """
        # QPushButton {
        #     color: white;
        #     background-color: rgba(255, 255, 255, 40);
        #     border: 1px solid rgba(255, 255, 255, 90);
        #     border-radius: 14px;
        #     padding: 12px;
        #     font-size: 16px;
        # }
        # QPushButton:hover {
        #     background-color: rgba(255, 255, 255, 70);
        # }
        # QPushButton:pressed {
        #     background-color: rgba(255, 255, 255, 110);
        # }
        # """

        # for b in [self.pushButton_Record, self.pushButton_2_ARS, self.pushButton_background,
        #         self.pushButton_Settings, self.pushButton_Quit]:
        #     if b:
        #         b.setStyleSheet(btn_style)


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
    
    # ======================================
    # RESIZE
    # ======================================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled_pixmap()
        # resize du voile
        for w in self.overlay_widget.findChildren(QtWidgets.QWidget, "veil"):
            w.setGeometry(self.overlay_widget.rect())
        # Footer resize
        if hasattr(self, "footer") and self.footer:
            h = 44
            self.footer.setGeometry(0, self.height() - h, self.width(), h)
            self.footer_text.setGeometry(0, 0, self.footer.width(), self.footer.height())
        # Centrer le panel
        if 'panel' in self.__dict__:
            w, h = 520, 610
            x = (self.width() - w) // 2
            y = (self.height() - h) // 2 + 40
            panel.setGeometry(x, y, w, h)
            panel.lower()          # derrière les boutons
            self.footer.raise_()   # footer toujours au-dessus
        # =========================
        # =========================
        # LEFT PANEL resize/position
        # =========================
        if hasattr(self, "left_panel") and self.left_panel:
            panel_w = 420  # largeur du panneau gauche (à ajuster)
            self.left_panel.setGeometry(0, 0, panel_w, self.height())



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
