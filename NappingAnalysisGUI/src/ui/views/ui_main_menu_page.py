from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from src.core.utils.paths import gui_path, asset_path


class MainMenuPage(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # --------------------------------------------------
        # LAYOUT PRINCIPAL
        # --------------------------------------------------
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # --------------------------------------------------
        # VIDEO BACKGROUND
        # --------------------------------------------------
        self.video_widget = QVideoWidget(self)
        self.video_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)

        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.0)  # Muet

        video_path = asset_path("videos", "main_menu.mp4")
        self.player.setSource(QUrl.fromLocalFile(video_path))
        self.player.mediaStatusChanged.connect(self.handle_media_status)
        self.player.play()

        # Ajouter la vidéo au layout
        self.main_layout.addWidget(self.video_widget)

        # --------------------------------------------------
        # UI CONTENT (au-dessus de la vidéo)
        # --------------------------------------------------
        tmp = QtWidgets.QMainWindow()
        uic.loadUi(gui_path("Main_Menu.ui"), tmp)

        content = tmp.centralWidget()
        content.setParent(self)
        content.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.main_layout.addWidget(content)

        # Mettre la vidéo en arrière-plan
        self.video_widget.lower()

        # --------------------------------------------------
        # RÉCUPÉRATION DES BOUTONS
        # --------------------------------------------------
        self.pushButton_Record = content.findChild(QtWidgets.QPushButton, "pushButton_Record")
        self.pushButton_2_ARS = content.findChild(QtWidgets.QPushButton, "pushButton_2_ARS")
        self.pushButton_background = content.findChild(QtWidgets.QPushButton, "pushButton_background")
        self.pushButton_Settings = content.findChild(QtWidgets.QPushButton, "pushButton_Settings")
        self.pushButton_Quit = content.findChild(QtWidgets.QPushButton, "pushButton_Quit")

        # Sécurité
        if not all([
            self.pushButton_Record,
            self.pushButton_2_ARS,
            self.pushButton_background,
            self.pushButton_Settings,
            self.pushButton_Quit
        ]):
            print("⚠️ Erreur : Boutons non trouvés dans Main_Menu.ui")

        # --------------------------------------------------
        # CONNECTIONS
        # --------------------------------------------------
        self.pushButton_Record.clicked.connect(self.go_to_record)
        self.pushButton_2_ARS.clicked.connect(self.go_to_RA)
        self.pushButton_background.clicked.connect(self.go_to_background)
        self.pushButton_Settings.clicked.connect(self.go_to_settings)
        self.pushButton_Quit.clicked.connect(self.quit_app)

    # --------------------------------------------------
    # LOOP VIDEO
    # --------------------------------------------------
    def handle_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.setPosition(0)
            self.player.play()

    # --------------------------------------------------
    # RESIZE (important pour full background)
    # --------------------------------------------------
    def resizeEvent(self, event):
        self.video_widget.setGeometry(self.rect())
        super().resizeEvent(event)

    # --------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------
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
