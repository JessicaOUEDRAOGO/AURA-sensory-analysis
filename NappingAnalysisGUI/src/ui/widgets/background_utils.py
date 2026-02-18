from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

def apply_page_background(widget: QWidget, image_path: str, object_name: str):
    p = image_path.replace("\\", "/")

    widget.setObjectName(object_name)

    # IMPORTANT: sinon Qt ignore parfois le background des stylesheets
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(False)

    widget.setStyleSheet(widget.styleSheet() + f"""
        QWidget#{object_name} {{
            border-image: url("{p}") 0 0 0 0 stretch stretch;
        }}
    """)

