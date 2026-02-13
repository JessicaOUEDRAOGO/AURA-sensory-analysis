from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem
from PyQt6.QtGui import QPen, QBrush, QColor
from PyQt6.QtCore import Qt

class Layer:
    def __init__(self, layer_id, layer_type):
        self.layer_id = layer_id
        self.layer_type = layer_type

    def update_properties(self, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")

class CircleLayer(Layer):
    def __init__(self, layer_id, x, y, radius, color, thickness=10, fill=False):
        super().__init__(layer_id, "circle")
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.thickness = thickness
        self.fill = fill
        self.graphics_item = None

    def render(self, scene):
        if self.graphics_item and self.graphics_item.scene() == scene:
            scene.removeItem(self.graphics_item)
        self.graphics_item = QGraphicsEllipseItem(
            self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2
        )
        self.graphics_item.setPen(QPen(QColor(self.color), self.thickness))
        if self.fill:
            self.graphics_item.setBrush(QBrush(QColor(self.color)))
        else:
            self.graphics_item.setBrush(QBrush(Qt.GlobalColor.transparent))
        scene.addItem(self.graphics_item)

    def update_properties(self, x=None, y=None, radius=None, color=None, thickness=None, fill=None):
        if x is not None: self.x = x
        if y is not None: self.y = y
        if radius is not None: self.radius = radius
        if color is not None: self.color = color
        if thickness is not None: self.thickness = thickness
        if fill is not None: self.fill = fill

class LineLayer(Layer):
    def __init__(self, layer_id, x1, y1, x2, y2, color, thickness):
        super().__init__(layer_id, "line")
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color
        self.thickness = thickness
        self.graphics_item = None

    def render(self, scene):
        if self.graphics_item:
            scene.removeItem(self.graphics_item)
        self.graphics_item = scene.addLine(
            self.x1, self.y1, self.x2, self.y2,
            QPen(QColor(self.color), self.thickness)
        )

    def update_properties(self, x1=None, y1=None, x2=None, y2=None, color=None, thickness=None):
        if x1 is not None: self.x1 = x1
        if y1 is not None: self.y1 = y1
        if x2 is not None: self.x2 = x2
        if y2 is not None: self.y2 = y2
        if color is not None: self.color = color
        if thickness is not None: self.thickness = thickness

class TextLayer(Layer):
    def __init__(self, layer_id, x, y, text, color, font_size, rotation=0, bold=False, italic=False, underline=False):
        super().__init__(layer_id, "text")
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.font_size = font_size
        self.rotation = rotation
        self.bold = bold
        self.italic = italic
        self.underline = underline
        self.graphics_item = None

    def render(self, scene):
        if self.graphics_item:
            scene.removeItem(self.graphics_item)
        self.graphics_item = QGraphicsTextItem(self.text)
        self.graphics_item.setDefaultTextColor(QColor(self.color))
        font = self.graphics_item.font()
        font.setPointSize(self.font_size)
        font.setBold(self.bold)
        font.setItalic(self.italic)
        font.setUnderline(self.underline)
        self.graphics_item.setFont(font)
        bounding_rect = self.graphics_item.boundingRect()
        centered_x = self.x - bounding_rect.width() / 2
        centered_y = self.y - bounding_rect.height() / 2
        self.graphics_item.setTransformOriginPoint(bounding_rect.center())
        self.graphics_item.setPos(centered_x, centered_y)
        self.graphics_item.setRotation(self.rotation)
        scene.addItem(self.graphics_item)

    def update_properties(self, x=None, y=None, text=None, color=None, font_size=None, rotation=None, bold=None, italic=None, underline=None):
        if x is not None: self.x = x
        if y is not None: self.y = y
        if text is not None: self.text = text
        if color is not None: self.color = color
        if font_size is not None: self.font_size = font_size
        if rotation is not None: self.rotation = rotation
        if bold is not None: self.bold = bold
        if italic is not None: self.italic = italic
        if underline is not None: self.underline = underline