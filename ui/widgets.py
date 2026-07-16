from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageDisplay(QLabel):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 1px solid #999; background-color:#f8f8f8;")
        self.setMinimumSize(450, 450)
        self.setAlignment(Qt.AlignCenter)

    def set_image(self, img):
        h, w, ch = img.shape
        q_img = QImage(img.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pix)