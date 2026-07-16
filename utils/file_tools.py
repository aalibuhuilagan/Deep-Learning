from PyQt5.QtWidgets import QFileDialog

def select_image(parent):
    return QFileDialog.getOpenFileName(parent, "选择图片", "", "Image Files (*.png *.jpg *.jpeg)")