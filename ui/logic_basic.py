from PyQt5.QtWidgets import *
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from ui.widgets import ImageDisplay
from core import load_model, preprocess_image
from core.cam_engine import generate_heatmap
from config import OUTPUT_DIR
from PIL import Image


class BasicWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.model, self.target_layer = load_model()
        self.img_path = None
        self.original_img_np = None  # 存**原图**
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        bar = QHBoxLayout()
        self.btn_open = QPushButton("选择图片")
        self.btn_run = QPushButton("生成热力图")
        self.model_select = QComboBox()
        self.model_select.addItems(["resnet50", "vgg16", "vit_b_16"])
        bar.addWidget(self.btn_open)
        bar.addWidget(self.model_select)
        bar.addWidget(self.btn_run)
        layout.addLayout(bar)

        img_layout = QHBoxLayout()
        self.img_original = ImageDisplay()
        self.img_result = ImageDisplay()
        img_layout.addWidget(self.img_original)
        img_layout.addWidget(self.img_result)
        layout.addLayout(img_layout)
        self.setLayout(layout)

        self.btn_open.clicked.connect(self.on_open)
        self.btn_run.clicked.connect(self.on_run)

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.img_path = path
            # 这里永远存原图，不缩放
            self.original_img_np = np.array(Image.open(path).convert("RGB"))
            self.img_original.set_image(self.original_img_np)
            self.img_result.clear()

    def on_run(self):
        if not self.img_path or self.original_img_np is None:
            return

        try:
            model_name = self.model_select.currentText()
            self.model, self.target_layer = load_model(model_name)
            is_vit = "vit" in model_name

            # preprocess_image 内部会缩放到 224 给模型用
            tensor, _ = preprocess_image(self.img_path)

            # 传给 generate_heatmap 的是：
            # 1) tensor：224 给模型
            # 2) original_img_np：**原图** 用来叠加
            heatmap = generate_heatmap(
                model=self.model,
                target_layers=self.target_layer,
                image_tensor=tensor,
                original_img_np=self.original_img_np,
                is_vit=is_vit
            )

            self.img_result.set_image(heatmap)
            self.save_comparison(self.original_img_np, heatmap, model_name)
            print("✅ 热力图生成并保存成功！")

        except Exception as e:
            print("❌ 错误：", str(e))

    def save_comparison(self, original, heatmap, model_name):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(heatmap)
        plt.title(f"Grad-CAM ({model_name})")
        plt.axis("off")

        filename = f"result_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()