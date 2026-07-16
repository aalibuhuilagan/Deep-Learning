from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QFont, QImage, QPixmap
import qtawesome as qta
import cv2
import numpy as np
import os
from PIL import Image
from core import load_model, preprocess_image
from core.cam_engine import generate_heatmap

# ------------------------------
# 视频线程（安全不崩溃）
# ------------------------------
class VideoWorker(QThread):
    frame_ready = pyqtSignal(QPixmap)
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, video_path, model, layers, is_vit, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model = model
        self.layers = layers
        self.is_vit = is_vit
        self.paused = False
        self.stop_flag = False
        self.mutex = QMutex()

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current = 0
            while not self.stop_flag:
                self.mutex.lock()
                if self.paused:
                    self.mutex.unlock()
                    self.msleep(50)
                    continue
                self.mutex.unlock()

                ret, frame = cap.read()
                if not ret:
                    break

                current += 1
                self.progress.emit(int(current / total * 100))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]

                temp = "temp_frame.jpg"
                Image.fromarray(frame_rgb).save(temp)
                tensor, _ = preprocess_image(temp)

                heat = generate_heatmap(
                    model=self.model,
                    target_layers=self.layers,
                    image_tensor=tensor,
                    original_img_np=frame_rgb,
                    is_vit=self.is_vit
                )

                qimg = QImage(heat.data, w, h, 3 * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(680, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.frame_ready.emit(pix)
                self.msleep(40)

            cap.release()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
# ------------------------------
# 摄像头线程（安全不崩溃）
# ------------------------------
class CameraWorker(QThread):
    frame_ready = pyqtSignal(QPixmap)
    error = pyqtSignal(str)

    def __init__(self, model, layers, is_vit, parent=None):
        super().__init__(parent)
        self.model = model
        self.layers = layers
        self.is_vit = is_vit
        self.paused = False
        self.stop_flag = False

    def run(self):
        try:
            cap = cv2.VideoCapture(0)
            while not self.stop_flag:
                if self.paused:
                    self.msleep(50)
                    continue
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                temp = "temp_cam.jpg"
                Image.fromarray(frame_rgb).save(temp)
                tensor, _ = preprocess_image(temp)
                heat = generate_heatmap(self.model, self.layers, tensor, frame_rgb, self.is_vit)
                qimg = QImage(heat.data, w, h, 3 * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(680, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.frame_ready.emit(pix)
                self.msleep(40)
            cap.release()
        except Exception as e:
            self.error.emit(str(e))
# ------------------------------
# 1. 批量图片
# ------------------------------
class BatchPage(QWidget):
    def __init__(self, parent_main):
        super().__init__()
        self.parent_main = parent_main
        self.folder = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("📁 批量图片热力图生成")
        title.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        layout.addWidget(title)

        self.btn_folder = QPushButton(qta.icon('fa5s.folder-open'), " 选择图片文件夹")
        self.btn_run = QPushButton(qta.icon('fa5s.tasks'), " 开始批量生成")
        self.status = QLabel("状态：等待操作")
        self.status.setStyleSheet("color:#333")

        layout.addWidget(self.btn_folder)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.status)
        layout.addStretch()

        self.btn_folder.clicked.connect(self.select_folder)
        self.btn_run.clicked.connect(self.run_batch)

    def select_folder(self):
        self.folder = QFileDialog.getExistingDirectory()
        if self.folder:
            self.status.setText(f"已选择：{self.folder}")

    def run_batch(self):
        if not self.parent_main.model or not self.folder:
            QMessageBox.warning(self, "提示", "请先加载模型并选择文件夹！")
            return

        self.status.setText("处理中...")
        out = "assets/output/batch"
        os.makedirs(out, exist_ok=True)
        exts = (".jpg", ".jpeg", ".png", ".bmp")

        for f in os.listdir(self.folder):
            if f.lower().endswith(exts):
                p = os.path.join(self.folder, f)
                img = np.array(Image.open(p).convert("RGB"))
                tensor, _ = preprocess_image(p)
                heat = generate_heatmap(
                    self.parent_main.model,
                    self.parent_main.layers,
                    tensor, img,
                    is_vit="vit" in self.parent_main.model_name
                )
                Image.fromarray(heat).save(os.path.join(out, f))

        self.status.setText("✅ 全部完成！已保存到 assets/output/batch")
        QMessageBox.information(self, "完成", "批量图片处理成功！")
# ------------------------------
# 2. 视频分析
# ------------------------------
class VideoPage(QWidget):
    def __init__(self, parent_main):
        super().__init__()
        self.parent_main = parent_main
        self.worker = None
        self.video_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🎬 视频热力图分析")
        title.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        layout.addWidget(title)

        self.display = QLabel("视频画面")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setStyleSheet("border:1px solid #ddd; background:#f9f9f9; min-height:420px")

        self.progress = QProgressBar()
        self.btn_select = QPushButton(qta.icon('fa5s.film'), " 选择视频")
        self.btn_start = QPushButton(qta.icon('fa5s.play'), " 开始")
        self.btn_pause = QPushButton(qta.icon('fa5s.pause'), " 暂停/继续")

        row = QHBoxLayout()
        row.addWidget(self.btn_select)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_pause)

        layout.addWidget(self.display)
        layout.addWidget(self.progress)
        layout.addLayout(row)

        self.btn_select.clicked.connect(self.select_video)
        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.pause)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.video_path = path

    def start(self):
        if not self.parent_main.model or not self.video_path:
            QMessageBox.warning(self, "提示", "请先加载模型并选择视频！")
            return
        self.worker = VideoWorker(
            self.video_path,
            self.parent_main.model,
            self.parent_main.layers,
            "vit" in self.parent_main.model_name
        )
        self.worker.frame_ready.connect(self.display.setPixmap)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.start()

    def pause(self):
        if self.worker:
            self.worker.paused = not self.worker.paused
# ------------------------------
# 3. 摄像头
# ------------------------------
class CameraPage(QWidget):
    def __init__(self, parent_main):
        super().__init__()
        self.parent_main = parent_main
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("📹 实时摄像头分析")
        title.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        layout.addWidget(title)

        self.display = QLabel("摄像头画面")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setStyleSheet("border:1px solid #ddd; background:#f9f9f9; min-height:420px")

        self.btn_start = QPushButton(qta.icon('fa5s.play'), " 打开摄像头")
        self.btn_pause = QPushButton(qta.icon('fa5s.pause'), " 暂停/继续")

        row = QHBoxLayout()
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_pause)

        layout.addWidget(self.display)
        layout.addLayout(row)

        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.pause)

    def start(self):
        if not self.parent_main.model:
            QMessageBox.warning(self, "提示", "请先加载模型！")
            return
        self.worker = CameraWorker(
            self.parent_main.model,
            self.parent_main.layers,
            "vit" in self.parent_main.model_name
        )
        self.worker.frame_ready.connect(self.display.setPixmap)
        self.worker.start()

    def pause(self):
        if self.worker:
            self.worker.paused = not self.worker.paused
# ------------------------------
# 4. 算法对比（最终稳定版，无任何报错）
# ------------------------------
class AlgoPage(QWidget):
    cam_result = pyqtSignal(int, np.ndarray)

    def __init__(self, parent_main):
        super().__init__()
        self.parent_main = parent_main
        self.img = None
        self.worker_thread = None
        self.init_ui()
        self.cam_result.connect(self.update_cam_display)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("📊 多算法可视化对比")
        title.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        self.btn_load = QPushButton(qta.icon('fa5s.image'), " 选择原图")
        layout.addWidget(title)
        layout.addWidget(self.btn_load)

        # 算法对比网格布局（包含标签+图片区域）
        self.grid = QGridLayout()
        self.image_labels = []  # 存储图片显示的label
        algo_names = ["原图", "Grad-CAM", "Grad-CAM++"]

        for i, name in enumerate(algo_names):
            # 1. 创建算法名称标签（加粗显示）
            name_label = QLabel(name)
            name_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("color:#2E86AB; padding:8px 0;")

            # 2. 创建图片显示区域
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("border:1px solid #ddd; padding:20px; background:#f9f9f9; min-height:280px;")
            img_label.setText("暂无图片")  # 默认提示文字

            # 3. 将标签和图片区域添加到网格
            self.grid.addWidget(name_label, 0, i)  # 第一行：算法名称
            self.grid.addWidget(img_label, 1, i)  # 第二行：图片显示
            self.image_labels.append(img_label)  # 保存图片label引用

        self.status_label = QLabel("状态：等待选择图片")
        self.status_label.setStyleSheet("color:#666; margin-top:10px")
        layout.addLayout(self.grid)
        layout.addWidget(self.status_label)

        self.btn_load.clicked.connect(self.load_and_run)

    def to_pix(self, img):
        qimg = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def update_cam_display(self, idx, img_array):
        # 更新对应位置的图片显示
        self.image_labels[idx].setPixmap(self.to_pix(img_array))
        if idx == 2:
            self.status_label.setText("✅ 所有算法计算完成")

    class CamWorker(QThread):
        result = pyqtSignal(int, np.ndarray)
        error = pyqtSignal(str)

        def __init__(self, model, layers, tensor, img, is_vit):
            super().__init__()
            self.model = model
            self.layers = layers
            self.tensor = tensor
            self.img = img
            self.is_vit = is_vit

        def run(self):
            try:
                self.result.emit(0, self.img)
                cam1 = generate_heatmap(
                    self.model, self.layers, self.tensor, self.img,
                    is_vit=self.is_vit, method="gradcam"
                )
                self.result.emit(1, cam1)
                cam2 = generate_heatmap(
                    self.model, self.layers, self.tensor, self.img,
                    is_vit=self.is_vit, method="gradcam++"
                )
                self.result.emit(2, cam2)
            except Exception as e:
                self.error.emit(f"计算异常：{str(e)}")

    def load_and_run(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return

        self.btn_load.setEnabled(False)
        self.status_label.setText("状态：加载图片中...")

        try:
            if not self.parent_main.model:
                raise Exception("请先在左侧加载模型！")

            self.img = np.array(Image.open(path).convert("RGB"))
            tensor, _ = preprocess_image(path)
            is_vit = "vit" in self.parent_main.model_name

            self.worker_thread = self.CamWorker(
                self.parent_main.model,
                self.parent_main.layers,
                tensor,
                self.img,
                is_vit
            )
            self.worker_thread.result.connect(self.cam_result)
            self.worker_thread.error.connect(self.on_error)
            self.worker_thread.finished.connect(self.thread_finish)
            self.worker_thread.start()

            self.status_label.setText("状态：正在计算热力图...")

        except Exception as e:
            self.status_label.setText(f"状态：图片加载失败 -> {str(e)}")
            self.btn_load.setEnabled(True)

    def on_error(self, err_msg):
        self.status_label.setText(f"状态：{err_msg}")
        QMessageBox.warning(self, "计算错误", err_msg)

    def thread_finish(self):
        self.btn_load.setEnabled(True)
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None
# ------------------------------
# 主进阶界面
# ------------------------------
class AdvancedWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.layers = None
        self.model_name = ""
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧菜单
        left = QWidget()
        left.setFixedWidth(220)
        left.setStyleSheet("background:#f5f5f5;")
        layout_left = QVBoxLayout(left)
        layout_left.setContentsMargins(12, 20, 12, 20)
        layout_left.setSpacing(10)

        self.menu = QListWidget()
        self.menu.addItem("  批量图片生成")
        self.menu.addItem("  视频热力图分析")
        self.menu.addItem("  摄像头实时分析")
        self.menu.addItem("  多算法效果对比")
        self.menu.setStyleSheet("""
            QListWidget::item{height:42px; padding-left:10px; border-radius:6px;}
            QListWidget::item:selected{background:#2E86AB; color:white;}
        """)

        self.model_box = QComboBox()
        self.model_box.addItems(["resnet50", "vgg16", "vit_b_16"])
        self.load_btn = QPushButton(qta.icon('fa5s.cogs'), " 加载模型")

        layout_left.addWidget(QLabel("功能菜单"))
        layout_left.addWidget(self.menu)
        layout_left.addSpacing(20)
        layout_left.addWidget(QLabel("模型设置"))
        layout_left.addWidget(self.model_box)
        layout_left.addWidget(self.load_btn)
        layout_left.addStretch()

        # 右侧页面
        self.stack = QStackedWidget()
        self.stack.addWidget(BatchPage(self))
        self.stack.addWidget(VideoPage(self))
        self.stack.addWidget(CameraPage(self))
        self.stack.addWidget(AlgoPage(self))

        main_layout.addWidget(left)
        main_layout.addWidget(self.stack)
        self.menu.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.load_btn.clicked.connect(self.load_model)

    def load_model(self):
        self.model_name = self.model_box.currentText()
        self.model, self.layers = load_model(self.model_name)
        QMessageBox.information(self, "成功", f"模型 {self.model_name} 加载完成！")