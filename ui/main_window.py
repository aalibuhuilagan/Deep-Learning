from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont
import qtawesome as qta

from ui.logic_basic import BasicWidget  # 原有功能不动
from ui.logic_advanced import AdvancedWidget  # 新增进阶功能

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grad_CAM AI可视化工具")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(qta.icon('fa5s.brain', color='#2E86AB'))

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 标题栏
        title = QLabel("Grad-CAM 可视化分析系统")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#2E86AB; padding:8px;")
        main_layout.addWidget(title)

        # 选项卡（基础功能 / 进阶功能）
        tab = QTabWidget()
        tab.setFont(QFont("Microsoft YaHei", 11))

        # 1. 基础功能（完全不动）
        self.basic_tab = BasicWidget()
        tab.addTab(self.basic_tab, qta.icon('fa5s.image'), " 基础功能（单张图片）")

        # 2. 进阶功能（新增：视频、批量、算法对比）
        self.advanced_tab = AdvancedWidget()
        tab.addTab(self.advanced_tab, qta.icon('fa5s.tools'), " 进阶功能（视频/批量/算法）")

        tab.setStyleSheet("""
            QTabWidget::pane {border:1px solid #ddd; border-radius:8px;}
            QTabBar::tab {padding:10px 20px; border-radius:6px;}
            QTabBar::tab:selected {background:#2E86AB; color:white;}
        """)

        main_layout.addWidget(tab)