# -*- coding: utf-8 -*-
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "assets", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "assets", "output")
TEMP_DIR = os.path.join(BASE_DIR, "assets", "temp")

# ======================
# 模型保存到项目/model
# ======================
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# 设置torch模型下载路径（关键！）
os.environ['TORCH_HOME'] = MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUPPORT_MODELS = ["resnet50", "vgg16", "vit_b_16"]
DEFAULT_MODEL = "resnet50"
SUPPORT_CAM_METHODS = ["gradcam", "scorecam", "ablationcam"]
DEFAULT_CAM_METHOD = "gradcam"

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]