import cv2
import numpy as np
from PIL import Image

def cv_imread(path):
    """支持中文路径读取图片，返回 RGB numpy数组"""
    # 优先用PIL，最稳
    try:
        with Image.open(path) as pil_img:
            return np.array(pil_img.convert("RGB"))
    except Exception as e:
        pass

    # 备用：cv2.imdecode
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("❌ 读取失败：", path, e)
        return None