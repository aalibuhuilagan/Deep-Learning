import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from config import IMAGE_SIZE, MEAN, STD, DEVICE

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

def np_to_qt(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    tensor = get_transforms()(img).unsqueeze(0).to(DEVICE)
    return tensor, img_np