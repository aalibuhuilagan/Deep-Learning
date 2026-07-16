import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
import warnings
warnings.filterwarnings("ignore")

class CAMEngine:
    @staticmethod
    def vit_reshape_transform(tensor):
        tensor = tensor[:, 1:, :]
        B, N, C = tensor.shape
        H = W = int(np.sqrt(N))
        return tensor.permute(0, 2, 1).view(B, C, H, W)

def generate_heatmap(model, target_layers, image_tensor, original_img_np, is_vit=False, method="gradcam"):
    was_training = model.training
    model.train()

    try:
        h_ori, w_ori = original_img_np.shape[:2]

        if method == "gradcam":
            cam_cls = GradCAM
        elif method == "gradcam++":
            cam_cls = GradCAMPlusPlus
        else:
            cam_cls = GradCAM

        cam = cam_cls(
            model=model,
            target_layers=target_layers,
            reshape_transform=CAMEngine.vit_reshape_transform if is_vit else None
        )
        grayscale_cam = cam(image_tensor)[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (w_ori, h_ori))

        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        alpha = 0.4
        overlay = cv2.addWeighted(original_img_np, 1 - alpha, heatmap, alpha, 0)

        return overlay
    except Exception as e:
        print(f"CAM错误: {e}")
        return original_img_np.copy()
    finally:
        if not was_training:
            model.eval()
        if 'cam' in locals():
            del cam
        if torch.cuda.is_available():
            torch.cuda.empty_cache()