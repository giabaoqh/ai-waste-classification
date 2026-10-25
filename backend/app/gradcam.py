"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module for MobileNetV3-Large.

Implements Explainable AI (XAI) to highlight the discriminative image regions
used by the deep neural network to make waste classification decisions.
Operates purely with PyTorch, NumPy, and Pillow (no OpenCV or Matplotlib dependencies).
"""

import base64
import io
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("waste_recognition_gradcam")


def _jet_colormap(gray_values: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi mảng giá trị độ sáng 2D (0.0 đến 1.0) thành ảnh RGB 3D
    sử dụng bảng màu Jet colormap (tính toán thuần NumPy không cần OpenCV/Matplotlib).
    Jet: Xanh lam (thấp) -> Lục (trung bình) -> Vàng -> Đỏ (cao).
    """
    val = np.clip(gray_values, 0.0, 1.0)
    
    r = np.clip(1.5 - np.abs(4.0 * val - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * val - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * val - 1.0), 0.0, 1.0)
    
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return rgb.astype(np.uint8)


class GradCAM:
    """
    Lớp tính toán Grad-CAM cho mạng nơ-ron MobileNetV3.
    Mặc định gắn hook vào tầng Conv cuối cùng của backbone: model.features[-1].
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.features[-1]
        
        self.gradients = None
        self.activations = None
        self.handlers = []

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out[0] có shape: (batch_size, channels, H, W)
        self.gradients = grad_out[0].detach()

    def register_hooks(self):
        self.handlers.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def remove_hooks(self):
        for h in self.handlers:
            h.remove()
        self.handlers.clear()

    def compute_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Tính toán ma trận kích hoạt Grad-CAM 2D chuẩn hóa trong khoảng [0, 1].
        """
        self.register_hooks()
        try:
            # Bật gradient để có thể backward
            with torch.enable_grad():
                tensor_clone = input_tensor.clone().requires_grad_(True)
                logits = self.model(tensor_clone)
                
                score = logits[0, class_idx]
                self.model.zero_grad()
                score.backward()

            if self.gradients is None or self.activations is None:
                raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

            # Global Average Pooling trên gradients để tính trọng số alpha
            # gradients: (1, C, H, W) -> weights: (1, C, 1, 1)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

            # Tổng có trọng số của các bản đồ kích hoạt: (1, 1, H, W)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)  # Chỉ giữ các đặc trưng đóng góp tích cực

            # Nội suy song tuyến tính (Bilinear interpolation) lên kích thước input tensor
            input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
            cam = F.interpolate(cam, size=(input_h, input_w), mode="bilinear", align_corners=False)

            cam = cam.squeeze().cpu().numpy()
            
            # Chuẩn hóa về [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-7:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            return cam
        finally:
            self.remove_hooks()


def generate_gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    original_pil_img: Image.Image,
    alpha: float = 0.45
) -> Tuple[Optional[str], Optional[str]]:
    """
    Sinh hình ảnh trực quan hóa Grad-CAM cho ảnh đầu vào:
    - overlay_base64: Ảnh gốc có phủ lớp nhiệt Grad-CAM mờ (alpha blend)
    - heatmap_base64: Bản đồ nhiệt màu Jet độc lập
    
    Trả về định dạng Data URI: "data:image/jpeg;base64,..."
    """
    try:
        gradcam = GradCAM(model)
        cam_2d = gradcam.compute_cam(input_tensor, class_idx)

        # 1. Tạo Heatmap Jet RGB
        heatmap_rgb = _jet_colormap(cam_2d)
        heatmap_pil = Image.fromarray(heatmap_rgb)
        
        # Resize heatmap khít với kích thước ảnh gốc của người dùng
        heatmap_pil_resized = heatmap_pil.resize(original_pil_img.size, resample=Image.Resampling.BILINEAR)

        # 2. Blend với ảnh gốc
        # Chuyển ảnh gốc sang RGB nếu cần
        base_rgb = original_pil_img.convert("RGB")
        overlay_pil = Image.blend(base_rgb, heatmap_pil_resized, alpha=alpha)

        # 3. Mã hóa base64 overlay
        overlay_buf = io.BytesIO()
        overlay_pil.save(overlay_buf, format="JPEG", quality=85)
        overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(overlay_buf.getvalue()).decode("utf-8")

        # 4. Mã hóa base64 heatmap
        heatmap_buf = io.BytesIO()
        heatmap_pil_resized.save(heatmap_buf, format="PNG")
        heatmap_b64 = "data:image/png;base64," + base64.b64encode(heatmap_buf.getvalue()).decode("utf-8")

        return overlay_b64, heatmap_b64
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}", exc_info=True)
        return None, None

