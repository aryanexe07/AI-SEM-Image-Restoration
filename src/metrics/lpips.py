"""Evaluation metrics module for LPIPS (Learned Perceptual Image Patch Similarity).

This module provides reference-verified LPIPS metric calculation supporting single-channel
and multi-channel images as NumPy arrays or PyTorch tensors. Single-channel inputs are
deterministically mapped to 3-channel RGB tensors and rescaled from [0, 1] to [-1, 1].
"""

import logging
from typing import Optional, Union

import numpy as np
import torch

from src.metrics.psnr_ssim import _validate_inputs

logger = logging.getLogger(__name__)

# Module-level model cache to avoid re-instantiating LPIPS on every batch
_LPIPS_MODEL_CACHE = {}


def _get_lpips_model(
    net_type: str = "alex", device: Union[str, torch.device] = "cpu"
):
    """Retrieve or instantiate cached LPIPS model.

    Args:
        net_type: LPIPS backbone network ('alex', 'vgg', or 'squeeze').
        device: Target device string or torch.device.

    Returns:
        Optional[torch.nn.Module]: Cached LPIPS model instance, or None if unavailable.
    """
    key = (net_type, str(device))
    if key in _LPIPS_MODEL_CACHE:
        return _LPIPS_MODEL_CACHE[key]

    try:
        import lpips

        model = lpips.LPIPS(net=net_type, verbose=False).to(device)
        model.eval()
        _LPIPS_MODEL_CACHE[key] = model
        return model
    except Exception as err:
        logger.warning(
            f"Failed to initialize LPIPS metric model (net='{net_type}', device='{device}'): {err}"
        )
        return None


def calculate_lpips(
    prediction: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    data_range: float = 1.0,
    net_type: str = "alex",
    device: str = "cpu",
) -> Optional[float]:
    """Calculate LPIPS perceptual distance between prediction and target images.

    Inputs are normalized to range [0, 1] and deterministically converted from
    single-channel (B, 1, H, W) to 3-channel (B, 3, H, W) before rescaling to [-1, 1]
    as expected by LPIPS models.

    Args:
        prediction: Predicted image array or tensor.
        target: Ground truth target image array or tensor.
        data_range: Dynamic range of image pixel values (default 1.0 for [0, 1]).
        net_type: LPIPS backbone network architecture ('alex', 'vgg', or 'squeeze').
        device: Device string or torch.device to execute LPIPS model.

    Returns:
        Optional[float]: Scalar LPIPS metric distance (lower is better, 0.0 is perfect match),
            or None if LPIPS is unavailable.
    """
    try:
        pred_np, target_np = _validate_inputs(prediction, target, data_range)
    except (ValueError, TypeError) as err:
        logger.warning(f"LPIPS input validation failed: {err}")
        return None

    model = _get_lpips_model(net_type=net_type, device=device)
    if model is None:
        return None

    # Convert normalized float32 NumPy arrays [0, data_range] -> [0, 1] PyTorch Tensors
    pred_tensor = torch.from_numpy(pred_np / data_range)
    target_tensor = torch.from_numpy(target_np / data_range)

    # Ensure 4D shape (B, C, H, W)
    if pred_tensor.ndim == 2:
        pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)
    elif pred_tensor.ndim == 3:
        pred_tensor = pred_tensor.unsqueeze(0)  # (1, C, H, W)
        target_tensor = target_tensor.unsqueeze(0)

    # Deterministically convert 1-channel grayscale to 3-channel RGB: (B, 1, H, W) -> (B, 3, H, W)
    if pred_tensor.shape[1] == 1:
        pred_tensor = pred_tensor.repeat(1, 3, 1, 1)
        target_tensor = target_tensor.repeat(1, 3, 1, 1)

    # Rescale range from [0, 1] to [-1, 1]
    pred_tensor = pred_tensor * 2.0 - 1.0
    target_tensor = target_tensor * 2.0 - 1.0

    # Move to target device
    device_obj = torch.device(device)
    pred_tensor = pred_tensor.to(device_obj)
    target_tensor = target_tensor.to(device_obj)

    with torch.no_grad():
        dist = model(pred_tensor, target_tensor)
        lpips_val = float(dist.mean().item())

    return lpips_val
