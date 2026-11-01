"""알파 → 잉크 마스크. 분석용이며 출력 이미지에는 쓰지 않는다."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold


def ink_mask(
    rgba: np.ndarray,
    method: str = "threshold",
    *,
    threshold: float = 16,
    low: float = 16,
    high: float = 64,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """잉크로 볼 픽셀의 bool 마스크.

    - threshold: α ≥ threshold
    - hysteresis: α ≥ high인 픽셀과, 거기에 이어진 α ≥ low 픽셀
    blur_sigma > 0이면 판정 전에 알파를 가우시안 블러한다 (연필 질감 완화).
    """
    alpha = rgba[..., 3].astype(np.float64)
    if blur_sigma > 0:
        alpha = ndimage.gaussian_filter(alpha, blur_sigma)
    if method == "threshold":
        return alpha >= threshold
    if method == "hysteresis":
        # skimage는 엄격한 부등호(>)를 쓰므로 ≥가 되도록 경계를 살짝 낮춘다.
        return apply_hysteresis_threshold(alpha, low - 1e-6, high - 1e-6)
    raise ValueError(f"알 수 없는 마스크 방법: {method}")
