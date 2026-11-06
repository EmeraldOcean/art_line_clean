"""알파 → 잉크 마스크. 분석용이며 출력 이미지에는 쓰지 않는다."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold

from lineclean.config import GraphParams


def ink_mask(
    rgba: np.ndarray,
    method: str = "threshold",
    *,
    threshold: float = 16,
    relative: float = 0.0,
    window_px: int = 7,
    low: float = 16,
    high: float = 64,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """잉크로 볼 픽셀의 bool 마스크.

    - threshold: α ≥ threshold. relative > 0이면 추가로 α ≥ relative × (주변 window_px 안 최대 알파).
      진한 선 옆의 옅은 번짐이 가까운 두 선을 이어 버리는 것을 막는다 (D-19)
    - hysteresis: α ≥ high인 픽셀과, 거기에 이어진 α ≥ low 픽셀
    blur_sigma > 0이면 판정 전에 알파를 가우시안 블러한다 (연필 질감 완화).
    """
    alpha = rgba[..., 3].astype(np.float64)
    if blur_sigma > 0:
        alpha = ndimage.gaussian_filter(alpha, blur_sigma)
    if method == "threshold":
        if relative > 0:
            local_max = ndimage.maximum_filter(alpha, size=window_px)
            return alpha >= np.maximum(threshold, relative * local_max)
        return alpha >= threshold
    if method == "hysteresis":
        # skimage는 엄격한 부등호(>)를 쓰므로 ≥가 되도록 경계를 살짝 낮춘다.
        return apply_hysteresis_threshold(alpha, low - 1e-6, high - 1e-6)
    raise ValueError(f"알 수 없는 마스크 방법: {method}")


def ink(rgba: np.ndarray, gp: GraphParams | None = None) -> np.ndarray:
    """설정의 잉크 판정 규칙([graph])으로 만든 마스크. 모든 단계와 검사가 이 함수 하나를 쓴다."""
    gp = gp or GraphParams()
    return ink_mask(rgba, threshold=gp.ink_threshold, relative=gp.ink_relative, window_px=gp.ink_relative_window_px)
