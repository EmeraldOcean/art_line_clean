import numpy as np

from lineclean.compare import (
    ADDED,
    LABEL_HEIGHT,
    PANEL_GAP,
    RECOLORED,
    REMOVED,
    diff_image,
    make_sheet,
    nongeneration_violations,
)


def _dot_image(h=20, w=30, y=10, x=10, alpha=200):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[y, x, 3] = alpha
    return arr


def test_identical_images_have_empty_diff():
    a = _dot_image()
    assert not diff_image(a, a.copy()).any()


def test_diff_colors():
    before = _dot_image()
    before[5, 5] = (10, 10, 10, 100)

    after = before.copy()
    after[2, 2, 3] = 50  # 추가
    after[10, 10, 3] = 0  # 제거
    after[5, 5, :3] = (200, 0, 0)  # 색만 변경

    d = diff_image(before, after)
    assert tuple(d[2, 2, :3]) == ADDED and d[2, 2, 3] > 0
    assert tuple(d[10, 10, :3]) == REMOVED and d[10, 10, 3] > 0
    assert tuple(d[5, 5, :3]) == RECOLORED and d[5, 5, 3] == 255
    d[[2, 10, 5], [2, 10, 5]] = 0
    assert not d.any()


def test_small_diff_is_still_visible():
    before = _dot_image(alpha=100)
    after = before.copy()
    after[10, 10, 3] = 101
    assert diff_image(before, after)[10, 10, 3] >= 64


def test_full_alpha_diff_does_not_overflow():
    before = _dot_image(alpha=0)
    after = _dot_image(alpha=255)
    assert diff_image(before, after)[10, 10, 3] == 255


def test_nongeneration_identity_passes():
    a = _dot_image()
    assert not nongeneration_violations(a, a.copy()).any()


def test_nongeneration_within_d_passes_and_far_fails():
    before = _dot_image()
    near = before.copy()
    near[10, 12, 3] = 255  # 거리 2
    assert not nongeneration_violations(before, near, d=2).any()

    far = before.copy()
    far[10, 20, 3] = 255  # 거리 10
    v = nongeneration_violations(before, far, d=2)
    assert v.sum() == 1 and v[10, 20]


def test_nongeneration_ignores_removed_ink():
    before = _dot_image()
    assert not nongeneration_violations(before, np.zeros_like(before)).any()


def test_sheet_size():
    a = _dot_image(h=20, w=30)
    sheet = make_sheet(a, a.copy())
    assert sheet.size == (3 * 30 + 2 * PANEL_GAP, 20 + LABEL_HEIGHT)
    assert sheet.mode == "RGBA"
