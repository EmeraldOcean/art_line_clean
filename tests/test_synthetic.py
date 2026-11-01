import numpy as np

from lineclean.synthetic import EXPECTED_TOPOLOGY, SIZE, generate_all


def test_every_case_has_expected_topology():
    assert EXPECTED_TOPOLOGY.keys() == generate_all().keys()


def test_all_cases_are_rgba_of_same_size():
    w, h = SIZE
    for name, arr in generate_all().items():
        assert arr.shape == (h, w, 4), name
        assert arr.dtype == np.uint8, name


def test_generation_is_deterministic():
    a, b = generate_all(), generate_all()
    assert a.keys() == b.keys()
    for name in a:
        assert np.array_equal(a[name], b[name]), name


def test_every_case_has_antialiased_ink():
    for name, arr in generate_all().items():
        alpha = arr[..., 3]
        assert (alpha > 0).sum() > 50, name
        assert ((alpha > 0) & (alpha < 255)).any(), name


def test_ink_stays_inside_canvas_border():
    # 선이 캔버스 가장자리에 잘리면 끝점 분석(M1)이 왜곡된다
    for name, arr in generate_all().items():
        alpha = arr[..., 3]
        border = np.concatenate([alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]])
        assert not border.any(), name


def test_faint_case_is_faint():
    assert generate_all()["faint_thin"][..., 3].max() <= 90
