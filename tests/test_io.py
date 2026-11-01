import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from lineclean.io import (
    PNG_SIGNATURE,
    PROTECTED_INPUT_DIR,
    OutputPathError,
    UnsupportedImageError,
    file_sha256,
    load_rgba,
    png_bit_depth,
    save_rgba,
)


@pytest.fixture
def random_rgba():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (37, 53, 4), dtype=np.uint8)
    arr[:10, :, 3] = 0  # 완전 투명 픽셀에도 RGB 값이 남아 있어야 한다
    return arr


def test_rgba_roundtrip_is_bit_exact(tmp_path, random_rgba):
    path = save_rgba(random_rgba, tmp_path / "a.png")
    loaded, info = load_rgba(path)
    assert loaded.shape == random_rgba.shape
    assert np.array_equal(loaded, random_rgba)
    assert info.source_mode == "RGBA"
    assert not info.luminance_to_alpha


def test_rgb_under_zero_alpha_is_preserved(tmp_path, random_rgba):
    loaded, _ = load_rgba(save_rgba(random_rgba, tmp_path / "a.png"))
    assert np.array_equal(loaded[:10, :, :3], random_rgba[:10, :, :3])


def test_refuses_overwrite(tmp_path, random_rgba):
    path = save_rgba(random_rgba, tmp_path / "a.png")
    before = file_sha256(path)
    with pytest.raises(OutputPathError):
        save_rgba(np.zeros_like(random_rgba), path)
    assert file_sha256(path) == before


def test_refuses_protected_dir(tmp_path, random_rgba):
    protected = tmp_path / "in"
    with pytest.raises(OutputPathError):
        save_rgba(random_rgba, protected / "sub" / "a.png", protected_dirs=[protected])
    assert not protected.exists()


def test_refuses_project_input_dir(random_rgba):
    with pytest.raises(OutputPathError):
        save_rgba(random_rgba, PROTECTED_INPUT_DIR / "should_not_exist.png")
    assert not (PROTECTED_INPUT_DIR / "should_not_exist.png").exists()


def test_load_does_not_modify_input(tmp_path, random_rgba):
    path = save_rgba(random_rgba, tmp_path / "a.png")
    before = file_sha256(path)
    load_rgba(path)
    assert file_sha256(path) == before


def test_la_converts_losslessly(tmp_path):
    rng = np.random.default_rng(1)
    la = rng.integers(0, 256, (20, 30, 2), dtype=np.uint8)
    path = tmp_path / "la.png"
    Image.fromarray(la, "LA").save(path)
    rgba, info = load_rgba(path)
    assert not info.luminance_to_alpha
    assert np.array_equal(rgba[..., 3], la[..., 1])
    assert np.array_equal(rgba[..., 0], la[..., 0])


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_luminance_to_alpha_reproduces_brightness_on_white(tmp_path, mode):
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, (20, 30), dtype=np.uint8)
    img = Image.fromarray(gray)
    if mode == "RGB":
        img = img.convert("RGB")
    path = tmp_path / f"{mode}.png"
    img.save(path)

    rgba, info = load_rgba(path)
    assert info.luminance_to_alpha
    assert info.color_discarded == (mode == "RGB")
    assert np.all(rgba[..., :3] == 0)
    # 흰 배경 위에 검정 잉크를 합성: 255 * (1 - a/255) = 255 - a
    composited = 255 - rgba[..., 3].astype(int)
    assert np.abs(composited - gray.astype(int)).max() <= 1


def test_palette_with_transparency_keeps_alpha(tmp_path):
    img = Image.new("P", (4, 4), 0)
    img.putpalette([0, 0, 0, 255, 0, 0])
    img.putpixel((1, 1), 1)
    img.info["transparency"] = 0
    path = tmp_path / "p.png"
    img.save(path, transparency=0)
    rgba, info = load_rgba(path)
    assert not info.luminance_to_alpha
    assert rgba[1, 1].tolist() == [255, 0, 0, 255]
    assert rgba[0, 0, 3] == 0


def test_rejects_16bit(tmp_path):
    path = tmp_path / "i16.png"
    Image.fromarray(np.full((4, 4), 40000, dtype=np.uint16)).save(path)
    with pytest.raises(UnsupportedImageError):
        load_rgba(path)


def _write_png16_rgba(path, rgba16):
    """Pillow는 16-bit RGBA PNG를 쓸 수 없으므로 직접 만든다."""

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    h, w, _ = rgba16.shape
    raw = b"".join(b"\x00" + row.astype(">u2").tobytes() for row in rgba16)
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 6, 0, 0, 0)  # 16-bit, 색 유형 6 = RGBA
    path.write_bytes(PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_rejects_16bit_rgba(tmp_path):
    path = tmp_path / "rgba16.png"
    _write_png16_rgba(path, np.array([[[0, 0, 0, 0x8000], [0, 0, 0, 0x80FF]]], dtype=np.uint16))
    assert png_bit_depth(path) == 16
    with pytest.raises(UnsupportedImageError):
        load_rgba(path)


def test_bit_depth_of_8bit_png(tmp_path, random_rgba):
    assert png_bit_depth(save_rgba(random_rgba, tmp_path / "a.png")) == 8


def test_bit_depth_of_non_png(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"not a png")
    assert png_bit_depth(path) is None


def test_rejects_wrong_array(tmp_path):
    with pytest.raises(ValueError):
        save_rgba(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "x.png")
