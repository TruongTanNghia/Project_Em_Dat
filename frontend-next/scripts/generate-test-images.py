"""Generate fake-but-plausible test images for spine + breast modules.

These images LOOK like medical scans but are entirely synthetic — for
UI/demo testing only. Use real datasets (TotalSegmentator examples,
BUSI, CBIS-DDSM) for actual model evaluation.

Run:
    python frontend-next/scripts/generate-test-images.py

Output: 3 PNG files in <project_root>/test_images/
"""
from pathlib import Path
import sys

try:
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
except ImportError:
    print("Need Pillow + numpy:  pip install Pillow numpy")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # frontend-next/scripts → root
OUT_DIR = PROJECT_ROOT / "test_images"
OUT_DIR.mkdir(exist_ok=True)


def make_spine_xray():
    """Fake lateral spine X-ray with 24 vertebrae stacked on a slight curve."""
    W, H = 512, 1280
    arr = np.random.randint(12, 30, (H, W), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    cx = W // 2

    # Soft body silhouette
    body_arr = np.array(img, dtype=np.int16)
    for y in range(H):
        x_off = int(35 * np.sin(y / H * np.pi))
        left = cx - 130 + x_off
        right = cx + 130 + x_off
        body_arr[y, max(0, left):min(W, right)] += 18
    body_arr = np.clip(body_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(body_arr, mode="RGB")
    draw = ImageDraw.Draw(img)

    # 24 vertebrae + sacrum along curved spine
    levels = [
        ("C", 7,  14, 175),  # cervical — small
        ("T", 12, 22, 195),  # thoracic — medium
        ("L", 5,  32, 210),  # lumbar — large
    ]
    y = 70
    for level, count, width, brightness in levels:
        v_height = 30 if level == "C" else (38 if level == "T" else 48)
        for i in range(count):
            x_off = int(28 * np.sin(y / H * np.pi))
            x_center = cx + x_off
            draw.ellipse(
                [x_center - width, y, x_center + width, y + v_height],
                fill=(brightness, brightness, brightness),
                outline=(brightness + 25, brightness + 25, brightness + 25),
                width=1,
            )
            # Spinous process behind
            draw.polygon(
                [
                    (x_center + width, y + 5),
                    (x_center + width + 22, y + v_height // 2),
                    (x_center + width, y + v_height - 5),
                ],
                fill=(brightness - 25, brightness - 25, brightness - 25),
            )
            y += v_height + 5  # disc space
        y += 6  # gap between levels

    # Sacrum at bottom — triangle-ish
    draw.polygon(
        [(cx - 40, y), (cx + 40, y), (cx + 10, y + 80), (cx - 10, y + 80)],
        fill=(195, 195, 195),
    )

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.9))

    # Add caption-ish corner labels
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "DEMO · SPINE X-RAY", fill=(150, 150, 150))
    draw.text((W - 70, H - 25), "L | R", fill=(150, 150, 150))

    path = OUT_DIR / "test_spine_xray.png"
    img.save(path, "PNG", optimize=True)
    print(f"[OK] {path.name}  ({path.stat().st_size // 1024} KB)")


def make_mammography():
    """Fake mammography (CC view) with a soft hypodense mass in UOQ."""
    W, H = 600, 800
    arr = np.zeros((H, W, 3), dtype=np.uint8)

    # Breast tissue: soft gradient ellipse on dark BG
    cx, cy = int(W * 0.45), H // 2
    yy, xx = np.mgrid[0:H, 0:W]
    dx = (xx - cx) / (W * 0.45)
    dy = (yy - cy) / (H * 0.42)
    d = np.sqrt(dx ** 2 + dy ** 2)
    tissue = np.where(d < 1, 200 * (1 - d ** 1.4), 0)

    # Slight glandular texture pattern
    texture = np.random.randn(H, W) * 8
    tissue = tissue + texture
    tissue = np.clip(tissue, 0, 255)

    for c in range(3):
        arr[:, :, c] = tissue

    # Add a suspicious mass — irregular bright spot in upper-outer quadrant
    mass_cx, mass_cy = cx + 90, cy - 110
    for r in range(80):
        a = np.random.uniform(0, 2 * np.pi, 12)
        boost = 50 - r * 0.4
        for theta in a:
            mx = mass_cx + int(r * np.cos(theta) + np.random.randn() * 5)
            my = mass_cy + int(r * np.sin(theta) + np.random.randn() * 5)
            if 0 <= mx < W and 0 <= my < H:
                arr[my, mx] = np.clip(arr[my, mx].astype(int) + boost, 0, 255)

    img = Image.fromarray(arr, mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=1.4))

    # Corner labels
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "DEMO · MAMMOGRAPHY · L CC", fill=(180, 180, 180))
    draw.text((W - 80, H - 25), "100 kVp · 200 mAs", fill=(150, 150, 150))

    path = OUT_DIR / "test_mammography.png"
    img.save(path, "PNG", optimize=True)
    print(f"[OK] {path.name}  ({path.stat().st_size // 1024} KB)")


def make_breast_ultrasound():
    """Fake breast ultrasound (B-mode) with a hypoechoic spiculated mass."""
    W, H = 600, 450
    # Trapezoidal US cone — speckle noise tissue
    speckle = np.random.randint(50, 130, (H, W))
    # Add some horizontal streaking for tissue planes
    for i in range(8):
        y_line = np.random.randint(50, H - 50)
        speckle[y_line:y_line + 2] = np.clip(speckle[y_line:y_line + 2] + 30, 0, 200)

    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(3):
        arr[:, :, c] = speckle

    # Mask cone shape (trapezoid, darker outside)
    yy, xx = np.mgrid[0:H, 0:W]
    cone_left = W / 2 - W * 0.4 - yy * (W * 0.1) / H
    cone_right = W / 2 + W * 0.4 + yy * (W * 0.1) / H
    in_cone = (xx > cone_left) & (xx < cone_right)
    for c in range(3):
        arr[:, :, c] = np.where(in_cone, arr[:, :, c], arr[:, :, c] // 3)

    # Hypoechoic mass — dark spot, irregular
    mass_cx, mass_cy = W // 2 + 30, H // 2 - 20
    yy_d = (yy - mass_cy) / 45
    xx_d = (xx - mass_cx) / 55
    d_mass = np.sqrt(xx_d ** 2 + yy_d ** 2)
    mass_factor = np.where(d_mass < 1, 0.25 + d_mass * 0.4, 1.0)
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] * mass_factor).astype(np.uint8)

    # Acoustic shadow below mass
    shadow_band = slice(mass_cy + 40, H)
    arr[shadow_band, max(0, mass_cx - 60):min(W, mass_cx + 60)] = \
        (arr[shadow_band, max(0, mass_cx - 60):min(W, mass_cx + 60)].astype(int) * 0.45).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    # Corner labels
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "DEMO · US BREAST · 12 MHz", fill=(180, 180, 180))
    draw.text((W - 70, H - 25), "10 cm", fill=(150, 150, 150))

    path = OUT_DIR / "test_breast_ultrasound.png"
    img.save(path, "PNG", optimize=True)
    print(f"[OK] {path.name}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    print(f"[OUT] {OUT_DIR}")
    make_spine_xray()
    make_mammography()
    make_breast_ultrasound()
    print("\nDone. Upload these PNG files into the spine/breast tabs to test.")
