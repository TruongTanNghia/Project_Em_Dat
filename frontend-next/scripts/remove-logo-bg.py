"""Convert the white-background ADA Group logo (JPG) into a PNG with
transparent background. Uses a soft chroma-key on the white channel
so the edges anti-alias smoothly instead of going jagged.

Usage:
    python remove-logo-bg.py
"""
from pathlib import Path
import sys

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Need Pillow + numpy. Install:  pip install Pillow numpy")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # frontend-next/
SRC = PROJECT_ROOT / "public" / "img" / "logo.jpg"
DST = PROJECT_ROOT / "public" / "img" / "logo.png"

if not SRC.exists():
    print(f"[ERR] Source logo not found at {SRC}")
    sys.exit(1)

print(f"[1/4] Loading {SRC} ({SRC.stat().st_size / 1024:.1f} KB)")
img = Image.open(SRC).convert("RGB")
print(f"      Dimensions: {img.size[0]}x{img.size[1]}")

print(f"[2/4] Converting to RGBA + computing alpha mask")
data = np.array(img).astype(np.int16)  # int16 so we can subtract without overflow
r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]

# Distance from pure white per pixel — uses the channel furthest from 255.
# Pure white (255,255,255) → 0. Pure colored pixel → 255.
min_channel = np.minimum(np.minimum(r, g), b)
distance_from_white = 255 - min_channel

# Soft chroma-key: scale distance into alpha. Multiplier 4× makes the
# transition tight (anti-aliased edges keep partial opacity but
# off-white BG goes fully transparent quickly).
alpha = np.clip(distance_from_white.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)

print(f"      Transparent pixels: {(alpha == 0).sum():,}")
print(f"      Opaque pixels:      {(alpha == 255).sum():,}")
print(f"      Edge (partial):     {((alpha > 0) & (alpha < 255)).sum():,}")

print(f"[3/4] Composing RGBA PNG")
out = np.dstack([
    np.clip(r, 0, 255).astype(np.uint8),
    np.clip(g, 0, 255).astype(np.uint8),
    np.clip(b, 0, 255).astype(np.uint8),
    alpha,
])
result = Image.fromarray(out, mode="RGBA")

print(f"[4/4] Saving to {DST}")
# optimize=True compresses better, but keep mode RGBA
result.save(DST, "PNG", optimize=True)
final_kb = DST.stat().st_size / 1024
print(f"      Final size: {final_kb:.1f} KB")
print("Done.")
