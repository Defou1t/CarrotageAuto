"""Инструменты ручной оцифровки «своими глазами» (без скриптов проекта): обзор листа, вырезка с сеткой.

  python tools.py overview <скан> <out.png> [--w 1400] [--h 1800]
  python tools.py crop <скан> <out.png> x0 x1 y0 y1 [--zoom 2] [--step 10] [--label 50]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["overview", "crop", "size"])
ap.add_argument("img")
ap.add_argument("out", nargs="?")
ap.add_argument("box", nargs="*", type=int)
ap.add_argument("--w", type=int, default=1400)
ap.add_argument("--h", type=int, default=1800)
ap.add_argument("--zoom", type=float, default=2.0)
ap.add_argument("--step", type=int, default=10)
ap.add_argument("--label", type=int, default=50)
a = ap.parse_args()
im = Image.open(a.img).convert("RGB")
if a.cmd == "size":
    print(im.size); sys.exit()
try:
    font = ImageFont.truetype("arial.ttf", 13)
except Exception:
    font = ImageFont.load_default()
if a.cmd == "overview":
    W, H = im.size
    sc = min(a.w / W, a.h / H)
    ov = im.resize((max(1, int(W * sc)), max(1, int(H * sc))), Image.LANCZOS)
    pad = 60
    can = Image.new("RGB", (ov.width + pad, ov.height + 20), "white")
    can.paste(ov, (pad, 10))
    dr = ImageDraw.Draw(can)
    for y in range(0, H, 1000):
        yy = 10 + int(y * sc)
        dr.line([(pad - 8, yy), (pad, yy)], fill=(255, 0, 0))
        dr.text((2, yy - 6), str(y), fill=(255, 0, 0), font=font)
    for x in range(0, W, 500):
        xx = pad + int(x * sc)
        dr.line([(xx, 0), (xx, 8)], fill=(0, 0, 255))
        dr.text((xx + 2, 0), str(x), fill=(0, 0, 255), font=font)
    can.save(a.out)
    print(f"размер {W}×{H}, масштаб {sc:.4f}")
elif a.cmd == "crop":
    x0, x1, y0, y1 = a.box
    cr = im.crop((x0, y0, x1, y1))
    z = a.zoom
    cr = cr.resize((int(cr.width * z), int(cr.height * z)), Image.NEAREST)
    padl, padt = 46, 18
    can = Image.new("RGB", (cr.width + padl, cr.height + padt), "white")
    can.paste(cr, (padl, padt))
    dr = ImageDraw.Draw(can, "RGBA")
    for x in range((x0 // a.step + 1) * a.step, x1, a.step):
        xx = padl + int((x - x0) * z)
        big = x % a.label == 0
        dr.line([(xx, padt), (xx, can.height)], fill=(0, 120, 255, 110 if big else 40), width=1)
        if big:
            dr.text((xx - 10, 2), str(x), fill=(0, 0, 200), font=font)
    for y in range((y0 // a.step + 1) * a.step, y1, a.step):
        yy = padt + int((y - y0) * z)
        big = y % a.label == 0
        dr.line([(padl, yy), (can.width, yy)], fill=(255, 60, 0, 110 if big else 40), width=1)
        if big:
            dr.text((1, yy - 7), str(y), fill=(200, 0, 0), font=font)
    can.save(a.out)
    print(f"вырезка x {x0}–{x1}, y {y0}–{y1}, ×{z} → {can.size}")
