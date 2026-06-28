r"""QC по глубинам эксперта: где пропуски и почему. dark-порог vs prob-модель vs реальный пробел."""
import sys
import numpy as np, cv2
sys.path.insert(0, r"F:\nds\Auto")
import auto.imaging as im, auto.meta as M, auto.frame as F, auto.prob as P
from auto.config import DEFAULT
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
SC = r"F:\nds\Auto\Yatskivska_1_MK_3130_3760_200_D1.jpg"; NL = SC.replace('.jpg', '.nlgx')
NPY = r"C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds-Auto\5bbdb336-3b32-4e7a-b006-ae7aeb0160e5\scratchpad\prob\Yatskivska_1_MK_3130_3760_200_D1_prob.npy"
OUT = r"C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds-Auto\5bbdb336-3b32-4e7a-b006-ae7aeb0160e5\scratchpad"
p = DEFAULT.cv
rgb = im.load_rgb(SC); H, W = rgb.shape[:2]
m = M.parse_filename(SC, r"F:\nds\Auto\mnemonics.json")
fr = F.frame_from_nlgx(NL, m, rgb=rgb)
prob = P.prob_from_npy(NPY)(rgb)
ppm = fr.px_per_m

def yof(d):
    return int(fr.top_y + (d - fr.top_depth) * ppm)

print(f"top_y={fr.top_y} top_depth={fr.top_depth} px/m={ppm:.2f} bottom_y={fr.bottom_y} bot_depth={fr.bottom_depth}")
dk = im.dark_mask(rgb, p) & ~im.structure_mask(rgb, p)
zones = [(3190, 3196, "наложение/пропуск"), (3220, 3224, "пропуск"), (3308, 3314, "бледная MPZ"),
         (3368, 3372, "пик MPZ"), (3740, 3744, "MPZ+MGZ"), (3748, 3752, "хвост")]
try:
    FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 24)
except Exception:
    FONT = ImageFont.load_default()
for d0, d1, label in zones:
    y0, y1 = yof(d0) - 40, yof(d1) + 40
    XL, XR = 70, 300
    win_dk = dk[y0:y1, XL:XR]
    win_pr = prob[y0:y1, XL:XR]
    # сколько строк имеют dark? сколько — prob>0.5 но НЕ dark (бледное, что ловит ТОЛЬКО модель)?
    rows_dk = np.mean(win_dk.any(1))
    faint_only = (win_pr >= 0.5) & ~win_dk
    rows_faint = np.mean(faint_only.any(1))
    print(f"{d0}-{d1} ({label}) y[{y0}..{y1}]: строк с dark={rows_dk*100:.0f}% | "
          f"строк с бледным(prob>0.5,не dark)={rows_faint*100:.0f}% | maxProb={win_pr.max():.2f}")
    # кроп: raw + разметка (dark=зелёным контур, бледное-prob=оранжевым)
    crop = rgb[y0:y1, XL:XR].copy()
    crop[win_dk] = (0, 170, 0)                       # что видит порог темноты
    crop[faint_only] = (255, 140, 0)                 # что добавляет МОДЕЛЬ (бледное)
    im_c = Image.fromarray(crop).resize(((XR - XL) * 3, (y1 - y0) * 3), Image.NEAREST)
    ImageDraw.Draw(im_c).text((4, 2), f"{d0}-{d1} {label}  зел=dark оранж=модель", fill=(0, 0, 0), font=FONT)
    im_c.save(f"{OUT}/QC_{d0}.png")
print("сохранены QC_*.png")
