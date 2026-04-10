import numpy as np
import cv2

# Paramètres
rows = 7       # 7 cases → 6 coins verticaux
cols = 10      # 10 cases → 9 coins horizontaux

DPI = 300
MM_PER_CASE = 24

# 1 inch = 25.4 mm → pixels par case
square_px = round(DPI * MM_PER_CASE / 25.4)  # = 283 px à 300 dpi

img_height = rows * square_px
img_width  = cols * square_px

img = np.ones((img_height, img_width), dtype=np.uint8) * 255

for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            img[i*square_px:(i+1)*square_px,
                j*square_px:(j+1)*square_px] = 0

# Bordure blanche = 1 case
border = square_px
img = cv2.copyMakeBorder(img, border, border, border, border,
                         cv2.BORDER_CONSTANT, value=255)

# Sauvegarder avec PIL pour encoder les DPI dans les métadonnées
from PIL import Image
pil_img = Image.fromarray(img)
pil_img.save("chessboard.png", dpi=(DPI, DPI))

print(f"Case: {square_px}px @ {DPI}dpi = {square_px/DPI*25.4:.1f}mm")
print(f"Damier total (sans bordure): {cols*square_px/DPI*25.4:.1f} x {rows*square_px/DPI*25.4:.1f} mm")