import cv2
import os
from mtcnn import MTCNN
from pathlib import Path
import numpy as np
from tqdm import tqdm

detector = MTCNN()

MASK_PATH = "/home/dun/face-recognition/images/surgical_blue.png"  # đặt file mask PNG transparent ở đây

def overlay_mask(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không đọc được ảnh: {img_path}")
        return
    
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print("Không load được mask!")
        return

    try:
        faces = detector.detect_faces(img)
    except Exception as e:
        print(f"MTCNN crash ở {img_path}: {e} → lưu nguyên ảnh")
        cv2.imwrite(out_path, img)
        return

    if not faces:
        print(f"Không detect mặt trong {img_path} → lưu nguyên ảnh")
        cv2.imwrite(out_path, img)
        return

    overlaid = False
    for face in faces:
        x, y, w, h = face['box']
        
        # Bỏ qua box quá nhỏ (thường gây empty patches ở ONet)
        if w < 40 or h < 40:  # tăng threshold để an toàn hơn
            continue
        
        # Scale mask (điều chỉnh tỷ lệ cho tự nhiên)
        mask_resized = cv2.resize(mask, (int(w * 1.3), int(h * 0.7)))
        
        mx = x + (w - mask_resized.shape[1]) // 2
        my = y + int(h * 0.4)  # dịch xuống để che mũi/miệng
        
        mh, mw = mask_resized.shape[:2]
        
        # Check vượt biên (đã có, nhưng thêm margin an toàn)
        if mx < 0 or my < 0 or mx + mw > img.shape[1] or my + mh > img.shape[0]:
            print(f"Mask vượt biên ở {img_path}, skip face này")
            continue
        
        alpha = mask_resized[:, :, 3] / 255.0
        for c in range(3):
            img[my:my+mh, mx:mx+mw, c] = alpha * mask_resized[:, :, c] + \
                                         (1 - alpha) * img[my:my+mh, mx:mx+mw, c]
        
        overlaid = True

    if not overlaid:
        print(f"Không overlay được mask nào ở {img_path} → lưu nguyên")
    
    cv2.imwrite(out_path, img)

# Run
input_dir = "/home/dun/face-recognition/data/val/lfw_112x112"
output_dir = "/home/dun/face-recognition/data/val/lfw_masked_112x112"

Path(output_dir).mkdir(parents=True, exist_ok=True)

bmp_files = []
for root, _, files in os.walk(input_dir):
    for f in files:
        if f.lower().endswith('.bmp'):
            bmp_files.append(os.path.join(root, f))

for bmp_path in tqdm(bmp_files, desc="Adding mask"):
    rel = os.path.relpath(bmp_path, input_dir)
    out_path = os.path.join(output_dir, rel.replace('.bmp', '.bmp'))  # giữ .bmp hoặc đổi .jpg
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    overlay_mask(bmp_path, out_path)
    try:
        overlay_mask(bmp_path, out_path)
    except Exception as e:
        print(f"Lỗi tổng ở {bmp_path}: {e}")
        # Fallback: copy ảnh gốc
        cv2.imwrite(out_path, cv2.imread(bmp_path))

print("Xong! Folder lfw_masked_112x112 sẵn sàng.")