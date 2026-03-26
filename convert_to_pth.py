import torch
from models import MobileNetV2
from models.mobilenetv1 import MobileNetV1  # đổi nếu bạn dùng network khác

ckpt_path = "/home/dun/FACE-RECOGNITION/face-recognition/weights/mobilenetv1_040_MCP_best.ckpt"
save_path = "/home/dun/FACE-RECOGNITION/face-recognition/weights/mobilenetv1_040.pth"

#Load checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")
#Build lại backbone đúng kiến trúc
# model = MobileNetV2(embedding_dim=512, width_mult=0.25)
model = MobileNetV1(embedding_dim=512,width_mult=0.4)  # đổi nếu bạn dùng network khác

#Load trong so
model.load_state_dict(ckpt["model"])

#Lưu weights dưới dạng .pth
torch.save(model.state_dict(), save_path)

print("Saved backbone weights to:", save_path)