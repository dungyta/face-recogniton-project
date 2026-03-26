# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
from utils.face_utils import face_alignment
from insightface.app import FaceAnalysis
from models import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)

from utils.face_utils import compute_similarity

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0,det_size=(160, 160))  # max_num=1 để chỉ lấy khuôn mặt chính

def get_network(model_name: str) -> torch.nn.Module:
    """
    Returns the appropriate model based on the provided model name.

    Args:
        model_name (str): Name of the model architecture.

    Returns:
        torch.nn.Module: The selected deep learning model.
    """
    models = {
        "sphere20": sphere20(embedding_dim=512, in_channels=3),
        "sphere36": sphere36(embedding_dim=512, in_channels=3),
        "sphere64": sphere64(embedding_dim=512, in_channels=3),
        "mobilenetv1": MobileNetV1(embedding_dim=512),
        "mobilenetv1_050": MobileNetV1(embedding_dim=512, width_mult=0.5),
        "mobilenetv2": MobileNetV2(embedding_dim=512),
        "mobilenetv3_small": mobilenet_v3_small(embedding_dim=512),
        "mobilenetv3_large": mobilenet_v3_large(embedding_dim=512),
        "mobilenetv2_025": MobileNetV2(embedding_dim=512, width_mult=0.25),
        "mobilenetv1_040": MobileNetV1(embedding_dim=512, width_mult=0.4),
        
    }

    if model_name not in models:
        raise ValueError(f"Unsupported network '{model_name}'! Available options: {list(models.keys())}")

    return models[model_name]


def load_model(model_name: str, model_path: str, device: torch.device = None) -> torch.nn.Module:  
    """  
    Loads a deep learning model with pre-trained weights.  
    Supports both training checkpoints (.ckpt) and plain state_dict files (.pth).  
    """  
    model = get_network(model_name)  
  
    try:  
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)  
        # Training checkpoints contain metadata and a 'model' key  
        if isinstance(checkpoint, dict) and 'model' in checkpoint:  
            state_dict = checkpoint['model']  
        else:  
            state_dict = checkpoint  
        model.load_state_dict(state_dict)  
        model.to(device).eval()  
    except Exception as e:  
        raise RuntimeError(f"Error loading model '{model_name}' from {model_path}: {e}")  
  
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),  # thêm dòng này
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
def extract_features(model, device, img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    faces = face_app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected")

    landmark = faces[0].kps  # 5 điểm landmark

    aligned = face_alignment(img, landmark, image_size=112)

    aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(aligned)

    transform = get_transform()
    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(tensor)
        features = torch.nn.functional.normalize(features, dim=1)  # rất quan trọng
        features = features.squeeze().cpu().numpy()

    return features

# def extract_features(model, device, img_path: str) -> np.ndarray:
#     """
#     Extracts face features from an image.
#     """
#     transform = get_transform()

#     try:
#         img = Image.open(img_path).convert("RGB")
#     except Exception as e:
#         raise FileNotFoundError(f"Error opening image {img_path}: {e}")

#     tensor = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         features = model(tensor).squeeze().cpu().numpy()
#     return features


def compare_faces(model, device, img1_path: str, img2_path: str, threshold: float = 0.35) -> tuple[float, bool]:
    """
    Compares two face images and determines if they belong to the same person.
    """
    feat1 = extract_features(model, device, img1_path)
    feat2 = extract_features(model, device, img2_path)

    similarity = compute_similarity(feat1, feat2)
    is_same = similarity > threshold

    return similarity, is_same


if __name__ == "__main__":
    # Example usage with model selection
    # model_name = "mobilenetv1_050"  # Change this to select different models
    model_name = "mobilenetv1_040"  # Change this to select different models
    model_path = "/home/dun/FACE-RECOGNITION/face-recognition/weights/mobilenetv1_040.pth"
    threshold = 0.27

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_name, model_path, device)

    # Compare faces
    similarity, is_same = compare_faces(
        model, device,
        img1_path="/home/dun/Desktop/test_anh/hyun_bin.png",
        img2_path="/home/dun/FACE-RECOGNITION/NNE_V1.0.05/nne_compile_tool/DataSet/database_image/Ji Chang Wook/9.jpg",
        threshold=threshold
    )

    print(f"Similarity: {similarity:.4f} - {'same' if is_same else 'different'} (Threshold: {threshold})")
