import os
from pathlib import Path
from PIL import Image
from torchvision import transforms


# ====== CONFIG ======
INPUT_DIR = "assets"
OUTPUT_DIR = "assets_processed"
IMAGE_SIZE = 112  # match training size


# ====== TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])


def preprocess_image(input_path: Path, output_path: Path):
    try:
        img = Image.open(input_path).convert("RGB")
        img = transform(img)
        img.save(output_path)
        print(f"Processed: {input_path.name}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def preprocess_folder():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    for file in input_dir.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            output_path = output_dir / file.name
            preprocess_image(file, output_path)


if __name__ == "__main__":
    preprocess_folder()