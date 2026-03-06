import os  
import cv2  
import numpy as np  
import uniface  
import argparse  
from tqdm import tqdm  
  
from models import ONNXFaceEngine  
from utils.face_utils import compute_similarity  
  
  
def load_pairs(annotation_file):  
    """Load image pairs from annotation file"""  
    pairs = []  
    with open(annotation_file) as f:  
        lines = f.readlines()[1:]  # Skip header  
        for line in lines:  
            parts = line.strip().split()  
            if len(parts) == 3:  
                is_same, path1, path2 = parts[0], parts[1], parts[2]  
                pairs.append((is_same, path1, path2))  
    return pairs  
  
  
def extract_onnx_features(model, detector, img_path, root_dir):  
    """Extract features using ONNX model with face detection"""  
    img_path = os.path.join(root_dir, img_path)  
    img = cv2.imread(img_path)  
      
    if img is None:  
        return None  
      
    # Detect face  
    detections = detector.detect(img)  
    if len(detections) == 0:  
        return None  
      
    # Get landmarks for first detected face  
    landmarks = np.array(detections[0]['landmarks'], dtype=np.float32)  
      
    # Extract embedding  
    try:  
        embedding = model.get_embedding(img, landmarks)  
        return embedding  
    except:  
        return None  
  
  
def eval_onnx(model_path, dataset='lfw', root='data/val'):  
    """Evaluate ONNX model on specified dataset"""  
    # Initialize model and detector  
    model = ONNXFaceEngine(model_path)  
    detector = uniface.RetinaFace(model="retinaface_mnet_v2", conf_thresh=0.45)  
      
    # Load pairs  
    ann_file = os.path.join(root, f'{dataset}_ann.txt')  
    pairs = load_pairs(ann_file)  
      
    print(f"Evaluating {len(pairs)} pairs on {dataset}...")  
      
    # Extract features and compute similarities  
    predicts = []  
    failed = 0  
      
    for is_same, path1, path2 in tqdm(pairs):  
        feat1 = extract_onnx_features(model, detector, path1, root)  
        feat2 = extract_onnx_features(model, detector, path2, root)  
          
        if feat1 is None or feat2 is None:  
            failed += 1  
            continue  
          
        similarity = compute_similarity(feat1, feat2)  
        predicts.append([path1, path2, similarity, is_same])  
      
    print(f"Failed to detect faces in {failed} pairs")  
      
    if len(predicts) == 0:  
        print("No valid pairs found!")  
        return 0.0, []  
      
    predicts = np.array(predicts)  
      
    # K-fold validation  
    thresholds = np.arange(-1.0, 1.0, 0.005)  
    accuracies = []  
      
    # Simple 10-fold split (you can reuse k_fold_split from evaluate.py)  
    fold_size = len(predicts) // 10  
    for i in range(10):  
        start = i * fold_size  
        end = (i + 1) * fold_size if i < 9 else len(predicts)  
          
        test_indices = list(range(start, end))  
        train_indices = [idx for idx in range(len(predicts)) if idx not in test_indices]  
          
        # Find best threshold on train set  
        train_preds = predicts[train_indices]  
        best_thresh = 0.0  
        best_acc = 0.0  
          
        for thresh in thresholds:  
            predictions = (train_preds[:, 2].astype(float) > thresh).astype(int)  
            labels = train_preds[:, 3].astype(int)  
            acc = np.mean(predictions == labels)  
            if acc > best_acc:  
                best_acc = acc  
                best_thresh = thresh  
          
        # Evaluate on test set  
        test_preds = predicts[test_indices]  
        predictions = (test_preds[:, 2].astype(float) > best_thresh).astype(int)  
        labels = test_preds[:, 3].astype(int)  
        accuracy = np.mean(predictions == labels)  
        accuracies.append(accuracy)  
      
    mean_accuracy = np.mean(accuracies)  
    std_accuracy = np.std(accuracies)  
      
    print(f'{dataset.upper()} ACC: {mean_accuracy:.4f} | STD: {std_accuracy:.4f}')  
    return mean_accuracy, predicts  
  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='ONNX Model Evaluation')  
    parser.add_argument('--model', type=str, required=True,   
                        help='Path to ONNX model file')  
    parser.add_argument('--dataset', type=str, default='lfw',  
                        choices=['lfw', 'calfw', 'cplfw', 'agedb_30'],  
                        help='Dataset to evaluate on')  
    parser.add_argument('--root', type=str, default='data/val',  
                        help='Root directory for validation data')  
      
    args = parser.parse_args()  
      
    # Run evaluation  
    accuracy, predictions = eval_onnx(args.model, args.dataset, args.root)