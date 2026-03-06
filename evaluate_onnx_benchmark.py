"""
Comprehensive ONNX Model Benchmark Evaluation
================================================
Đánh giá toàn diện model ONNX trên các benchmark face verification:
- Accuracy trên 4 datasets: LFW, CALFW, CPLFW, AgeDB_30
- RAM usage (trước/sau load, inference)
- Model file size
- Inference latency (trung bình)
- Số lượng parameters
- FLOPs estimation

Author: Auto-generated benchmark tool
Usage:
    python evaluate_onnx_benchmark.py
    python evaluate_onnx_benchmark.py --model weights/mobilenetv1_0.25_mcp.onnx
    python evaluate_onnx_benchmark.py --metrics-only
    python evaluate_onnx_benchmark.py --output results.csv
"""

import os
import sys
import glob
import time
import argparse

import cv2
import numpy as np
import psutil
import onnx
import onnxruntime as ort

from tqdm import tqdm
from utils.face_utils import compute_similarity

# ============================================================================
# CONSTANTS
# ============================================================================
DATASETS = ['lfw', 'calfw', 'cplfw', 'agedb_30']
VAL_ROOT = 'data/val'
INPUT_SHAPE = (1, 3, 112, 112)
WARMUP_RUNS = 10
LATENCY_RUNS = 100


# ============================================================================
# MODEL PROFILER — Resource Metrics
# ============================================================================
class ModelProfiler:
    """Đo lường tài nguyên của model ONNX: size, RAM, latency, params, FLOPs."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]

    def get_file_size_mb(self) -> float:
        """Dung lượng file model trên disk (MB)."""
        size_bytes = os.path.getsize(self.model_path)
        return size_bytes / (1024 * 1024)

    def get_num_params(self) -> int:
        """Tổng số parameters từ ONNX graph initializers."""
        model = onnx.load(self.model_path)
        total_params = 0
        for initializer in model.graph.initializer:
            shape = initializer.dims
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            total_params += num_elements
        return total_params

    def get_flops_estimation(self) -> int:
        """
        Ước tính FLOPs từ ONNX graph.
        Tính dựa trên các Conv, Gemm, MatMul nodes.
        """
        model = onnx.load(self.model_path)
        graph = model.graph

        # Build shape map from initializers and value_info
        shape_map = {}
        for init in graph.initializer:
            shape_map[init.name] = list(init.dims)
        for vi in graph.value_info:
            if vi.type.tensor_type.shape.dim:
                dims = []
                for d in vi.type.tensor_type.shape.dim:
                    dims.append(d.dim_value if d.dim_value > 0 else 1)
                shape_map[vi.name] = dims
        for inp in graph.input:
            if inp.type.tensor_type.shape.dim:
                dims = []
                for d in inp.type.tensor_type.shape.dim:
                    dims.append(d.dim_value if d.dim_value > 0 else 1)
                shape_map[inp.name] = dims

        total_flops = 0

        for node in graph.node:
            if node.op_type == 'Conv':
                # Get weight shape: [out_channels, in_channels/groups, kH, kW]
                weight_name = node.input[1]
                if weight_name in shape_map:
                    w_shape = shape_map[weight_name]
                    # Get output shape if available
                    out_name = node.output[0]
                    if out_name in shape_map:
                        out_shape = shape_map[out_name]
                        # FLOPs = 2 * out_H * out_W * out_C * in_C/groups * kH * kW
                        if len(out_shape) == 4 and len(w_shape) == 4:
                            flops = 2 * out_shape[2] * out_shape[3] * w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
                            total_flops += flops
                    else:
                        # Estimate: use weight params * 2 * spatial (assume 112->56->28...)
                        kernel_ops = w_shape[1] * w_shape[2] * w_shape[3] * 2
                        total_flops += kernel_ops * w_shape[0]

            elif node.op_type in ('Gemm', 'MatMul'):
                # Get input shapes
                if len(node.input) >= 2:
                    a_name = node.input[0]
                    b_name = node.input[1]
                    if b_name in shape_map:
                        b_shape = shape_map[b_name]
                        if len(b_shape) == 2:
                            # FLOPs = 2 * M * N * K
                            total_flops += 2 * b_shape[0] * b_shape[1]
                    elif a_name in shape_map:
                        a_shape = shape_map[a_name]
                        if len(a_shape) == 2:
                            total_flops += 2 * a_shape[0] * a_shape[1]

            elif node.op_type in ('BatchNormalization',):
                # ~4 ops per element
                inp_name = node.input[0]
                if inp_name in shape_map:
                    s = shape_map[inp_name]
                    n_elements = 1
                    for d in s:
                        n_elements *= d
                    total_flops += 4 * n_elements

            elif node.op_type in ('Relu', 'PRelu', 'Sigmoid', 'HardSwish', 'HardSigmoid'):
                inp_name = node.input[0]
                if inp_name in shape_map:
                    s = shape_map[inp_name]
                    n_elements = 1
                    for d in s:
                        n_elements *= d
                    total_flops += n_elements

            elif node.op_type in ('Add', 'Mul'):
                inp_name = node.input[0]
                if inp_name in shape_map:
                    s = shape_map[inp_name]
                    n_elements = 1
                    for d in s:
                        n_elements *= d
                    total_flops += n_elements

        return total_flops

    def measure_ram_usage(self) -> dict:
        """
        Đo RAM usage tại các thời điểm:
        - Trước khi load model
        - Sau khi load model
        - Sau khi inference
        Returns dict với ram_before, ram_after_load, ram_after_inference, ram_model_only (MB)
        """
        process = psutil.Process(os.getpid())

        ram_before = process.memory_info().rss / (1024 * 1024)

        session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )
        ram_after_load = process.memory_info().rss / (1024 * 1024)

        # Run inference
        dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        input_name = session.get_inputs()[0].name
        session.run(None, {input_name: dummy_input})

        ram_after_inference = process.memory_info().rss / (1024 * 1024)

        del session

        return {
            'ram_before': ram_before,
            'ram_after_load': ram_after_load,
            'ram_after_inference': ram_after_inference,
            'ram_model_only': ram_after_load - ram_before,
            'ram_peak_inference': ram_after_inference - ram_before,
        }

    def measure_latency(self, num_warmup: int = WARMUP_RUNS, num_runs: int = LATENCY_RUNS) -> dict:
        """
        Đo inference latency trung bình.
        Returns dict với avg_ms, min_ms, max_ms, std_ms
        """
        session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )
        dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(num_warmup):
            session.run(None, {input_name: dummy_input})

        # Measure
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        del session

        latencies = np.array(latencies)
        return {
            'avg_ms': float(np.mean(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'std_ms': float(np.std(latencies)),
        }

    def profile_all(self) -> dict:
        """Chạy toàn bộ profiling metrics."""
        print(f"\n{'='*60}")
        print(f"  Profiling: {self.model_name}")
        print(f"{'='*60}")

        print("  → Measuring file size...")
        file_size = self.get_file_size_mb()

        print("  → Counting parameters...")
        num_params = self.get_num_params()

        print("  → Estimating FLOPs...")
        flops = self.get_flops_estimation()

        print("  → Measuring RAM usage...")
        ram = self.measure_ram_usage()

        print(f"  → Measuring latency ({WARMUP_RUNS} warmup + {LATENCY_RUNS} runs)...")
        latency = self.measure_latency()

        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'file_size_mb': file_size,
            'num_params': num_params,
            'flops': flops,
            'ram': ram,
            'latency': latency,
        }


# ============================================================================
# ACCURACY EVALUATION — Reusing logic from evaluate_onnx.py
# ============================================================================
def load_pairs(annotation_file: str) -> list:
    """Load image pairs from annotation file."""
    pairs = []
    with open(annotation_file) as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                is_same, path1, path2 = parts[0], parts[1], parts[2]
                pairs.append((is_same, path1, path2))
    return pairs


def extract_onnx_features(session, input_name, output_names, img_path: str, root_dir: str):
    """
    Trích xuất features từ ảnh khuôn mặt đã crop sẵn (112x112).
    Không cần face detection vì ảnh trong LFW/CALFW/CPLFW/AgeDB_30 đã được crop.
    """
    full_path = os.path.join(root_dir, img_path)
    img = cv2.imread(full_path)

    if img is None:
        return None

    try:
        # Preprocess: resize to 112x112, normalize, convert BGR->RGB
        img_resized = cv2.resize(img, (112, 112))
        blob = cv2.dnn.blobFromImage(
            img_resized,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True
        )

        # Run inference
        embedding = session.run(output_names, {input_name: blob})[0]

        # Cũng trích xuất features từ ảnh lật (flipped) để kết hợp
        img_flipped = cv2.flip(img_resized, 1)  # Horizontal flip
        blob_flipped = cv2.dnn.blobFromImage(
            img_flipped,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True
        )
        embedding_flipped = session.run(output_names, {input_name: blob_flipped})[0]

        # Kết hợp features: concat original + flipped
        combined = np.concatenate([embedding, embedding_flipped], axis=1).flatten()
        return combined
    except Exception as e:
        print(f"    ⚠ Error processing {img_path}: {e}")
        return None


def k_fold_accuracy(predicts: np.ndarray, n_folds: int = 10) -> tuple:
    """K-Fold cross-validation for finding best threshold and accuracy."""
    thresholds = np.arange(-1.0, 1.0, 0.005)
    accuracies = []
    best_thresholds = []

    fold_size = len(predicts) // n_folds
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(predicts)

        test_indices = list(range(start, end))
        train_indices = [idx for idx in range(len(predicts)) if idx not in test_indices]

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

        test_preds = predicts[test_indices]
        predictions = (test_preds[:, 2].astype(float) > best_thresh).astype(int)
        labels = test_preds[:, 3].astype(int)
        accuracy = np.mean(predictions == labels)

        accuracies.append(accuracy)
        best_thresholds.append(best_thresh)

    return float(np.mean(accuracies)), float(np.std(accuracies)), float(np.mean(best_thresholds))


def evaluate_on_dataset(model_path: str, dataset: str, root: str = VAL_ROOT) -> dict:
    """Đánh giá model ONNX trên 1 dataset. Ảnh đã crop sẵn, không cần face detection."""
    ann_file = os.path.join(root, f'{dataset}_ann.txt')
    if not os.path.exists(ann_file):
        print(f"  ⚠ Không tìm thấy annotation: {ann_file}")
        return {'accuracy': 0.0, 'std': 0.0, 'threshold': 0.0, 'failed': 0, 'total': 0}

    # Load ONNX session trực tiếp (không cần ONNXFaceEngine)
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_cfg = session.get_inputs()[0]
    input_name = input_cfg.name
    output_names = [o.name for o in session.get_outputs()]

    pairs = load_pairs(ann_file)
    total = len(pairs)

    predicts = []
    failed = 0

    for is_same, path1, path2 in tqdm(pairs, desc=f"    {dataset.upper()}", leave=False):
        feat1 = extract_onnx_features(session, input_name, output_names, path1, root)
        feat2 = extract_onnx_features(session, input_name, output_names, path2, root)

        if feat1 is None or feat2 is None:
            failed += 1
            continue

        similarity = compute_similarity(feat1, feat2)
        predicts.append([path1, path2, similarity, is_same])

    del session

    if len(predicts) == 0:
        return {'accuracy': 0.0, 'std': 0.0, 'threshold': 0.0, 'failed': failed, 'total': total}

    predicts = np.array(predicts)
    mean_acc, std_acc, mean_thresh = k_fold_accuracy(predicts)

    return {
        'accuracy': mean_acc,
        'std': std_acc,
        'threshold': mean_thresh,
        'failed': failed,
        'total': total,
    }


def evaluate_all_datasets(model_path: str, root: str = VAL_ROOT) -> dict:
    """Evaluate a model on all 4 datasets."""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n  Evaluating accuracy: {model_name}")

    results = {}
    for dataset in DATASETS:
        result = evaluate_on_dataset(model_path, dataset, root)
        results[dataset] = result
        if result['accuracy'] > 0:
            print(f"    {dataset.upper()} ACC: {result['accuracy']*100:.2f}% ± {result['std']*100:.2f}% "
                  f"(failed: {result['failed']}/{result['total']})")
        else:
            print(f"    {dataset.upper()}: SKIPPED or FAILED")

    return results


# ============================================================================
# RESULTS TABLE — Formatted Output
# ============================================================================
def format_params(n: int) -> str:
    """Format parameter count to human readable: 1.25M, 350K, etc."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_flops(n: int) -> str:
    """Format FLOPs count."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}G"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def compute_efficiency_score(result: dict) -> float:
    """
    Tính điểm hiệu suất tổng hợp (Efficiency Score).
    Cân bằng giữa accuracy, model size, và latency.

    Score = (avg_accuracy * 100) / (size_mb * latency_ms)^0.5
    Score cao hơn = model tối ưu hơn.
    """
    accuracies = []
    for ds in DATASETS:
        if ds in result.get('accuracy_results', {}) and result['accuracy_results'][ds]['accuracy'] > 0:
            accuracies.append(result['accuracy_results'][ds]['accuracy'])

    if not accuracies:
        return 0.0

    avg_acc = np.mean(accuracies) * 100
    size_mb = result['profile']['file_size_mb']
    latency_ms = result['profile']['latency']['avg_ms']

    # Tránh chia cho 0
    if size_mb <= 0 or latency_ms <= 0:
        return 0.0

    score = avg_acc / (size_mb * latency_ms) ** 0.3
    return round(score, 2)


def print_results_table(all_results: list, metrics_only: bool = False):
    """In bảng kết quả formatted."""

    print("\n")
    print("=" * 140)
    print("  📊  ONNX MODEL BENCHMARK RESULTS")
    print("=" * 140)

    if not metrics_only:
        # ── Accuracy Table ──
        print("\n┌─── ACCURACY COMPARISON ───────────────────────────────────────────────────────────────────────┐")
        header = f"│ {'Model':<30} │ {'LFW (%)':<12} │ {'CALFW (%)':<12} │ {'CPLFW (%)':<12} │ {'AgeDB_30 (%)':<14} │"
        sep =    f"├{'─'*31}─┼{'─'*13}─┼{'─'*13}─┼{'─'*13}─┼{'─'*15}─┤"
        print(f"├{'─'*31}─┬{'─'*13}─┬{'─'*13}─┬{'─'*13}─┬{'─'*15}─┤")
        print(header)
        print(sep)

        for res in all_results:
            name = res['profile']['model_name'][:30]
            accs = {}
            for ds in DATASETS:
                if ds in res.get('accuracy_results', {}):
                    a = res['accuracy_results'][ds]['accuracy']
                    accs[ds] = f"{a*100:.2f}" if a > 0 else "N/A"
                else:
                    accs[ds] = "N/A"

            row = (f"│ {name:<30} │ {accs.get('lfw', 'N/A'):<12} │ {accs.get('calfw', 'N/A'):<12} │ "
                   f"{accs.get('cplfw', 'N/A'):<12} │ {accs.get('agedb_30', 'N/A'):<14} │")
            print(row)

        print(f"└{'─'*31}─┴{'─'*13}─┴{'─'*13}─┴{'─'*13}─┴{'─'*15}─┘")

    # ── Resource Metrics Table ──
    print("\n┌─── RESOURCE METRICS ──────────────────────────────────────────────────────────────────────────────────────────────┐")
    header2 = (f"│ {'Model':<30} │ {'Params':<12} │ {'Size (MB)':<12} │ {'RAM (MB)':<12} │ "
               f"{'Latency (ms)':<14} │ {'FLOPs':<12} │")
    sep2 = f"├{'─'*31}─┼{'─'*13}─┼{'─'*13}─┼{'─'*13}─┼{'─'*15}─┼{'─'*13}─┤"
    print(f"├{'─'*31}─┬{'─'*13}─┬{'─'*13}─┬{'─'*13}─┬{'─'*15}─┬{'─'*13}─┤")
    print(header2)
    print(sep2)

    for res in all_results:
        p = res['profile']
        name = p['model_name'][:30]
        params = format_params(p['num_params'])
        size = f"{p['file_size_mb']:.2f}"
        ram = f"{p['ram']['ram_model_only']:.1f}"
        latency = f"{p['latency']['avg_ms']:.2f}"
        flops = format_flops(p['flops'])

        row = (f"│ {name:<30} │ {params:<12} │ {size:<12} │ {ram:<12} │ "
               f"{latency:<14} │ {flops:<12} │")
        print(row)

    print(f"└{'─'*31}─┴{'─'*13}─┴{'─'*13}─┴{'─'*13}─┴{'─'*15}─┴{'─'*13}─┘")

    # ── Detailed Latency ──
    print("\n┌─── LATENCY DETAILS ──────────────────────────────────────────────────────────────────────────┐")
    header3 = f"│ {'Model':<30} │ {'Avg (ms)':<12} │ {'Min (ms)':<12} │ {'Max (ms)':<12} │ {'Std (ms)':<12} │"
    print(f"├{'─'*31}─┬{'─'*13}─┬{'─'*13}─┬{'─'*13}─┬{'─'*13}─┤")
    print(header3)
    print(f"├{'─'*31}─┼{'─'*13}─┼{'─'*13}─┼{'─'*13}─┼{'─'*13}─┤")

    for res in all_results:
        p = res['profile']
        name = p['model_name'][:30]
        lat = p['latency']
        row = (f"│ {name:<30} │ {lat['avg_ms']:<12.2f} │ {lat['min_ms']:<12.2f} │ "
               f"{lat['max_ms']:<12.2f} │ {lat['std_ms']:<12.2f} │")
        print(row)

    print(f"└{'─'*31}─┴{'─'*13}─┴{'─'*13}─┴{'─'*13}─┴{'─'*13}─┘")

    # ── RAM Details ──
    print("\n┌─── RAM USAGE DETAILS ────────────────────────────────────────────────────────────────────────────────┐")
    header4 = (f"│ {'Model':<30} │ {'Before (MB)':<14} │ {'After Load (MB)':<17} │ "
               f"{'After Infer (MB)':<17} │ {'Model Only (MB)':<16} │")
    print(f"├{'─'*31}─┬{'─'*15}─┬{'─'*18}─┬{'─'*18}─┬{'─'*17}─┤")
    print(header4)
    print(f"├{'─'*31}─┼{'─'*15}─┼{'─'*18}─┼{'─'*18}─┼{'─'*17}─┤")

    for res in all_results:
        p = res['profile']
        name = p['model_name'][:30]
        ram = p['ram']
        row = (f"│ {name:<30} │ {ram['ram_before']:<14.1f} │ {ram['ram_after_load']:<17.1f} │ "
               f"{ram['ram_after_inference']:<17.1f} │ {ram['ram_model_only']:<16.1f} │")
        print(row)

    print(f"└{'─'*31}─┴{'─'*15}─┴{'─'*18}─┴{'─'*18}─┴{'─'*17}─┘")

    # ── Efficiency Score ──
    if not metrics_only:
        print("\n┌─── EFFICIENCY RANKING (Accuracy / (Size × Latency)^0.3) ─────────────────────────────────────┐")
        header5 = f"│ {'Rank':<6} │ {'Model':<30} │ {'Avg Acc (%)':<14} │ {'Size (MB)':<12} │ {'Latency (ms)':<14} │ {'Score':<10} │"
        print(f"├{'─'*7}─┬{'─'*31}─┬{'─'*15}─┬{'─'*13}─┬{'─'*15}─┬{'─'*11}─┤")
        print(header5)
        print(f"├{'─'*7}─┼{'─'*31}─┼{'─'*15}─┼{'─'*13}─┼{'─'*15}─┼{'─'*11}─┤")

        # Compute scores and rank
        scored = []
        for res in all_results:
            score = compute_efficiency_score(res)
            scored.append((res, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        for rank, (res, score) in enumerate(scored, 1):
            p = res['profile']
            name = p['model_name'][:30]
            accs = []
            for ds in DATASETS:
                if ds in res.get('accuracy_results', {}) and res['accuracy_results'][ds]['accuracy'] > 0:
                    accs.append(res['accuracy_results'][ds]['accuracy'] * 100)
            avg_acc = f"{np.mean(accs):.2f}" if accs else "N/A"
            size = f"{p['file_size_mb']:.2f}"
            latency = f"{p['latency']['avg_ms']:.2f}"
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "

            row = (f"│ {medal} {rank:<3} │ {name:<30} │ {avg_acc:<14} │ {size:<12} │ "
                   f"{latency:<14} │ {score:<10} │")
            print(row)

        print(f"└{'─'*7}─┴{'─'*31}─┴{'─'*15}─┴{'─'*13}─┴{'─'*15}─┴{'─'*11}─┘")

    print("\n" + "=" * 140)
    print("  ✅  Benchmark completed!")
    print("=" * 140 + "\n")


def save_to_csv(all_results: list, output_path: str, metrics_only: bool = False):
    """Xuất kết quả ra CSV file."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Model', 'Num_Params', 'Size_MB', 'RAM_Model_MB', 'RAM_Peak_MB',
                  'Latency_Avg_ms', 'Latency_Min_ms', 'Latency_Max_ms', 'Latency_Std_ms',
                  'FLOPs']
        if not metrics_only:
            header.extend(['LFW_%', 'CALFW_%', 'CPLFW_%', 'AgeDB_30_%', 'Avg_Acc_%', 'Efficiency_Score'])
        writer.writerow(header)

        for res in all_results:
            p = res['profile']
            row = [
                p['model_name'],
                p['num_params'],
                f"{p['file_size_mb']:.2f}",
                f"{p['ram']['ram_model_only']:.1f}",
                f"{p['ram']['ram_peak_inference']:.1f}",
                f"{p['latency']['avg_ms']:.2f}",
                f"{p['latency']['min_ms']:.2f}",
                f"{p['latency']['max_ms']:.2f}",
                f"{p['latency']['std_ms']:.2f}",
                p['flops'],
            ]
            if not metrics_only:
                accs = []
                for ds in DATASETS:
                    if ds in res.get('accuracy_results', {}):
                        a = res['accuracy_results'][ds]['accuracy']
                        row.append(f"{a*100:.2f}" if a > 0 else "N/A")
                        if a > 0:
                            accs.append(a * 100)
                    else:
                        row.append("N/A")
                row.append(f"{np.mean(accs):.2f}" if accs else "N/A")
                row.append(compute_efficiency_score(res))
            writer.writerow(row)

    print(f"\n  📁 Results saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Comprehensive ONNX Model Benchmark Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_onnx_benchmark.py                           # Evaluate all ONNX in weights/
  python evaluate_onnx_benchmark.py --model weights/m1.onnx   # Evaluate specific model
  python evaluate_onnx_benchmark.py --metrics-only             # Only resource metrics (no accuracy)
  python evaluate_onnx_benchmark.py --output results.csv       # Save to CSV
        """
    )
    parser.add_argument('--model', type=str, nargs='+', default=None,
                        help='Path(s) to ONNX model file(s). If not specified, all *.onnx in weights/ are used.')
    parser.add_argument('--root', type=str, default=VAL_ROOT,
                        help='Root directory for validation data (default: data/val)')
    parser.add_argument('--metrics-only', action='store_true',
                        help='Only measure resource metrics (skip accuracy evaluation)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to CSV file')
    parser.add_argument('--warmup', type=int, default=WARMUP_RUNS,
                        help=f'Number of warmup runs for latency (default: {WARMUP_RUNS})')
    parser.add_argument('--runs', type=int, default=LATENCY_RUNS,
                        help=f'Number of runs for latency measurement (default: {LATENCY_RUNS})')

    return parser.parse_args()


def main():
    args = parse_args()

    global WARMUP_RUNS, LATENCY_RUNS
    WARMUP_RUNS = args.warmup
    LATENCY_RUNS = args.runs

    # Find ONNX models
    if args.model:
        model_paths = args.model
    else:
        model_paths = sorted(glob.glob('weights/*.onnx'))

    if not model_paths:
        print("❌ No ONNX models found! Use --model to specify path(s).")
        sys.exit(1)

    # Validate paths
    for mp in model_paths:
        if not os.path.exists(mp):
            print(f"❌ Model not found: {mp}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  🔬  ONNX Model Comprehensive Benchmark")
    print("=" * 60)
    print(f"  Models: {len(model_paths)}")
    for mp in model_paths:
        print(f"    • {mp}")
    print(f"  Metrics only: {args.metrics_only}")
    print(f"  Latency: {WARMUP_RUNS} warmup + {LATENCY_RUNS} runs")
    if not args.metrics_only:
        print(f"  Datasets: {', '.join([d.upper() for d in DATASETS])}")
    print("=" * 60)

    all_results = []

    for model_path in model_paths:
        # Profile resources
        profiler = ModelProfiler(model_path)
        profile = profiler.profile_all()

        result = {'profile': profile}

        # Evaluate accuracy if not metrics-only
        if not args.metrics_only:
            acc_results = evaluate_all_datasets(model_path, args.root)
            result['accuracy_results'] = acc_results

        all_results.append(result)

    # Print results table
    print_results_table(all_results, metrics_only=args.metrics_only)

    # Save to CSV if requested
    if args.output:
        save_to_csv(all_results, args.output, metrics_only=args.metrics_only)


if __name__ == '__main__':
    main()
