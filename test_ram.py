import psutil
import os
import onnxruntime


process = psutil.Process(os.getpid())

print("RAM trước load:", process.memory_info().rss / 1024 / 1024)

# load model
session = onnxruntime.InferenceSession("/home/dun/face-recognition/weights/mobilenetv1_0.25_mcp.onnx")

print("RAM sau load:", process.memory_info().rss / 1024 / 1024)

# tạo input
import numpy as np
input = np.random.randn(1,3,112,112).astype(np.float32)

print("RAM trước inference:", process.memory_info().rss / 1024 / 1024)

session.run(None, {"input": input})

print("RAM sau inference:", process.memory_info().rss / 1024 / 1024)