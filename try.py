import torch, onnxruntime as ort
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("Available ONNX providers:", ort.get_available_providers())
