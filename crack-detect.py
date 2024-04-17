from ultralytics import YOLO 
import cv2 
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available, using CPU instead.")
    device = torch.device("cpu")

model = YOLO("crack-detect-v3.pt").to(device)
model.predict(source="0", show=True, conf=0.5)