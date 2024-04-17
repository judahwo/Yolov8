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

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Predict the objects in the frame
    results = model.predict(source="0", show=True, conf=0.5)

    # If results is a list, no objects were found in the frame
    if isinstance(results, list):
        img = frame
    else:
        img = results.imgs[0]

    # Resize the window
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam', 1000, 1000)

    # Show the frame with the predictions
    cv2.imshow('Webcam', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break