# Import the required libraries
!pip install opencv-python torch torchvision
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5/
!pip install -r requirements.txt

import cv2
import torch
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression, scale_coords, plot_one_box

# Load the trained model
model = attempt_load('yolov5s.pt', map_location=torch.device('cpu'))

# Define the function for object detection
def detect(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)  # Convert the image to a tensor
    _, _, height, width = img.shape  # Get the height and width of the image
    device = select_device('')  # Select the device to use (GPU or CPU)
    img = img.to(device)  # Move the image to the device
    pred = model(img)[0]  # Get the predictions from the model
    pred = non_max_suppression(pred, 0.4, 0.5)  # Apply non-max suppression to the predictions
    for i, det in enumerate(pred):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (height, width)).round()  # Rescale the coordinates to the original image
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, image, label=label, color=(0, 255, 0), line_thickness=3)
    return image

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop through the frames
while True:
    ret, frame = cap.read()  # Capture the frame from the webcam
    if not ret:
        break
    frame = detect(frame)  # Detect the objects in the frame
    cv2.imshow('Object Detection', frame)  # Show the frame with detected objects
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
