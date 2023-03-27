import cv2
import torch
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords, plot_one_box
from yolov5.utils.torch_utils import time_synchronized

# Load the YOLOv5 model
model = attempt_load("yolov5s.pt", map_location="cpu")

# Set the device to use for inference
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Set the confidence threshold for detection
conf_thres = 0.5

# Set the non-maximum suppression threshold for detection
nms_thres = 0.5

# Initialize the video stream
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to the input size of the YOLOv5 model
    img = cv2.resize(frame, (640, 640))

    # Convert the image to a PyTorch tensor
    img = torch.from_numpy(img).to(device)

    # Normalize the image
    img = img.float() / 255.0

    # Add a batch dimension to the image tensor
    img = img.unsqueeze(0)

    # Perform inference on the image using the YOLOv5 model
    with torch.no_grad():
        t1 = time_synchronized()
        outputs = model(img)
        t2 = time_synchronized()

    # Apply non-maximum suppression to the outputs
    outputs = non_max_suppression(outputs, conf_thres, nms_thres)[0]

    # Loop over the outputs and draw bounding boxes on the frame
    if outputs is not None:
        for output in outputs:
            x1, y1, x2, y2, conf, cls = output
            x1, y1, x2, y2 = scale_coords(img.shape[2:], (x1, y1, x2, y2), frame.shape).round()
            label = f"{model.names[int(cls)]} {conf:.2f}"
            plot_one_box((x1, y1, x2, y2), frame, label=label, color=(0, 255, 0))

    # Display the frame with the detected objects
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
