import cv2
import numpy as np
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")


cap = cv2.VideoCapture("http://192.168.10.145:8080/video")




classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Loop through each frame in the video
#from google.colab.patches import cv2_imshow
while(cap.isOpened()):
    # Read the next frame
    ret, frame = cap.read()

    # Resize the frame to the input size expected by the network
    #resized = cv2.resize(frame, (200, 200))
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        resized = cv2.resize(frame, (200, 200))
    else:
        continue

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(resized, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the neural network
    net.setInput(blob)

    # Forward the blob through the neural network
    outputs = net.forward(output_layers)

    # Extract the bounding boxes and confidence scores from the network outputs
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width/2)
                top = int(center_y - height/2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes on the
    # Draw the bounding boxes on the original frame
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            class_id = class_ids[i]
            label = classes[class_id]
            color = (0, 255, 0)
            cv2.rectangle(frame, (left, top), (left+width, top+height), color, 2)
            cv2.putText(frame, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow("detect",frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
