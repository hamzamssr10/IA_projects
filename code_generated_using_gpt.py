import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
img = cv2.imread("image.jpg")

# Create input blob
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), True, crop=False)

# Set input blob
net.setInput(blob)

# Run inference
outs = net.forward(net.getUnconnectedOutLayersNames())

# Process outputs
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            width = int(detection[2] * img.shape[1])
            height = int(detection[3] * img.shape[0])
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(img, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show image
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
