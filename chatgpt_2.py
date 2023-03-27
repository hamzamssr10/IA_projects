import cv2
import numpy as np

# Load the YOLOv5 model
net = cv2.dnn.readNet("yolov5s.onnx")

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

    # Normalize the image
    img = img / 255.0

    # Convert the image to a 4D blob with the shape (1, 3, 640, 640)
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True)

    # Set the input for the network
    net.setInput(blob)

    # Run inference on the input
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Loop over the outputs and draw bounding boxes on the frame
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_thres:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                x1 = int(center_x - width / 2)
                y1 = int(center_y - height / 2)
                x2 = x1 + width
                y2 = y1 + height
                label = f"{class_id} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the detected objects
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
