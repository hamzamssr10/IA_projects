import cv2

# Load the image and convert it to grayscale
image = cv2.imread("example.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
_, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

# Define the pixels per metric ratio
pixelsPerMetric = None

# Loop over the contours
for c in contours:
    # If the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue

    # Compute the rotated bounding box of the contour
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # Order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left order
    box = perspective.order_points(box)

    # Compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between bottom-left and bottom-right points
    (tltrX, tltrY) = midpoint(box[0], box[1])
    (blbrX, blbrY) = midpoint(box[2], box[3])

    # Compute the midpoint between the top-right and bottom-right points,
    # followed by the midpoint between top-left and bottom-left points
    (tlblX, tlblY) = midpoint(box[1], box[3])
    (trbrX, trbrY) = midpoint(box[0], box[2])

    # Compute the Euclidean distance between midpoints,
    # then construct reference object dimensions
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetric is None:
        pixelsPerMetric = dB / width_of_reference_object

# Compute size of object in some unit of measurement
dimA = dA / pixelsPerMetric
dimB = dB / pixelsPerMetric

print(f"Object has dimensions {dimA:.1f} x {dimB:.1f}")
