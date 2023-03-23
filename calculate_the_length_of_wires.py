from scipy.interpolate import splprep, splev

# Load the image and run it through the YOLO model to get the segmentation points
image = cv2.imread('wire_image.jpg')
points = yolo_model.segment(image)

# Fit a B-spline curve to the segmentation points
tck, u = splprep(points.T, u=None, s=0.0)

# Calculate the arc length of the curve
arc_length = 0
for i in range(len(u) - 1):
    x1, y1 = splev(u[i], tck)
    x2, y2 = splev(u[i+1], tck)
    arc_length += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

print(f'The length of the wire is {arc_length} pixels')
#############################################################################################################################

from scipy.interpolate import splprep, splev
from scipy.integrate import quad

# Load the image and run it through your fine-tuned OpenPose model to get the skeleton points
image = cv2.imread('wire_image.jpg')
points = openpose_model.detect_wire_skeleton(image)

# Fit a B-spline curve to the skeleton points
tck, u = splprep(points.T, u=None, s=0.0)

# Define a function to calculate the integrand at a given value of t
def integrand(t):
    dx_dt = splev(t, tck, der=1)[0]
    dy_dt = splev(t, tck, der=1)[1]
    return (dx_dt ** 2 + dy_dt ** 2) ** 0.5

# Calculate the arc length of the curve using numerical integration
arc_length = quad(integrand, u[0], u[-1])[0]

print(f'The length of the wire is {arc_length} pixels')
