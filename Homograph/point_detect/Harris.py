# import the necessary packages
import numpy as np
import cv2
import imutils
import opencv_wrapper as cvw

# import  poly_point_isect as bot

# construct the argument parse and parse the arguments
# load the image
image = cv2.imread("/home/tony/Desktop/teamplay/yolov3/Homograph/images/cartoon_basket.jpg")
print(image.shape)
# define the list of boundaries
boundaries = [([180, 180, 100], [255, 255, 255])]

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
outimg = imutils.resize(output,width=1080)
cv2.imshow("Image", outimg)
cv2.waitKey(0)

# Start my code
gray = cvw.bgr2gray(output)

corners = cv2.cornerHarris(gray, 9, 3, 0.01)
corners = cvw.normalize(corners).astype(np.uint8)

thresh = cvw.threshold_otsu(corners)
dilated = cvw.dilate(thresh, 3)

contours = cvw.find_external_contours(dilated)

for contour in contours:
    cvw.circle(image, contour.center, 3, cvw.Color.RED, -1)
image = imutils.resize(image,width=1080)
cv2.imshow("Image", image)
cv2.waitKey(0)