import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images/basketball.png',0)
plt.imshow(img), plt.show()
print(img.shape)
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)

for marker in kp:
	mark1,mark2 = marker.pt
	img2 = cv2.drawMarker(img,(int(mark1),int(mark2)), color=(255,0,0))
plt.imshow(img2), plt.show()

# Print all default params
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))


# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print( "Total Keypoints without nonmaxSuppression: ", len(kp))

for marker in kp:
	mark1,mark2 = marker.pt
	img3 = cv2.drawMarker(img,(int(mark1),int(mark2)), color=(255,0,0))
plt.imshow(img3), plt.show()

