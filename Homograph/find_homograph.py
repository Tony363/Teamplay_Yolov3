import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_coefficients(src, dst):
    """ Save the camera matrix and thsave_txte distortion coefficients to given path/file. """
    path = 'homograph_pts/homographic.yml'
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("src_pts", src)
    cv_file.write("dst_pts", dst)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
    print("file saved")

def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("src_pts").mat()
    dist_matrix = cv_file.getNode("dst_pts").mat()
    print(camera_matrix,dist_matrix)
    cv_file.release()
    return camera_matrix, dist_matrix

#explicit is better than implicit cv2.IMREAD_GRAYSCALE is better than 0
img1 = cv2.imread("images/court.png", cv2.IMREAD_GRAYSCALE) # queryImage
img2 = cv2.imread("images/basketball.png", cv2.IMREAD_GRAYSCALE)  # trainImage
#CV doesn't hold hands, do the checks.
if (img1 is None) or (img2 is None):
    raise IOError("No files {0} and {1} found".format("img0.png", "img1.png"))

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create() 

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

tmp1 = cv2.drawKeypoints(img1, kp1,None)
tmp2 = cv2.drawKeypoints(img2, kp2,None)
plt.imshow(tmp1)
plt.show()
plt.imshow(tmp2)
plt.show()

index_params = dict(algorithm = 0, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)
matches = np.asarray(matches)

if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    # print(src)
    # print(dst)
    save_coefficients(src,dst)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
else:
    raise AssertionError("Can't find enough keypoints.")


