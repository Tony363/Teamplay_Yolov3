import cv2
import numpy as np
import os, sys,argparse
import glob
import imutils

def undistort(img_path,coeff):
    K,D,DIM = coeff
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite('{img_path}_undistorted.jpg'.format(img_path=img_path),undistorted_img)
    undistorted_img = imutils.resize(undistorted_img,width=1080)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def undistort_detailed(img_path,count,h,w,coeff, balance=0.0, dim2=None, dim3=None):
    K,D,DIM = coeff
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    if count == 0:
        h,w = dim1
    dim1 = (h,w)
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    undistorted_img = imutils.resize(undistorted_img,width=1080)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return h,w

def save_coefficients(DIM,mtx, dist, path):
    """ Save the camera matrix and save_text distortion coefficients to given path/file. """
    path = '{path}'.format(path=path)
    print(path)
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("DIM",DIM)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
    print("file saved")

def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    DIM = cv_file.getNode("DIM").mat()
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return camera_matrix, dist_matrix,(int(DIM[0]),int(DIM[1]))

def arguments():
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--image_format', type=str, required=True,  help='image format, png/jpg')
    parser.add_argument('--save_file', type=str, required=False, help='YML file to save calibration matrices')
    parser.add_argument('--read_yaml', type=str,required=False,help='chose yaml file to read')
    parser.add_argument('--detailed',nargs='+',type=float,required=False,
    help=
    """
    If you don\'t want to see these black hills, set balance to 0. 
    If you do want to see these black hills, set balance to 1.

    arg1 = 0 > balance < 1
    arg2 = dim2 dimension increase ratio
    arg3 = final dimension increase ratio
    """)
    args = parser.parse_args()
    return parser,args

def get_matrix(parser,args):
    CHECKERBOARD = (6,9)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('{folder}*.{format}'.format(folder=args.image_dir,format=args.image_format))
    print('{folder}*.{format}'.format(folder=args.image_dir,format=args.image_format))
    print(images)
    count = 0
    for fname in images:
        img = cv2.imread(fname)
        if count == 0 :
            h,w = img.shape[:2]
            count+=1
        img = cv2.resize(img,(w,h))
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM = " + str(_img_shape[::-1]))
    print("K = np.array(" + str(K.tolist()) + ")")
    print("D = np.array(" + str(D.tolist()) + ")")
    # You should replace these 3 lines with the output in calibration step
    DIM =_img_shape[::-1]
    K = np.array(K.tolist())
    D = np.array(D.tolist())
    save_coefficients(DIM,K,D,args.save_file)

def read_matrix(parser,args):
    coeff = load_coefficients('{file}'.format(file=args.read_yaml))
    images = glob.glob('{folder}*.{format}'.format(folder=args.image_dir,format=args.image_format))
    print('{folder}*.{format}'.format(folder=args.image_dir,format=args.image_format),'\n')
    print(images,'\n')
    count = 0
    h,w,dim2,dim3 = None,None,None,None
    if args.detailed:
        for p in images:
            print(p)
            h,w = undistort_detailed(p,count,h,w,coeff,args.detailed[0],dim2)
            count += 1
            dim2 = (int(h + h * args.detailed[1]), int(w + w * args.detailed[1]))
            dim3 = (int(h + h * args.detailed[2]), int(w + w * args.detailed[2]))
    else:
        for p in images:
            print(p)
            undistort(p,coeff)

if __name__ == "__main__":
    parser,args = arguments() 
    if not args.read_yaml:
        get_matrix(parser,args)
    if args.read_yaml:
        read_matrix(parser,args)