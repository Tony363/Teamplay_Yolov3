import numpy as np
import cv2
import glob
import argparse
import imutils
import yaml

# from calibration_store import save_coefficients

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dirpath, prefix, image_format, square_size, width=9, height=6):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]
    
    print(dirpath+'/' + prefix + '.' + image_format)
    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)
    count = 0
    total = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            print("picture {count} of {length}".format(count=count,length=len(range(total))))
            count += 1
        total += 1
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return [ret, mtx, dist, rvecs, tvecs]

def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    path = '/calib_matrix/{path}'.format(path=path)
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return [camera_matrix, dist_matrix]

def getCroppedImage(img,centroid):
    if centroid[0]-200 < 0:
        xmin = 0
        xmax = 200*2
    elif centroid[0] + 200 > img.shape[1]:
        xmin = img.shape[1] - 200* 2
        xmax = img.shape[1]
    else:
        # todo max/min unecessary
        xmin = max(0, centroid[0] - 200)
        xmax = min(img.shape[1], centroid[0] + 200)
    
    if centroid[1] - 200 < 0:
        ymin = 0
        ymax = 100 * 2
    elif centroid[1] + 200 > img.shape[0]:

        ymin = img.shape[0] - 200*2
        ymax = img.shape[0]
    else:
        ymin = max(0, centroid[1] - 200)
        ymax = min(img.shape[0],centroid[1] + 200)

    # Crop a fixed size img using centroid

    return xmin ,xmax, ymin, ymax

def undistortimg(mtx,dist,vid,view=False,write=False):
    cap = cv2.VideoCapture('input_vid/{vid}.mp4'.format(vid=vid))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    count = 0
    total = 0
    while True:
        ret,frame = cap.read()
        if ret:
            h,w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
            mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
            dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
            outh,  outw = dst.shape[:2] 
            if count == 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
                out = cv2.VideoWriter('output/output.mp4', fourcc, fps, (outw,outh)) 
            try:
                resized = imutils.resize(dst,width=1080)
            except Exception as e:
                print(e)
                pass
            if view:
                cv2.imshow('frame',resized)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    break
            if write:
               out.write(dst)
            print("frame {count} of {total}".format(count=count,total=total))
            count += 1
        else:
            print("end of video")
            break
        total += 1
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def zoom(mtx,dist,vid,write=False):
    cap = cv2.VideoCapture('input_vid/{vid}.mp4'.format(vid=vid))
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    count = 0
    while True:
        ret,frame = cap.read()
        if ret:
            frame = frame[100:600,500:1000]
            h, w = frame.shape[:2] 
            if count == 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
                out = cv2.VideoWriter('output/output.avi', fourcc, fps, (w,h)) 
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
            mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
            dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
            cv2.imshow('zoom',dst)
            cv2.waitKey()
            count += 1
        else:
            print('end of video')
            break
    cap.release()
    cv2.destroyAllWindows()

def opencv_matrix(loader,node):
    mapping = loader.construct_mapping(node)
    mat = np.array(mapping['data'])
    mat.resize(mapping['rows'],mapping['cols'])
    return mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--image_dir', type=str, required=False, help='image directory path')
    parser.add_argument('--image_format', type=str, required=False,  help='image format, png/jpg')
    parser.add_argument('--prefix', type=str, required=False, help='image prefix')
    parser.add_argument('--square_size', type=float, required=False, help='chessboard square size')
    parser.add_argument('--width', type=int, required=False, help='chessboard width size, default is 9')
    parser.add_argument('--height', type=int, required=False, help='chessboard height size, default is 6')
    parser.add_argument('--save_file', type=str, required=False, help='YML file to save calibration matrices')
    parser.add_argument('--read_image', type=str,required=True,help='chose path of image to undistort')
    parser.add_argument('--write_image', type=str,required=True, help='name undistortion image to undistort')
    parser.add_argument('--read_yaml', type=str,required=False,help='chose yaml file to read')
    parser.add_argument('--read_vid',type=str,required=False,help='enter video to read')
    parser.add_argument('--write_vid',action='store_true',help='write video to file')
    parser.add_argument('--zoom',action='store_true',required=False,help='zoom or not to zoom[True/False]')
    parser.add_argument('--view_vid',action='store_true',help='view video')
    args = parser.parse_args()

    if args.read_yaml:
        coeff = load_coefficients('calib_matrix/{file}'.format(file=args.read_yaml))
        mtx = coeff[0]
        dist = coeff[1]
        img = cv2.imread('calib_matrix/{undistort_img}.jpg'.format(undistort_img=args.read_image))
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        # undistort
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        # camera find straight line and work backwards from there
        cv2.imwrite('chessboardout/{img_name}.jpg'.format(img_name=args.write_image),dst)

        if args.view_vid:
            undistortimg(mtx,dist,args.read_vid,args.view_vid)
        elif args.write_vid:
            undistortimg(mtx,dist,args.read_vid,False,args.write_vid)
        elif args.zoom:
            zoom(mtx,dist,args.read_vid,args.write_vid,args.zoom)
    else:
        ret, mtx, dist, rvecs, tvecs = calibrate(args.image_dir, args.prefix, args.image_format, args.square_size, args.width, args.height)
        if args.save_file:
            save_coefficients(mtx, dist, args.save_file)
            print("Calibration is finished. RMS: ", ret)
        
        img = cv2.imread('calib_matrix/{undistort_img}.jpg'.format(undistort_img=args.read_image))
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
    
        cv2.imwrite('chessboardout/{img_name}.jpg'.format(img_name=args.write_image),dst)
        cv2.imwrite('chessboardout/zoom.jpg',dst[100:600,500:1000])

        if args.view_vid:
            undistortimg(mtx,dist,args.read_vid,args.view_vid)
        elif args.write_vid:
            undistortimg(mtx,dist,args.read_vid,args.write_vid)
        elif args.zoom:
            zoom(mtx,dist,args.read_vid,args.write_vid,args.zoom)
        