import cv2
import numpy as np  
import imutils
import argparse
class CoordinateStore:
    def __init__(self,args,ratio):
        self.points = []
        self.ratio = ratio
        self.args = args

    def select_point(self,event,x,y,flags,param):
        if args.resized:
            if event == cv2.EVENT_LBUTTONDBLCLK:
                x = int(x * self.ratio)
                y = int(y * self.ratio)
                cv2.circle(img,(x,y),30,(255,0,0),-1)
                self.points.append([x,y])
        else:
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(img,(x,y),30,(255,0,0),-1)
                self.points.append([x,y])

    def save_pts(self,pts,args):
        """ Save the camera matrix and thsave_txte distortion coefficients to given path/file. """
        if args.src:
            path = 'homograph_pts/src_pts.yml'
            cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
            cv_file.write("src_pts", pts)
        elif args.dst:
            path = 'homograph_pts/dst_pts.yml'
            cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
            cv_file.write("dst_pts",pts)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()
        print("file saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="src_pts or dst_pts")
    parser.add_argument("--image",type=str,required=True,help="chose image from images/")
    parser.add_argument("--resized",action="store_true",help="take coordinates from resized window")
    parser.add_argument("--src",action="store_true",required=False,help="src_pts true")
    parser.add_argument("--dst",action="store_true",required=False,help="dst_pts true")
    args = parser.parse_args()

    img = cv2.imread('{image}'.format(image=args.image))
    resized = imutils.resize(img,width=1080)
    resized_ratio = img.shape[1]/resized.shape[1]
    print("resized ratio: ",img.shape[1]/resized.shape[1])

    #instantiate class
    coordinateStore1 = CoordinateStore(args,resized_ratio)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',coordinateStore1.select_point)

    while(1):
        if args.resized: 
            cv2.imshow('image',resized)
        else:
            cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()

    print("Selected Coordinates: ",np.array(coordinateStore1.points))
    coordinateStore1.save_pts(np.array(coordinateStore1.points),args) 
