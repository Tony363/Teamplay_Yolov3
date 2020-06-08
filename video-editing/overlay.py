import cv2
import numpy as np
import argparse
import imutils



def readCommand():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagePath", help="path to the background image")
    parser.add_argument("overlayPath", help="path to the image overlay")
    parser.add_argument("-o", "--opacity", type=float, default=1.0, help="overlay opacity")
    parser.add_argument("-c", "--correct", action='store_true', help="correct overlay transparency")
    parser.add_argument("-l", "--lines", action='store_true', help="show selected corners and lines")
    args = vars(parser.parse_args())
    return args

def select_corners(event, x, y, flags, param):
    # grab reference to the global variables
    global ptsDst

    # Read callback parameters
    imageSrc, overlaySrc, opacity, lines = param
    overlayWidth = overlaySrc.shape[0]
    overlayHeight = overlaySrc.shape[1]

    # Four corners of overlay image
    pts_src = np.array([[0, 0],[0,overlayWidth],[overlayHeight, overlayWidth],[overlayHeight,0]])

    # Points selection order to get the right overlay position. Top left - Bottom left - Bottom right - Top right
    if len(ptsDst) < 4:
        if event == cv2.EVENT_LBUTTONDOWN:
            print("x : {}, y : {}".format(x,y))
            ptsDst.append((x,y))
            if lines :
                if len(ptsDst) > 1:
                    imageSrc = cv2.line(imageSrc,ptsDst[len(ptsDst)-1],ptsDst[len(ptsDst)-2],(255,0,0),2)

                if len(ptsDst) == 4:
                    imageSrc = cv2.line(imageSrc,ptsDst[len(ptsDst)-1],ptsDst[0],(255,0,0),2)

            cv2.imshow("image", imageSrc)

    else:
        # Calculate Homography
        h, status = cv2.findHomography(pts_src, np.array(ptsDst))
        
        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(overlaySrc, h, (imageSrc.shape[1],imageSrc.shape[0]))
        
        # Blend the two images
        # Opacity is the transparency of the overlay. It is also possible to adjust the beta and gamma parameters from the OpenCV function (See cv2.addWeighted)
        cv2.addWeighted(im_out, opacity, imageSrc, 1, 0, imageSrc)

        # Display images
        cv2.imshow("Source Image", overlaySrc)
        cv2.imshow("Destination Image", imageSrc)
        cv2.imshow("Warped Source Image", im_out)





if __name__ == '__main__' : 

    # initialize the list of destination reference points
    ptsDst = []

    args = readCommand()
    # Read source image.    
    imageSrc = cv2.imread(args["imagePath"],cv2.IMREAD_UNCHANGED)
    # Add the extra dimension related to transparency
    imageSrc = np.dstack([imageSrc, np.ones((imageSrc.shape[0], imageSrc.shape[1]), dtype="uint8") * 255])
    imageSrc = imutils.resize(imageSrc, width=1000)
    # Clone image for future usage
    clone = imageSrc.copy()

    # Read overlay image
    overlaySrc = cv2.imread(args["overlayPath"],cv2.IMREAD_UNCHANGED)
    overlaySrc = imutils.resize(overlaySrc, width = 400)
    
    if args["correct"]:
        (B, G, R, A) = cv2.split(overlaySrc)
        B = cv2.bitwise_and(B, B, mask=A)
        G = cv2.bitwise_and(G, G, mask=A)
        R = cv2.bitwise_and(R, R, mask=A)
        overlaySrc = cv2.merge([B, G, R, A])


    # Set image window for callback
    cv2.namedWindow("image")
    # Set mouse callback
    cv2.setMouseCallback("image", select_corners, [imageSrc,overlaySrc, args['opacity'], args['lines']])

   

    while True:

        # display the image and wait for a keypress
        cv2.imshow("image", imageSrc)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the selected corners
        if key == ord("r"):
            imageSrc = clone.copy()
            ptsDst = []

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
        # Select 4 points

    cv2.destroyAllWindows()