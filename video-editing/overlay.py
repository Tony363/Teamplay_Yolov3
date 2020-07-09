import cv2
import numpy as np
import argparse
import imutils
import os
import time

COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)
FILL = -1
LINE_THICKNESS = 2
POINT_RADIUS = 5
RESULT_IMG = "result.png"
RESULT_VID = "result_video.mp4"

IMG_FORMATS = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
VID_FORMATS = ['.mov', '.avi', '.mp4']

# Example usage :
# python3 overlay.py image.jpg logo.png -o 0.5

def readCommand():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagePath", help="path to the background image")
    parser.add_argument("overlayPath", help="path to the image overlay")
    parser.add_argument("-o", "--opacity", type=float, default=0.5, help="overlay opacity")
    parser.add_argument("-l", "--lines", action='store_true', help="show selected corners and lines")
    parser.add_argument("-v", "--view_img", action='store_true', help="display intermediate images")
    args = vars(parser.parse_args())
    return args


def select_corners(event, x, y, flags, param):
    # grab reference to the global variables
    global ptsDst
    global isEstimated
    global h
    global warped_image

    # Read callback parameters
    imageSrc, imageTemp,imageFinal, overlaySrc, opacity = param
    overlayWidth = overlaySrc.shape[0]
    overlayHeight = overlaySrc.shape[1]

    # Four corners of overlay image
    pts_src = np.array([[0, 0],[0,overlayWidth],[overlayHeight, overlayWidth],[overlayHeight,0]])

    # Points selection order to get the right overlay position. Top left - Bottom left - Bottom right - Top right
    if len(ptsDst) < 4:
        if event == cv2.EVENT_LBUTTONDOWN:
            print("A new corner has been added, x : {}, y : {}".format(x,y))
            ptsDst.append((x,y))
            cv2.circle(imageSrc, (x,y), POINT_RADIUS, COLOR_RED, FILL) 

            if len(ptsDst) > 1:
                cv2.line(imageSrc,ptsDst[len(ptsDst)-1],ptsDst[len(ptsDst)-2],COLOR_BLUE,LINE_THICKNESS)

            if len(ptsDst) == 4:
                cv2.line(imageSrc,ptsDst[len(ptsDst)-1],ptsDst[0],COLOR_BLUE,LINE_THICKNESS)

            cv2.imshow("image", imageSrc)

    else:
        if isEstimated == False:
            # Calculate Homography matrix mapping the source overlay image into the chosen destination points
            h, status = cv2.findHomography(pts_src, np.array(ptsDst))
            isEstimated = True
            print("[INFO] Homography has been successfuly estimated.")
            # Warp source image to destination based on homography (w/ black background)
            warped_image = cv2.warpPerspective(overlaySrc, h, (imageSrc.shape[1],imageSrc.shape[0]))

            # Get the transparent overlay (with the original image background)
            # Using transparent overlay allows to remove the logo transparency
            im_out = transparentOverlay(imageTemp, warped_image, ptsDst)

            # Blend the two images
            # Opacity is the transparency of the overlay. It is also possible to adjust the beta and gamma parameters from the OpenCV function (See cv2.addWeighted)
            # docs : https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html
            cv2.addWeighted(im_out, opacity, imageFinal, 1-opacity , 0, imageFinal)

            # Display images
            cv2.imshow("Source Overlay Image", overlaySrc)
            cv2.imshow("Warped Overlay Image", warped_image)
            cv2.imshow("Transparent Overlay Image", im_out)
            cv2.imshow("Destination Image", imageFinal)

            # Save final image
            cv2.imwrite(RESULT_IMG, imageFinal)

    
# This method applies the warped overlay (Transformed overlay with black background) to the original image
# args : source image, warped overlay image, destination points 
# returns the warped overlay with the source background
def transparentOverlay(src, overlay, dstPts):

    # Fix on non-square dstPts
    ymin = min(list(map(lambda pos: pos[1], dstPts )))
    ymax = max(list(map(lambda pos: pos[1], dstPts )))
    xmin = min(list(map(lambda pos: pos[0], dstPts )))
    xmax = max(list(map(lambda pos: pos[0], dstPts )))

    
    
    # Slow loop
    """
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            if overlay[i][j][3] > 0: # alpha channel
                alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
                src[i][j] = alpha * overlay[i][j] + (1 - alpha) * src[i][j]
    """

    # Numpy loop (fast-loop)
    src_sub = np.asarray(src[ymin:ymax,xmin:xmax])
    overlay_sub = np.asarray(overlay[ymin:ymax, xmin:xmax])

    # Extract transparency layer
    overlay_alpha = overlay_sub[:,:,3]
    overlay_alpha = overlay_alpha / 255.0
    overlay_alpha_3d = np.repeat(overlay_alpha[:, :, np.newaxis], 4, axis=2)



    # Element wise condition replacing
    # Replace the src image pixel by overlay pixel if pixel is located in the same position
    transparent_sub = np.where(overlay_alpha_3d / 255.0 > 0, overlay_alpha_3d * overlay_sub + (1-overlay_alpha_3d) * src_sub, src_sub )
    
    # Replace on original image
    src[ymin:ymax, xmin:xmax] = transparent_sub
    


    return src


if __name__ == '__main__' : 

    # Initialize the list of destination reference points
    ptsDst = []
    # Homography initialization
    h = []
    isEstimated = False
    # Warped image initialization
    warped_image = []

    args = readCommand()


    # If the overlay is applied on a image
    if os.path.splitext(args["imagePath"])[-1].lower() in IMG_FORMATS :
        # Read source image.    
        imageSrc = cv2.imread(args["imagePath"], -1)
        # Add the extra dimension related to transparency if extension does not handle alpha channel (..jpg)
        if imageSrc.shape[2] != 4:
            imageSrc = np.dstack([imageSrc, np.ones((imageSrc.shape[0], imageSrc.shape[1]), dtype="uint8") * 255])
        imageSrc = imutils.resize(imageSrc, width=1920)

        # Clone image to reset the overlay
        imageReset = imageSrc.copy() 
        imageTemp = imageSrc.copy()
        imageFinal = imageSrc.copy()

        # Read overlay image
        overlaySrc = cv2.imread(args["overlayPath"],cv2.IMREAD_UNCHANGED)
        if overlaySrc.shape[2] != 4:
            raise Exception('Overlay (logo) image should have BGRA pixel format i.e with a transparency channel (.png). Current overlay shape {}'.format(overlaySrc.shape))
        overlaySrc = imutils.resize(overlaySrc, width = 400)

        # Set image window for callback
        cv2.namedWindow("image")
        # Set mouse callback
        cv2.setMouseCallback("image", select_corners, [imageSrc,imageTemp,imageFinal,overlaySrc, args['opacity']])

        while True:

            # display the image and wait for a keypress
            cv2.imshow("image", imageSrc)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the selected corners
            if key == ord("r"):
                imageSrc = imageReset.copy()
                ptsDst = []

            # if the 'c' key is pressed, break from the loop
            elif key == ord("q"):
                break


    # If the overlay is applied on a video
    elif os.path.splitext(args["imagePath"])[-1].lower() in VID_FORMATS:
        
        # Read the video
        cap = cv2.VideoCapture(args["imagePath"])
        if (cap.isOpened() == False):
            raise Exception('Unable to open the camera feed')
        
        # Initialize the video writer
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #vid_writer = cv2.VideoWriter(RESULT_VID, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (1920, 1080))
        vid_writer = cv2.VideoWriter(RESULT_VID, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (1920, 879))
        # Read overlay image
        overlaySrc = cv2.imread(args["overlayPath"],cv2.IMREAD_UNCHANGED)
        if overlaySrc.shape[2] != 4:
            raise Exception('Overlay (logo) image should have BGRA pixel format i.e with a transparency channel (.png). Current overlay shape {}'.format(overlaySrc.shape))
        overlaySrc = imutils.resize(overlaySrc, width = 400)
        
        opacity = args['opacity']

        # Count the number of frames read
        readFrames = 0


        while(True):
            ret, frame = cap.read()
            
            if ret == True:
                # Start timer
                start_time = time.time()
                # Add the extra dimension related to transparency if extension does not handle alpha channel (..jpg)
                if frame.shape[2] != 4:
                    frame = np.dstack([frame, np.ones((frame.shape[0], frame.shape[1]), dtype="uint8") * 255]) # Slow operation ...
                frame = imutils.resize(frame, width=1920)
                # Clone image to apply transparency on overlay
                imageTemp = frame.copy()
                imageFinal = frame.copy()
                # Clone image to reset the overlay
                imageReset = frame.copy() 
                

                # Apply the overlay on the first frame 
                if readFrames == 0:
                    
                    print("Waiting for the destination points ... \n 1.top-left, 2.bottom-left, 3. bottom-right, 4. top-right")
                    # Set image window for callback
                    cv2.namedWindow("image")
                    # Set mouse callback
                    imageSrc = frame
                    cv2.setMouseCallback("image", select_corners, [imageSrc,imageTemp,imageFinal,overlaySrc, args['opacity']])

                    while True:
                        # display the image and wait for a keypress
                        cv2.imshow("image", imageSrc)
                        key = cv2.waitKey(1) & 0xFF

                        # if the 'r' key is pressed, reset the selected corners
                        if key == ord("r"):
                            imageSrc = imageReset.copy()
                            ptsDst = []

                        # if the 'c' key is pressed, break from the loop
                        elif key == ord("q"):
                            break
                        
                        # if the homography is estimated we can apply it on next frames
                        elif isEstimated:
                            cv2.setMouseCallback("image",lambda *args : None) # remove callback
                            break

                # Then use the same destination points on next frames
                else:   
                    
                    # Warp source image to destination based on homography
                    # warped_image = cv2.warpPerspective(overlaySrc, h, (frame.shape[1],frame.shape[0]))
                    
                    # Get the transparent overlay (with the original image background)
                    # Using transparent overlay allows to remove the logo transparency
                    im_out = transparentOverlay(imageTemp, warped_image, ptsDst)
                    
                    # Blend the two images
                    # Opacity is the transparency of the overlay. It is also possible to adjust the beta and gamma parameters from the OpenCV function (See cv2.addWeighted)
                    # docs : https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html
                    
                    cv2.addWeighted(im_out, opacity, imageFinal, 1-opacity , 0, imageFinal)
                

                imageFinal = imageFinal[:,:,:3]
                print("Image final shape : {}".format(imageFinal.shape))
                #Display the image and wait for a keypress
                cv2.imshow("Final image", imageFinal)

                # Write image in video writer
                vid_writer.write(imageFinal)
 
                # Press 'Q' to quit the program
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                

                # End timer
                print("[INFO] Frame {}/{} Done. {:.3f} s".format(readFrames + 1,totalFrames,float(time.time() - start_time)))
                
                # Increment number of frames read
                readFrames += 1      

            else:
                break

        print("[INFO] Clean memory ... ") 
        vid_writer.release()
        
    else:
        raise Exception('Wrong source file format. {} was given but only {} \n {} are supported'.format(os.path.splitext(args["imagePath"])[-1], IMG_FORMATS, VID_FORMATS))

    cv2.destroyAllWindows()