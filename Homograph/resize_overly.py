import cv2 as cv
import numpy as np
import imutils
import argparse

def combine_two_color_images(image1, image2):
    foreground, background = image1.copy(), image2.copy()
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    alpha =0.5
    # do composite on the upper-left corner of background image.
    blended_portion = cv.addWeighted(foreground,
                alpha,
                background[:foreground_height,:foreground_width,:],
                1 - alpha,
                0,
                background)
    background[:foreground_height,:foreground_width,:] = blended_portion
    background = imutils.resize(background,width=1080)
    cv.imshow('composited image', background)
    cv.waitKey(0)

def arguments():
    parser = argparse.ArgumentParser(description='overlay videos')
    parser.add_argument('--background_vid',type=str,required=True,help='background video')
    parser.add_argument('--subframe_vid',type=str,required=True,help='subframe video')
    args = parser.parse_args()
    return parser,args

def overlay_subframe(background,subframe):
    h1,w1 = background.shape[:2]
    h2,w2 = subframe.shape[:2]

    shrunk_img = cv.resize(subframe,(w1//3,h1//3))
    shrunkh,shrunkw = shrunk_img.shape[:2]

    background[1424:h1//3+1424,1550:w1//3+1550] = shrunk_img
    background_resized = imutils.resize(background,width=1080)
    return background,background_resized

def overlay_vid(parser,args):
    vid_background = cv.VideoCapture("{background}".format(background=args.background_vid))
    vid_subframe = cv.VideoCapture("{subframe}".format(subframe=args.subframe_vid))
    vid_writer = None
    read_vids = True
    while read_vids:
        retB,frameB = vid_background.read()
        retS,frameS = vid_subframe.read()
        if retB and retS:
            if vid_writer is None:
                h,w = frameB.shape[:2]
                fourcc = cv.VideoWriter_fourcc(*'mp4v') 
                fps = vid_background.get(cv.CAP_PROP_FPS)
                vid_writer = cv.VideoWriter("../output/output.mp4",fourcc,fps,(w,h),True)  
            vid_frame,vid_frameR = overlay_subframe(frameB,frameS)
            cv.imshow('overlay',vid_frameR)
            if cv.waitKey(1) == ord('q'):
                read_vids = False
            vid_writer.write(vid_frame)
        else:
            print('end of video')
            read_vids = False
    vid_writer.release()
    vid_background.release()
    vid_subframe.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser,args = arguments()
    overlay_vid(parser,args)

