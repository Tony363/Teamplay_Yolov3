import cv2 
import os 
import argparse


VIDEO_EXT = [".MP4", ".mp4", ".avi"]

# usage : 
# Extract first frame of video.mp4 and save it in video folder
# python3 videoToImages.py video.mp4 -p 0

# Exctract first frame of video.mp4 and save it in output_path/data/ folder
# python3 videoToImages.py video.mp4 -i output_path/data/ -p 0

def readCommand():
    parser = argparse.ArgumentParser()
    # Positional arguments
    parser.add_argument('videoPath', help='Source path to  video file')
    

    # Optional arguments
    parser.add_argument('-i','--imagesPath',help='Destination saving path to images')
    parser.add_argument('-s', '--start',dest='start', type=int, default=0, help='starting time in seconds')
    parser.add_argument('-p', '--period',dest='period', type=int, default=1, help='images period. Default is 1 meaning that it saves EVERY frames. 60 means that a frame is extracted every 60 frames. 0 means only first frame is extracted')
    args = parser.parse_args()
    return args

def videoToImages(videoPath, imagesPath, imagesPeriod, startTime):




    # Read the video from specified path 
    cam = cv2.VideoCapture(videoPath) 
    if(cam.isOpened() == 0): 
            print("Video initialization failed. Please verify your video file input. \n")


    if imagesPath is None:
        imagesPath = os.path.dirname(videoPath) + "/images/"


    fileName = os.path.basename(videoPath)
    fileNameNoExtension = os.path.splitext(fileName)[0]
    print(fileNameNoExtension)

    # creating a folder named data in the video folder
    try: 
        if not os.path.exists(imagesPath): 
            os.makedirs(imagesPath) 

    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of images') 
    
    # Frame counter
    currentFrame = 0
    FPS = cam.get(cv2.CAP_PROP_FPS)

    while(True): 
        
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret: 
            # start extracting images only after a certain period
            if (currentFrame >= startTime * FPS ):
                # write only first frame
                if (imagesPeriod == 0):
                    name = imagesPath + fileNameNoExtension + "_" + str(currentFrame) + '.jpg'
                    print ('Creating...  ' + name) 
                    cv2.imwrite(name, frame)
                    break

                # writing the extracted images 
                if(currentFrame % imagesPeriod == 0):
                    name = imagesPath + fileNameNoExtension + "_" +  str(currentFrame // imagesPeriod) + '.jpg'
                    print ('Creating...  ' + name) 
                    cv2.imwrite(name, frame) 
        
            currentFrame += 1
            
        else: 
            print("[INFO] Reading frame, cam.read() failed.")
            break

    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    args = readCommand()

    # Check if videoPath is a folder or a file
    if os.path.isdir(args.videoPath):
        # Assume directory contains only videos
        for filename in os.listdir(args.videoPath):
            if(os.path.splitext(filename)[1] in VIDEO_EXT):
                videoToImages(args.videoPath, args.imagesPath, args.period, args.start)
                

    elif os.path.isfile(args.videoPath):
        if(os.path.splitext(args.videoPath)[1] in VIDEO_EXT):
            videoToImages(args.videoPath, args.imagesPath, args.period, args.start)

    else:
        print("[INFO] Path is not a directory nor a file.")
            
