import cv2 
import os 
import argparse

def readCommand():
    parser = argparse.ArgumentParser()
    # Positional arguments
    parser.add_argument('videoPath', help='Source path to  video file')
    parser.add_argument('imagesPath',help='Destination saving path to images')

    # Optional arguments
    parser.add_argument('-s', '--start',dest='start', type=int, default=0, help='starting time in seconds')
    parser.add_argument('-p', '--period',dest='period', type=int, default=1, help='images period. Default is 1 meaning that it saves EVERY frames ')
    args = parser.parse_args()
    return args

def videoToImages(videoPath, imagesPath, imagesPeriod, startTime):
    # Read the video from specified path 
    cam = cv2.VideoCapture(videoPath) 
    if(cam.isOpened() == 0): 
            print("Video initialization failed. Please verify your video file input. \n")

    # creating a folder named data 
    try: 
        if not os.path.exists(imagesPath): 
            os.makedirs(imagesPath) 

    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of images') 
    
    # Frame counter
    currentFrame = 0
    FPS = 30

    while(True): 
        
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret: 
            # if video is still left continue creating images 
            
            if (currentFrame > startTime * FPS ):
                # writing the extracted images 
                if(currentFrame % imagesPeriod == 0):
                    name = imagesPath + '/frame' + str(currentFrame) + '.jpg'
                    print ('Creating...' + name) 
                    cv2.imwrite(name, frame) 
        
            currentFrame += 1
            
        else: 
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    args = readCommand()
    videoToImages(args.videoPath, args.imagesPath, args.period, args.start)