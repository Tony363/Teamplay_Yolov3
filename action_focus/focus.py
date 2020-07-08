# This file includes methods enabling cropping a sub video from the stitched video. The sub video will follow the game's action
# using players' position and ball' position when the latter is detected.

# Manual panning centroid values
horizontalPan = [ (x*10,800) for x in range(460)]
count = 0
forward = True

# args : players' position, ball's position, last centroid
# returns the virtual camera center
def getVirtualCameraCenter(playersPosition, ballPosition, centroid, motionWeight = 0.9):

    # Calculate the motion center (camera view center)
    # By using detected players and detected ball
    
    x = [p[0] for p in playersPosition]
    y = [p[1] for p in playersPosition]

    avgCentroid = (int(sum(x)/len(playersPosition)), int(sum(y) /len(playersPosition)))

    if (centroid == (0,0)):
        centroid = avgCentroid
    else :
        centroid = tuple(map(lambda oldCentroidCoord, avgCentroidCoord: 
                            int(motionWeight * oldCentroidCoord + (1 - motionWeight) * avgCentroidCoord), centroid, avgCentroid))


    return centroid


# Manual horizontal panning using predefined camera center values 
def getVirtualCameraCenterTest():
    global count, forward

    if forward:
        if count == len(horizontalPan) - 1: # Right end reached
            count -= 1
            forward = False

        centroid = horizontalPan[count] 
        count += 1
    else:
        if count == 0: # left end reached
            count += 1
            forward = True
        centroid = horizontalPan[count] 
        count -= 1


    return centroid
    

# The faster the motion the larger view the result needs to be
# Args : last camera view center (x,y), new camera view center (x,y), virtual camera offset width and height
# returns the final camera view
# Ex : Virtual camera of size 1920x1080 -> offsetWidth = 960, offsetHeight = 540
#                           2560x1440-> offsetWidth = 1280, offsetHeight = 720
def getCroppedImage(img, centroid, offsetWidth=1280, offsetHeight = 720):
    # For simplicity we set a fix motion speed so that the camera view has a fix size
   
    h,w,_ = img.shape

    # x-axis
    if centroid[0]-offsetWidth < 0:
        xmin = 0
        xmax = offsetWidth*2
    elif centroid[0] + offsetWidth > w:
        xmin = w - offsetWidth* 2
        xmax = w
    else:
        xmin = max(0, centroid[0] - offsetWidth)
        xmax = min(w, centroid[0] + offsetWidth)
    
    # y-axis
    if centroid[1] - offsetHeight < 0:
        ymin = 0
        ymax = offsetHeight * 2
    elif centroid[1] + offsetHeight > h:

        ymin = h - offsetHeight*2
        ymax = h
    else:
        ymin = max(0, centroid[1] - offsetHeight)
        ymax = min(h,centroid[1] + offsetHeight)

    # Crop a fixed size img using centroid
    return xmin ,xmax, ymin, ymax

