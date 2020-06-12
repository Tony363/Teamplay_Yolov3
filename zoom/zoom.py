from tennis.tennis import *
# This method will recalculate the center of the cropped image to run a smooth zoom on the given object
# Instead of cropping the image with the object's new bounding box center, we calculate a new center with a motion weight.
# The higher is the motion weight, the slower to the object's new bounding box the new center will converge.
# The object detection model may predict two bounding boxes on two consecutive frames whose center are far from each other leading to a "jumping camera".

def getZoomCentroid(oldCentroid, objectCentroid, motionWeight = 0.9):
    
    # Initialize centroid on the first detected object
    if (oldCentroid == (0,0)):
        centroid = objectCentroid

    # Apply motion weight to calculate new centroid
    else :
        centroid = tuple(map(lambda oldCentroidCoord, objectCentroidCoord: 
                                int(motionWeight * oldCentroidCoord + (1 - motionWeight) * objectCentroidCoord), oldCentroid, objectCentroid))

    return centroid

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def getCroppedImage(img,centroid, offset):
    if centroid[0]-offset < 0:
        xmin = 0
        xmax = offset*2
    elif centroid[0] + offset > img.shape[1]:
        xmin = img.shape[1] - offset* 2
        xmax = img.shape[1]
    else:
        # todo max/min unecessary
        xmin = max(0, centroid[0] - offset)
        xmax = min(img.shape[1], centroid[0] + offset)
    if centroid[1] - offset < 0:
        ymin = 0
        ymax = 100 * 2
    elif centroid[1] + offset > img.shape[0]:
        ymin = img.shape[0] - offset*2
        ymax = img.shape[0]
    else:
        ymin = max(0, centroid[1] - offset)
        ymax = min(img.shape[0],centroid[1] + offset)
    # Crop a fixed size img using centroid
    return xmin ,xmax, ymin, ymax

# zoomin func
def zoomin(zoom,im0,xyxy,count,lastCentroid, motionWeight):
    objectCentroid = getRectCenter(xyxy)
    centroid = getZoomCentroid(lastCentroid, objectCentroid, motionWeight)
    lastCentroid = centroid
    if lastCentroid == (0,0):
        crop = im0[:800,:800]
        count += 1 
        return zoom,crop,count,lastCentroid
    xmin,xmax,ymin,ymax = getCroppedImage(im0,centroid, 400)
    crop = im0[int(ymin):int(ymax),int(xmin):int(xmax)] 
    crop = increase_brightness(crop,value=20)
    count += 1 
    return zoom,crop,count, lastCentroid
 


# zoom out func
def zoom_out(zoom,im0):
    crop = im0[:,:]
    zoom = False
    return zoom,crop

# "Smooth zoom to detection"
def zoom_player(zoom,player,xyxy,speed,shift=20):
    centroid = getRectCenter(xyxy)
    xmin,xmax,ymin,ymax = getCroppedImage(im0,centroid, 400)
    rows,cols,rgb = player.shape
    if rows > int(ymin) and cols > int(xmin):
        speed += shift
    else:
        zoom = False
        player = im0
        speed = 0
        return zoom,player,speed
    frame = player[speed:int(xmax),speed:int(ymax)]
    return zoom,frame,speed

def zoom_ball1(zoom,frame, count):
    # if count > 85 and count < 110:
    crop = frame[1000:1400,3800:4200] 
    crop = increase_brightness(crop,value=20)
    count += 1 
    if count > 93 and count < 100:
        circle = cv2.circle(crop,(232,183),radius=3,color=(45, 255, 255),thickness=-1)
        return zoom,circle,count
    return zoom,crop,count

def zoom_impact(zoom,frame,count):
    # crop = frame[1160:1190,4020:4070]
    crop = frame[1150:1200,4010:4060]
    crop = increase_brightness(crop,value=20)
    count += 1 
    (h, w) = crop.shape[:2]
    if count > 100 and count < 140:                      
        ellipse = cv2.ellipse(crop,(w//2,h//2),(6,3),0,1,360,color=(45,255,255),thickness=-1)
        return zoom,ellipse,count
    return zoom,crop,count



