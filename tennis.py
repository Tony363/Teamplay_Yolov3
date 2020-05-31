import math
import cv2
import imutils
import numpy as np
import time

CODE_INFO = "[INFO] "
CODE_ERROR = "[ERROR] "
TOP_BOTTOM_DISTANCE = 10.97 # Reference meters distance from top court line to bottom court line

class Queue:
  "A container with a first-in-first-out (FIFO) queuing policy."
  def __init__(self, maxBalls):
    self.list = []
    self.max = maxBalls

  def push(self,item):
    "Enqueue the 'item' into the queue"
    if (len(self.list) == self.maxBalls):
        self.list.pop()

    self.list.insert(0,item)

  def pop(self):
    """
      Dequeue the earliest enqueued item still in the queue. This
      operation removes the item from the queue.
    """
    return self.list.pop()

  def isEmpty(self):
    "Returns true if the queue is empty"
    return len(self.list) == 0



class TennisState:

    " Tennis Game state including ball, players position, ball tracer, court key points ..."
    def __init__(self, maxBalls):
        self.players = [
            { "box" : (0,0,0,0),
            "conf" : 0},
            { "box" : (0,0,0,0),
            "conf" : 0}
        ]
        self.ball = {
            "box" : (0,0,0,0),
            "conf" : 0
        }

        self.courtIsDetected = False
        self.court = []

        self.distances = [0,0] # Walked distance of left and right players
        self.scaleDistance = 1
        self.balls = Queue(maxBalls=maxBalls) # Trace last balls positions
        
        self.trainingBalls = [] # (xA, yA, xB, yB) - Store in memory training balls in the FIRST frame to avoid tracking on them
        self.lastFrameBalls = [] # Store last frame balls position to display only moving ball in the current frame
        self.currentFrameBalls = []

        # Patch inference variables
        self.lastFrameBallsPatch = 30 * [[]] # 30 patch # element i : lastFrameBalls of patch i
        self.currentFrameBallsPatch = 30 * [[]]

        self.timeWatch = 0


    # Update the player distance based on the new bouding rectangle
    # args : xyxy new bounding rect
    #        index index of players. 0 is left player and 1 is right player.
    def updatePlayersDistance(self,index,xyxy):
        self.distances[index] +=  (getEuclideanDistance(getRectCenter(self.players[index]["box"]), xyxy)  * TOP_BOTTOM_DISTANCE) / (self.scaleDistance*100)


    # Identify the real players based on the detected persons in the frame
    def identifyPlayersAndPlot(self,im, leftPersons, rightPersons, colors):
        persons = []
        if leftPersons != []:
            leftPlayer = max(leftPersons, key=lambda col : col[1]) # max applied on confidence
            self.updatePlayersDistance(0, leftPlayer[0])
            self.players[0]["box"] = leftPlayer[0]
            self.players[0]["conf"] = leftPlayer[1]
        if rightPersons != []:
            rightPlayer = max(rightPersons, key=lambda col: col[1])
            self.updatePlayersDistance(1, rightPlayer[0])
            self.players[1]["box"] = rightPlayer[0]
            self.players[1]["conf"] = rightPlayer[1]
                    
        for index,player in enumerate(self.players):
            label = '%s %.2f' % ("person", player["conf"])
            #label = '%s %.2f' % (names[int(player[2])], player["conf"])
            plot_players_box(player["box"], im, label = label, color=colors[int(0)])
        
        plot_players_distances(self.distances, im)

    # Update the Time watch based on the video fps
    def updateTimeWatch(self, img, fps):
        self.timeWatch += 1/fps
        strTime = time_convert(self.timeWatch)
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        cv2.putText(img, strTime, (img.shape[1] - 500, img.shape[0]-100), cv2.FONT_HERSHEY_DUPLEX, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    # Clear object detections to keep only relevant detections with rules ( 2 players and 1 ball)
    # Return True if the object need to be displayed through detectSmall.py
    def clearDetections(self, xyxy, conf, className):
        if className == "sports ball":

            # Check ball size
            # False positive rejection by area size
            if getArea(xyxy) > 10000:
                return False

            self.currentFrameBalls.append(xyxy)

            # Initialize training balls (non-moving ball in first frames)
            if self.lastFrameBalls == []:
                #self.trainingBalls.append(xyxy)
                return True

            else:
            # Check if detected ball is in the same position as at least ONE ball of the last frame balls
                for i,ball in enumerate(self.lastFrameBalls):
                    print("ball : {}, lastframe ball {} {}  . IoU = {}".format(tensorPointToList(xyxy), i,tensorPointToList(ball),bb_intersection_over_union(xyxy,ball)))
                    if bb_intersection_over_union(xyxy,ball) > 0.1:
                        return False
                
                # Keep only moving balls
                print("[INFO] Real ball detected !")
                self.ball['box'] = xyxy
                self.ball['conf'] = conf
                self.balls.push(xyxy)
                return True

                
        elif className == "person":
            if self.players[0]['box'] == (0,0,0,0):
                self.players[0]['box'] = xyxy
                self.players[0]['conf'] = conf
                return True
            elif self.players[1]['box'] == (0,0,0,0):
                self.players[1]['box'] = xyxy
                self.players[1]['conf'] = conf
                return True
            elif conf > self.players[0]['conf']:
                self.players[0]['box'] = xyxy
                self.players[0]['conf'] = conf
            else:
                return False
                
        elif className == "tennis racket":
            return True

        return True

    def clearPatchDetections(self, indexPatch, xyxy, conf, className):
        if className == "sports ball":

            # Check ball size
            # False positive rejection by area size
            if getArea(xyxy) > 10000:
                return False


            self.currentFrameBallsPatch[indexPatch].append(xyxy)
            # Initialize training balls (non-moving ball in first frames)
            if self.lastFrameBallsPatch[indexPatch] == []:
                #self.trainingBalls.append(xyxy)
                return True
            else:
            # Check if detected ball is in the same position as ONE of the last frame balls
                for i,ball in enumerate(self.lastFrameBallsPatch[indexPatch]):
                    print("ball : {}, lastframe ball {} {}  . IoU = {}".format(tensorPointToList(xyxy), i,tensorPointToList(ball),bb_intersection_over_union(xyxy,ball)))
                    if bb_intersection_over_union(xyxy,ball) > 0.1:
                        return False
                
                print("[INFO] Real ball detected !")
                self.ball['box']= xyxy
                self.ball['conf']= conf
                return True

                
        elif className == "person":
            if self.players[0]['box'] == (0,0,0,0):
                self.players[0]['box'] = xyxy
                self.players[0]['conf'] = conf
                return True
            elif self.players[1]['box'] == (0,0,0,0):
                self.players[1]['box'] = xyxy
                self.players[1]['conf'] = conf
                return True
            elif conf > self.players[0]['conf']:
                self.players[0]['box'] = xyxy
                self.players[0]['conf'] = conf

            else:
                return False


        elif className == "tennis racket":
            return True



        return True

# We extract the tennis court using the dominant color in the center of the image.
def detectTennisCourt(img, plot = False, offset = 100):

    # Extract center of img
    centerX = int(img.shape[1] / 2)
    centerY = int(img.shape[0] / 2)
    cropImg = img[centerY - offset: centerY + offset, centerX-offset : centerX + offset]
    #cv2.imshow("Center", cropImg)

    # HSV of the region of interest
    hsvRoi = cv2.cvtColor(cropImg,cv2.COLOR_BGR2HSV) # shape : (2*offset,2*offset, 3 )
    histRoi = cv2.calcHist( [hsvRoi], [0, 1], None, [180, 256], [0, 180, 0, 256] ) # shape : (180,256 )
    
    # Get peak index in the H-S histogram
    peakHue, peakSaturation = np.unravel_index(np.argmax(histRoi, axis=None), histRoi.shape)

    # Plot H-S histogram
    if plot:
        plt.imshow(histRoi,interpolation = 'nearest')
        plt.title("2D Hue/Saturation Histogram")
        plt.ylabel("Hue")
        plt.xlabel("Saturation")
        plt.colorbar()
        #plt.imshow(histRoi)
        plt.show()

    # Range value for Hue, Saturation and Value colors space
    colorMin = (int(peakHue -2), 140, 55)
    colorMax = (int(peakHue +2), 200, 255)
    # HSV of orginal image
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Build the mask based on the HSV threshold
    mask = cv2.inRange(hsv, colorMin, colorMax)
    
    #cv2.imshow("Mask", mask)

    
    # Apply mask to the orginal image
    #result = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow("resultIni", result)

    # Close Image (Dilation followed by an erosion) - first transformation
    
    kernelCloseSize = (30,15)
    kernelClose =  cv2.getStructuringElement(cv2.MORPH_RECT, kernelCloseSize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelClose)
    #cv2.imshow("Closed Mask", mask)
    

    # Erode left objects
    kernelErosionSize = (15,15)
    kernelErosion =  cv2.getStructuringElement(cv2.MORPH_RECT, kernelErosionSize)
    mask = cv2.erode(mask, kernelErosion)
    #cv2.imshow("Eroded mask", mask)


    # Dilate again
    kernelDilationSize = (30,30)
    kernelDilate =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelDilationSize)
    mask = cv2.dilate(mask, kernelDilate)
    #cv2.imshow("Dilated mask", mask)

    # Apply canny edge detector. Default threshold set to 100.
    cannyMask = cv2.Canny(mask, 100, 100 * 2)
    #cv2.imshow("Canny mask", cannyMask)
    # Find contours
    contours, hierarchy = cv2.findContours(cannyMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get biggest contour (should be the game court)
    contoursArea = [cv2.contourArea(contour) for _, contour in enumerate(contours)]
    contour = contours[np.argmax(contoursArea)]

    # Draw contours
    cv2.drawContours(img, [contour], 0, (0,255,0), 3)
    #cv2.imshow("Game Contour", img)


    
    corners = getCornersFromContour(contour)
    if corners is not None:
        corners = corners.reshape(4,1,2)
        cv2.drawContours(img, corners, -1, (0, 0, 255), 10)
        im = imutils.resize(img, width=1920)
        cv2.imshow("CORNERS", im)

    # Find contours Vertices (corners)
    # todo : check houghlines transform if can be used on non linear plan
    #Apply mask to the orginal image
    #result = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow("result", result)
    
    
    #cv2.waitKey(0)
    return corners

# This functions returns the tennis court estimated corners.
# Use poly curves approximation to get a smaller set of points that approximates the initial contour
# NB : epsilon argument is the accuracy of the approximation. There lower it is the more accurate the approx is meaning that there will be more points.
# Returns the left top, right top, left bottom, right bottom corners.

def getCornersFromContour(contour):

    epsilon = 0.01 * cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    print("approx shape : " ,approx.shape)
    if (len(approx) < 4):
        print(CODE_INFO, "Court segmentation approximation failed. Approximation does not have enough points. Please check epsilon parameters")
        return None
    else:
        print(CODE_INFO, "Court segmentation approximation success. ")
        print(approx.shape)
        approx = approx.reshape(approx.shape[0],approx.shape[2])
        sortedByHeightApprox =  approx[np.argsort(approx[:, 1])] # sort by second column. y position

        # Get top corners
        topCorners = sortedByHeightApprox[:2]
        sortedTopCorners = topCorners[np.argsort(topCorners[:,0])] # sort by second column. x position
        topLeft = sortedTopCorners[0]
        topRight = sortedTopCorners[1]

        # Get bottom corners
        bottomPoints = sortedByHeightApprox[:-2]
        sortedBottomPoints = bottomPoints[np.argsort(bottomPoints[:,0])] # sort by second column. x position
        bottomLeft = sortedBottomPoints[0]
        bottomRight = sortedBottomPoints[-1]

        return np.asarray([topLeft, topRight, bottomLeft, bottomRight])




def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def tensorPointToList(xyxy):
    return (int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3]))


def getArea(xyxy):
    xA, yA, xB, yB = tensorPointToList(xyxy)
    return (xB - xA) * (yB - yA)

def getRectCenter(xyxy):
    x,y,p,q = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    return x + (p-x)/2 , y + (q-y)/2

def getManhattanDistance(xy,pq):
        return (abs(xy[0]) - pq[0]) + abs(xy[1] - pq[1])


def getEuclideanDistance(xy,pq):
        distance = math.sqrt((xy[0] - pq[0])**2 + (xy[1] - pq[1])**2)
        return math.sqrt((xy[0] - pq[0])**2 + (xy[1] - pq[1])**2)


# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_ball_history(balls, img, line_thickness=None):

    for ballxy in balls.list:
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = [0,255,255]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled


def plot_players_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_players_distances(distances, img):
    h,w = img.shape[0], img.shape[1]
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)

    for index, distance in enumerate(distances):
        label = str(round(distance,2)) + "m"
        # Left player
        if index == 0:
            cv2.putText(img, label, (10, 100), cv2.FONT_HERSHEY_DUPLEX, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # Right player
        else:
            cv2.putText(img, label, (w - 500, 100), cv2.FONT_HERSHEY_DUPLEX, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def time_convert(sec):
    mins = sec // 60
    sec = round(sec % 60,4)
    hours = mins // 60
    mins = mins % 60
    strTime = "{0}:{1}".format(int(mins),sec)
    return strTime

