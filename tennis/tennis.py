import math
import cv2
import imutils
import numpy as np
import time
import os
import csv

CODE_INFO = "[INFO] "
CODE_ERROR = "[ERROR] "
TOP_BOTTOM_DISTANCE = 10.97 # Reference meters distance from top court line to bottom court line

# Number of balls in the ball tracing
MAX_BALLS = 10

# Start ball tracking frame. 
START_BALL_TRACK_FRAME = 140


# Initialize background substractor object 
bgSubstractor = cv2.createBackgroundSubtractorMOG2(); 


# Speed dictionnary (manually annoted)for tennis_play_1.mp4

SpeedDict = { "216" : {"start" : 138,"end" : 216 ,"distance" : 21.04},
             "333" : {"start" : 259,"end" : 333 ,"distance" : 22.07},
             "412" : {"start" : 358,"end" : 412 ,"distance" : 20.27},
             "518" : {"start" : 452,"end" : 518 ,"distance" : 21.57},
             "615" : {"start" : 549,"end" : 615 ,"distance" : 21.80},
             "719" : {"start" : 648,"end" : 719 ,"distance" : 20.11},
             "825" : {"start" : 751,"end" : 825 ,"distance" : 24.74},
             "935" : {"start" : 851,"end" :  935,"distance" : 20.99},
             "1038" : {"start" : 973,"end" :  1038,"distance" : 21.08},
             "1128" : {"start" : 1084,"end" :  1128,"distance" : 25.22},
             "1256" : {"start" : 1190,"end" :  1256,"distance" : 24.39},
             "1363" : {"start" : 1277,"end" :  1363,"distance" : 24.85} ,
             "1453" : {"start" : 1390,"end" :  1453,"distance" : 18.29},
             "1539" : {"start" : 1479,"end" :  1539,"distance" : 18.57},
             "1655" : {"start" : 1571,"end" :  1655,"distance" : 21.34},
             "1746" : {"start" : 1686,"end" :  1746,"distance" : 17.77},
             "1845" : {"start" : 1784,"end" :  1845,"distance" : 19.28},
             "1913": {"start" : 1868,"end" :  1913,"distance" : 25.61},
             "1944" : {"start" : 1913,"end" :  1944,"distance" : 11.64},
             "2129" : {"start" : 1990,"end" :  2129,"distance" : 25.22},
             "2223" : {"start" : 2129,"end" :  2179,"distance" : 13.91},
             "2275" : {"start" : 2223,"end" :  2275,"distance" : 24.18},
             "2325" : {"start" : 2275,"end" :  2325,"distance" : 14.07},
             "2458" : {"start" : 2348,"end" :  2458,"distance" : 20.86},
             "2516" : {"start" : 2458,"end" :  2516,"distance" : 7.80}
}

# Heart beat rate folder
HEART_BEAT_RATE_FOLDER = "tennis/data/20200602/heartrate"

class Queue:
  "A container with a first-in-first-out (FIFO) queuing policy."
  def __init__(self, maxBalls):
    self.list = []
    self.maxBalls = maxBalls

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

        # Court indicators
        self.courtIsDetected = False
        self.court = []

        # Players indicators
        self.distances = [0,0] # Walked distance of left and right players
        self.leftHeartRates = []
        self.rightHeartRates = []
        
        # Balls indicators
        self.lastLeftSpeed = 0
        self.lastRightSpeed = 0
        self.hit = 0
        self.scaleDistance = 1 # Scaled used to caculate players run distance using court reference size
        self.balls = Queue(maxBalls=maxBalls) # Trace last balls positions
        
        self.trainingBalls = [] # (xA, yA, xB, yB) - Store in memory training balls in the FIRST frame to avoid tracking on them
        self.lastFrameBalls = [] # Store last frame balls position to display only moving ball in the current frame
        self.currentFrameBalls = []
        self.lastFrameContours = [] # Store last frame contours to remove slow moving contour which is not considered as ball
        self.currentFrameContours = []

        # Patch inference variables
        self.lastFrameBallsPatch = 30 * [[]] # 30 patch # element i : lastFrameBalls of patch i
        self.currentFrameBallsPatch = 30 * [[]]

        # Game timer
        self.timeWatch = 0

        # Motion detector limit
        self.motionDetectionCorners = 0


    # Update the player distance based on the new bouding rectangle
    # args : xyxy new bounding rect
    #        index index of players. 0 is left player and 1 is right player.
    def updatePlayersDistance(self,index,xyxy):
        self.distances[index] +=  (getEuclideanDistance(getRectCenter(self.players[index]["box"]), xyxy)  * TOP_BOTTOM_DISTANCE) / (self.scaleDistance*100)
        self.distances[index] = round(self.distances[index],2)

    # Identify the real players based on the detected persons in the frame
    def identifyPlayersAndPlot(self,im, leftPersons, rightPersons, colors, plotBoxes = True):
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

        if plotBoxes:    
            for index,player in enumerate(self.players):
                label = '%s %.2f' % ("person", player["conf"])
                #label = '%s %.2f' % (names[int(player[2])], player["conf"])
                plot_players_box(player["box"], im, label = label, color=colors[int(0)])
        
        plot_players_distances(self.distances, im)

    # Calculate the ball speed from the racket hit to the ground hit and display it on video. 
    # For now we manually chose the frame to update. Later a model to detect a racket hit and a ground hit will be implemented.
    def updateHitSpeed(self,img, readFrame):
        key = str(readFrame)
        if key in SpeedDict.keys():
            if self.hit % 2 == 0: # left player hit
                self.lastLeftSpeed = self.getSpeed(SpeedDict[key]["start"],SpeedDict[key]["end"],SpeedDict[key]["distance"])
                
            else:
                self.lastRightSpeed = self.getSpeed(SpeedDict[key]["start"],SpeedDict[key]["end"],SpeedDict[key]["distance"])

            self.hit +=1


        drawSpeedText(img, self.lastLeftSpeed, "left")
        drawSpeedText(img, self.lastRightSpeed, "right")

    # Get the speed from the frames interval and the estimated distance
    def getSpeed(self,startFrame, endFrame, distance):
        speed = distance / ((endFrame - startFrame) /60)  # 60fps
        # Convert from m/s to km/h
        speed = round(speed * 3.6, 2)
        return speed

    # Update the player heart beat rate and display it based on players' position
    def updateHeartRate(self, img, readFrame, fps):
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness

  
        rateIndex = int(readFrame // fps)

        leftHeartRate = str(self.leftHeartRates[rateIndex])
        leftPlayerX, leftPlayerY = self.players[0]["box"][0], self.players[0]["box"][1]

        rightHeartRate = str(self.rightHeartRates[rateIndex])
        rightPlayerX, rightPlayerY = self.players[1]["box"][0], self.players[1]["box"][1]

        cv2.putText(img, leftHeartRate, (leftPlayerX, leftPlayerY - 50), cv2.FONT_HERSHEY_SIMPLEX, tl / 5, [0, 0, 255], thickness=int(tf/2), lineType=cv2.LINE_AA)
        cv2.putText(img, rightHeartRate, (rightPlayerX, rightPlayerY - 50), cv2.FONT_HERSHEY_SIMPLEX, tl / 5, [0, 0, 255], thickness=int(tf/2), lineType=cv2.LINE_AA)


    # Update the Time watch based on the video fps
    def updateTimeWatch(self, img, fps):
        self.timeWatch += 1/fps
        strTime = time_convert(self.timeWatch)
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        cv2.putText(img, strTime, (int(img.shape[1] / 2) - 200, 300), cv2.FONT_HERSHEY_SIMPLEX, tl / 4, [0, 255, 255], thickness=int(tf/2), lineType=cv2.LINE_AA)

    # Update the ball position and display its tracing using previous frames ball position
    # This method use contours detected by motion
    def updateBallPositionFromMotion(self,img, contours, startFrame):
        imageHeight = img.shape[0]
        imageWidth = img.shape[1]
        
        # Initialize ball distance to select the right contour
        ballDistance = 0 # distance  metric to find the right ball among the potential balls
        potentialBalls = [] # store all potential balls detected by motion detection

        
        # Initialize threshold for ball area
        if imageWidth <= 1920:
            ballAreaThres = 1400
            ballDistanceThres = 100
        else:
            ballAreaThres = 14500
            ballDistanceThres = 500

        # Calculate (x,y) position of motion boxes
        for _, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            print("[INFO] Detected Area : {}".format(area))
            x,y,w,h = cv2.boundingRect(contour)
            rectCenter = (x + int(w / 2), y + int(h/2))

            # Check if motion box center is in the court rectangle
            # TODO :  getTennisCourt function. Temporary court rectangle will be manually fix.
            courtX, courtY, courtX2, courtY2 = self.motionDetectionCorners
            imgRect = cv2.rectangle(img,  (courtX, courtY), (courtX2,courtY2),[155,0,0], 2 )

            if isInRectangle(rectCenter,courtX, courtY, courtX2, courtY2):

                # Display all objects detected in the court
                imgRect = cv2.rectangle(img,(x,y), (x+w, y+h), (255,0,0),2)
                # Save contour position
                self.currentFrameContours.append( (x,y,x+w,y+h))

                if(startFrame > START_BALL_TRACK_FRAME):

                    # Detect balls contours
                    if  area < ballAreaThres:
                        if self.ball["box"] == (0,0,0,0):
                            # First detected ball should be on the left side of the video
                            if (x < (imageWidth / 2) - 500):
                                self.ball["box"]= (x,y,x+w,y+h)
                        else:
                            potentialBalls.append((x,y,x+w,y+h))

                        
                    """
                    # Detect player contours
                    else:
                        # Detect left player
                        if(rectCenter[0] < imageWidth/2):
                            # initialize frame
                            if self.playersPosition[0] == (0,0):
                                self.playersPosition[0] = rectCenter
                                self.playersBoxes[0] = (x,y,x + w, y+h)
                            # Use IoU to track from previous frame detection
                            elif bb_intersection_over_union(self.playersBoxes[0], (x,y,x+w,y+h)) > 0.2:
                                # TODO : if area < previous area -> keep previous bbox
                                self.playersBoxes[0] = (x,y,x+w,y+h)
                        
                        # Detect right player
                        else:
                            if self.playersPosition[1] == (0,0):
                                self.playersPosition[1] = rectCenter
                                self.playersBoxes[1] = (x,y,x + w, y+h)
                                # Use IoU to track from previous frame detection
                            elif bb_intersection_over_union(self.playersBoxes[1], (x,y,x+w,y+h)) > 0.2:

                                self.playersBoxes[1] = (x,y,x+w,y+h)
                    """

        """
        # Draw players boxes
        for box in self.playersBoxes:
            pass
            imgRect = cv2.rectangle(imgRect,(box[0],box[1]), (box[2], box[3]), (0,255,0),2)
        """

        # Draw ball box
        # Clear potential balls
        
        # NB : instead of distance IoU could be used if we were detecting ball specifically but as we detect motion, the ball can be hidden my moving players.
        
        if self.ball["box"] != (0,0,0,0):
            # Remove potential balls that have same position as the latest detected.
            potentialBalls = [ ball for ball in potentialBalls if distance(getRectCenter(self.ball["box"]), getRectCenter(ball)) > 10 ]

            if potentialBalls != []:
                print("Potential balls {}".format(potentialBalls))
                
                # Choose the detected ball which is closest to the latest detected ball
                minDistanceBall = min(potentialBalls, key = lambda ball: distance(getRectCenter(self.ball["box"]), getRectCenter(ball)))
                print("[INFO] Min distance of closest ball : {}".format(distance(getRectCenter(minDistanceBall), getRectCenter(self.ball["box"]))))
                # if the min distance is low enough. (assume it is the real ball)
                if distance(getRectCenter(minDistanceBall), getRectCenter(self.ball["box"])) < ballDistanceThres:
                    self.ball["box"] = minDistanceBall
                    self.balls.push(minDistanceBall)
                    
                else:
                    # if the min distance is too far (means the real ball is hidden by something so the min ball distance is another object)
                    # remove the oldest ball from the queue
                    if (self.balls.list != []):
                        self.balls.pop()

            # Draw ball tracing
            for ball in self.balls.list:
                (x,y) = getRectCenter(ball)
                print("[BALL INFO] ({}, {})".format(x,y))
                img = cv2.circle(img,(x,y), 4, (0,255,255), 3)

        # Update frame detected contours
        self.lastFrameContours = self.currentFrameContours
        self.currentFrameContours = []

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


# Use background substraction with predefined and configurable kernels to detect motion.
# arg -motionDetectionCorners speed up the detection by applying the detection only on a cropped sub-image. It can divide the processing time up to 2 times faster if used correctly.
# Returns detected contours
def getMotionContours(img, background=False, motionDetectionCorners = None):
    
    # Kernels used for erosion and dilation masks
    if (img.shape[1] <= 1920):
        kernelErosionSize = (3,3)
        kernelErosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelErosionSize)

        kernelDilationSize = (25,36)
        kernelDilation =  cv2.getStructuringElement(cv2.MORPH_RECT, kernelDilationSize) # Rectangular shape with height > width to better find players
    else:
        kernelErosionSize = (10,10)
        kernelErosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelErosionSize)

        kernelDilationSize = (75,120)
        kernelDilation =  cv2.getStructuringElement(cv2.MORPH_RECT, kernelDilationSize) # Rectangular shape with height > width to better find players


    # Initialize the motion detector corners limit.
    if motionDetectionCorners != None:
        x,y,xx,yy = motionDetectionCorners
        img=img[y:yy,x:xx]

    # Apply background subtractor to get initial foreground mask
    fgMask = bgSubstractor.apply(img); 

    # Remove noisy motion
    erodedMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN,kernelErosion)

    # Dilate motion 
    dilatedMask = cv2.dilate(erodedMask, kernelDilation, iterations=1)


    if background:
        imshowScreen("Initial Foreground Mask",fgMask)
        imshowScreen("Eroded Mask", erodedMask)
        imshowScreen("Dilation Mask",dilatedMask)


    # Draw contours from the mask to get motion boxes
    contours, hierarchy = cv2.findContours(dilatedMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # relocate the contours position by adding the original image offse
    if contours!= [] and motionDetectionCorners != None :       
        # relocate the contours position by adding the original image offset
        contours = [ np.asarray([ [truepoint + [x,y] for truepoint in point] for point in contour]) for contour in contours]
            
    return contours

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
        cv2.drawContours(img, corners, -1, (0, 0, 255), 20)
        im = imutils.resize(img, width=1920)
        cv2.imshow("Court detection", im)

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



def isInRectangle(point,x,y,xx,yy):
    if (point[0] > x and point[0] < xx and point[1] > y and point[1] < yy):
        return True
    return False

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
    return x + int((p-x)/2) , y + int((q-y)/2)

def getManhattanDistance(xy,pq):
        return (abs(xy[0]) - pq[0]) + abs(xy[1] - pq[1])


def getEuclideanDistance(xy,pq):
        distance = math.sqrt((xy[0] - pq[0])**2 + (xy[1] - pq[1])**2)
        return distance

# Euclidian distance
def distance(point1,point2):
    dist = math.sqrt( abs ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2) )
    return dist

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
        label = str(distance) + " m"
        # Left player
        if index == 0:
            cv2.putText(img, label, (200, 300), cv2.FONT_HERSHEY_SIMPLEX, tl / 4, [0, 255, 255], thickness=int(tf/2), lineType=cv2.LINE_AA)
        # Right player
        else:
            cv2.putText(img, label, (w - 600, 300), cv2.FONT_HERSHEY_SIMPLEX, tl / 4, [0, 255, 255], thickness=int(tf/2), lineType=cv2.LINE_AA)

def drawSpeedText(img, speed, player):
    h,w = img.shape[0], img.shape[1]
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)


    speed = str(speed) + " km/h"

    # describe the type of font 
    # to be used. 
    font = cv2.FONT_HERSHEY_SIMPLEX 
  
    # Use putText() method for 
    # inserting text on video 
    if img.shape[1] <= 1920:
        if player == "left":
            cv2.putText(img,  
                        speed,  
                        (200, 100),  
                        font, tl / 3,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4) 

        else:
            cv2.putText(img,  
                        speed,  
                        (img.shape[1] - 400, 100),  
                        font, tl / 3,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4) 
    else :
        if player == "left":
            cv2.putText(img,  
                        speed,  
                        (200, 400),  
                        font, tl / 4,  
                        (0, 255, 255),  
                        thickness=int(tf/2),  
                        lineType=cv2.LINE_AA) 

        else:
            cv2.putText(img,  
                        speed,  
                        (w-600, 400),  
                        font, tl / 4,  
                        (0, 255, 255),  
                        thickness=int(tf/2),  
                        lineType=cv2.LINE_AA)



# Format time into min:sec
# Format time into km/h
def time_convert(sec):
    mins = sec // 60
    sec = round(sec % 60,4)
    hours = mins // 60
    mins = mins % 60
    strTime = "{}:{:.4f}".format(int(mins),sec)
    return strTime


# Resize and Plot video/image on screen
def imshowScreen(title,img):
    resizedImg = imutils.resize(img,width = 1920) 
    cv2.imshow(title, resizedImg)
    

# Read heartbeat rate csv file
# Default wrist band is one value per second
def readHeartRate(totalFrames,fps):
    
    leftHeartRates = []
    rightHeartRates = []
    for hbFile in os.listdir(HEART_BEAT_RATE_FOLDER):
        if hbFile.endswith(".csv"):
            hbFile = os.path.join(HEART_BEAT_RATE_FOLDER, hbFile)
            with open(hbFile, 'r') as csvFile:
                csv_reader = csv.reader(csvFile)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    else:
                        leftHeartRates.append(row[1])
                        rightHeartRates.append(row[2])
                        line_count += 1
        else:
            continue

    # Concatenate the heart beat rate value until it equals the video length in second
    while (len(leftHeartRates) < int(totalFrames//fps) or len(rightHeartRates) < int(totalFrames//fps)):
        leftHeartRates += leftHeartRates
        rightHeartRates += rightHeartRates

    # Cut out the last values to match the video number of seconds
    leftHeartRates = leftHeartRates[:int(totalFrames//fps)]
    rightHeartRates = rightHeartRates[:int(totalFrames//fps)]

    return leftHeartRates, rightHeartRates