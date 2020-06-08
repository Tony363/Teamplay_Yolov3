
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

if __name__ == '__main__':
    pass