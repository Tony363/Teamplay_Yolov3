from action_focus.focus import *


class Queue:
  "A container with a first-in-first-out (FIFO) queuing policy."
  def __init__(self):
    self.list = []

  def push(self,item):
    "Enqueue the 'item' into the queue"

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



class BasketBallState:

    " BasketBall Game state including ball, players position, ball tracer, court key points ..."
    def __init__(self):
        # Players' Queue
        self.playersQueue = Queue()

        # Virtual camera center and size
        self.cameraCenter = (0,0)
        self.cameraBorders = (0,0,0,0) # xmin, xmax, ymin, ymax

    " Update the virtual camera center. Use -test argument to enable manual panning."
    def updateCameraCenter(self, test = False):
      if test == False:
        self.cameraCenter = getVirtualCameraCenter(self.playersQueue.list, None, self.cameraCenter, motionWeight = 0.9)
      else:
        self.cameraCenter = getVirtualCameraCenterTest()
    

def getArea(xyxy):
    xA, yA, xB, yB = tensorPointToList(xyxy)
    return (xB - xA) * (yB - yA)

def tensorPointToList(xyxy):
    return (int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3]))


   