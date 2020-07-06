
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2,imutils
import time,random,argparse,progressbar, os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from shapely.geometry import Point, Polygon

from time import sleep
from collections import deque
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import GenericMask



def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def load_yaml(parser,args):
    if args.load_pts:
        src_pts,dst_pts = load_pts(args.load_pts)
        return src_pts,dst_pts
    if args.homographic_points:
        src_pts,dst_pts = load_coefficients(args.homographic_points)
        return src_pts,dst_pts

def load_pts(args):
    src,dst = tuple(args)
    src_file = cv2.FileStorage(f"homograph_pts/{src}",cv2.FILE_STORAGE_READ)
    dst_file = cv2.FileStorage(f"homograph_pts/{dst}",cv2.FILE_STORAGE_READ)
    src_pts = src_file.getNode("src_pts").mat()
    dst_pts = dst_file.getNode("dst_pts").mat()
    src_file.release()
    dst_file.release()
    print("size of both yml files: ",src_pts.size,dst_pts.size)
    return src_pts,dst_pts

def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    src_pts = cv_file.getNode("src_pts").mat()
    dst_pts = cv_file.getNode("dst_pts").mat()
    cv_file.release()
    return src_pts,dst_pts

def arguments():
    parser = argparse.ArgumentParser(description='Coordinate Homography')
    parser.add_argument('--image', type=str, required=True, help='image directory path')
    parser.add_argument('--court',type=str,required=True, help='directory of court image')
    parser.add_argument('--video',type=str,required=True,help='directory of video')
    parser.add_argument('--cfg_model',nargs='+',type=str,required=True,help="""
    arg1 = path to .yaml file e.g. ../../detectron2_repo/configs/COCO-InstanceSegmentation/
    arg2 = .yaml model e.g. mask_rcnn_R_50_FPN_3x.yaml
    """)
    parser.add_argument('--homographic_points',type=str,required=False,default='/homographic_pts/homographic.yml',help='read src_pts and dst_pts yaml file')
    parser.add_argument('--load_pts',nargs="+",type=str,required=False,help="load pts yml")
    args = parser.parse_args()
    return parser,args

def drawPlayers(im, pred_boxes, showResult=False):
    color = [255, 0, 0]   
    thickness = 1
    radius = 1
    i  = 0
    for box in pred_boxes:
        # Include only class Person
        if pred_classes[i] == 0:  
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            xc = x1 + int((x2 - x1)/2)
            player_pos1 = (xc - 1, y2)
            player_pos2 = (xc + 1, y2 + 1)
            court = Polygon(src_pts)
            # Draw only players that are within the basketball court
            if Point(player_pos1).within(court):
                if showResult:
                    print("[% 3d, % 3d]" %(xc, y2))
            cv2.rectangle(im, player_pos1, player_pos2, color, thickness)
            i = i + 1            
    if showResult:
        cv2.imshow('players',im)
        cv2.waitKey(0)

def homographyTransform(im, showResult=False):
    # Calculate Homography
    h, status = cv2.findHomography(src_pts, dst_pts)
    img_out = cv2.warpPerspective(im, h, (img_dst.shape[1], img_dst.shape[0]))
    if showResult:
        cv2.imshow('cal_homo',img_out)
        cv2.waitKey(0)
    return img_out  

def getPlayersMask(im):
    lower_range = np.array([255,0,0])                         # Set the Lower range value of blue in BGR
    upper_range = np.array([255,155,155])                     # Set the Upper range value of blue in BGR
    mask = cv2.inRange(im, lower_range, upper_range)     # Create a mask with range
    result = cv2.bitwise_and(im, img_out, mask = mask)   # Performing bitwise and operation with mask in img variable
    cv2.imshow('mask',result)                              
    return cv2.inRange(result, lower_range, upper_range)  

def drawPlayersOnCourt(im, coord, color, radius=10):
    for pos in coord:
        center_coordinates = (pos[0], pos[1])
        cv2.circle(im, center_coordinates, radius, color, thickness=-1) 
    return im

# Draft method to draw lines between history player positions to show trail
def drawCoordinateLines(result, pts, currentFrame, player):
    for i in np.arange(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(30 / float(i + 1)) * 2.5)
        print("player=%s" %player)
        x1 = pts[i - 1][0][0]
        x2 = pts[i - 1][0][1]
        print("x1=%d, x2=%d" %(x1, x2))
        y1 = pts[i][0][0]
        y2 = pts[i][0][1]
        print("y1=%d, y2=%d" %(y1, y2))
        print(" ---------------------- ")
        cv2.line(result, (x1, x2), (y1, y2), red_color, thickness)
    return result

def load_cfg(parser,args):
    detectron_model,model = args.cfg_model
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu' # for cpu usage
    cfg.merge_from_file('{detectron_dir}/{model}'.format(detectron_dir=detectron_model,model=model))

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

    predictor = DefaultPredictor(cfg)
    players_output = predictor(img)
    return players_output,cfg,predictor

def Visualizer_pred(img,cfg):
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    v = v.draw_instance_predictions(players_output["instances"].to("cpu"))
    cv2.imshow('Visualizer',v.get_image()[:, :, ::-1])
    cv2.waitKey(0)

def polylines(img,src_pts,dst_pts):
    im_poly = img.copy()
    # cv2.fillPoly(img_src, [src_pts], 255)
    cv2.polylines(im_poly, [src_pts], isClosed=True, color=[255,0,0], thickness=2)
    cv2.imshow('im_poly',im_poly)
    cv2.waitKey(0)

    cv2.polylines(img_dst, [dst_pts], isClosed=True, color=[255,0,0], thickness=2)
    cv2.imshow('img_dst',img_dst)
    cv2.waitKey(0)

def create_homograph(parser,args):
    vs = cv2.VideoCapture("{video}".format(video=args.video))
    totalFrames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    grabbed = True
    currentFrame = 0
    start = time.time()
    writer = None
    bar = progressbar.ProgressBar(maxval=totalFrames, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    court_img = cv2.imread('{court}'.format(court=args.court))
    blue_color = (255,0,0)
    red_color = (0,0,255)
    # loop over frames from the video file stream (207)
    while grabbed:     
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if writer is None:
            fps = vs.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter("../output/mini-map-output.mp4", fourcc, fps, (court_img.shape[1], court_img.shape[0]), True)
        if grabbed:
            # print(currentFrame)
            # Get player positions
            outputs = predictor(frame)  
            instances = outputs["instances"].to("cpu")
            boxes = instances.get("pred_boxes")
            court = court_img.copy()
            # Draw players on video frame
            drawPlayers(frame, boxes, False)
            img_out = homographyTransform(frame, False)
            # cv2_imshow(img_out)
            mask = getPlayersMask(img_out)
            # cv2_imshow(mask)
            # Get the contours from the players "dots" so we can reduce the coordinates
            # to the number of players on the court.
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)  
            if cnts is not None:      
                for cnt in cnts:
                    result = drawPlayersOnCourt(court, cnt[0], blue_color)                       
            writer.write(result)
            currentFrame += 1
            bar.update(currentFrame)
        else:
            grabbed = False
    # cv2_imshow(result)  
    writer.release()
    vs.release()
    bar.finish()
    end = time.time()
    elap = (end - start)
    print("[INFO] process took {:.4f} seconds".format(elap))
    print("Video created")



if __name__ == "__main__":
    # Use the boxes info from the tensor prediction result
    #
    # x1,y1 ------
    # |          |
    # |          |
    # |          |
    # --------x2,y2
    #
    # Four corners of the 3D court 
    # Start top-left corner and go anti-clock wise
    # Please manually extract homographic points from video for now
    src_pts = np.array([
            [0,711],
            [954,821],
            [1919,762],
            [1919,631],
            [1493,525],
            [493,505],
            [4,608]
        ]) 

    # Four corners of the court + mid-court circle point in destination image 
    # Start top-left corner and go anti-clock wise + mid-court circle point
    dst_pts = np.array([
        [144,  1060],  # LEFT BOTTOM
        [969,  1065],  # MIDDLE BOTTOM
        [1769, 1063],  # RIGHT BOTTOM
        [1885, 875],   # TOP BOTTOM RIGHT  (4 o'clock)
        [1882,  49],   # TOP RIGHT
        [50,    43],   # TOP LEFT
        [50,    871]   # TOP - BOTTOM LEFT (7 o'clock)
        ])    

    parser,args = arguments()
    src_pts,dst_pts = load_yaml(parser,args)

    img = cv2.imread("{image}".format(image=args.image))
    img_dst = cv2.imread('{court}'.format(court=args.court))
    players_output,cfg,predictor = load_cfg(parser,args)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    instances = players_output["instances"]
    print(instances)
    pred_boxes = instances.get("pred_boxes")
    pred_classes = instances.get("pred_classes")
    print(pred_boxes)
    print(pred_classes)

    Visualizer_pred(img,cfg)
    polylines(img,src_pts,dst_pts) 
    drawPlayers(img, pred_boxes, True)
    # Try out
    img_out = homographyTransform(img, True)
    # Try out  
    mask = getPlayersMask(img_out)    
    cv2.imshow('mask',mask)
    create_homograph(parser,args)
