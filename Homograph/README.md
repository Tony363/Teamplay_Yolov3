# Court Homograph

## Introduction

> 
+ First extract homographic points to Match between 2 images/frames.( In the Future we can implement a script to automate the findings of optimal homographic points to match.)
+ load model with desired params
+ Test model and homographic points on images first
+ Write video
    + detect on each frame
    + have option to draw players detection box on court
    + Transform frame on homographic points
    + get player mask through transformed frame (bitwise_and() first 2 params have to be the same shape)
    + have opencv find Contours and imutils to grab contours
    + for each point, draw blue dot 

## Code Samples

> 
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
    parser.add_argument('--detect_box',action="store_true",required=False,help="write detection box")
    args = parser.parse_args()

## Installation

> git clone yolov3 from Davids repo