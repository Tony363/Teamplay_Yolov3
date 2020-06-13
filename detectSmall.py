import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import imutils
import time

from multiprocessing.dummy import Pool as ThreadPool

# Tennis libraries
from tennis.lib_patches import *
from tennis.tennis import *

# Zoom library
from zoom.zoom import zoomin,increase_brightness,getZoomCentroid,getCroppedImage,zoom_ball1,zoom_impact,zoom_player

# python3 detectSmall.py --source video_path.mp4 --cfg cfg/yolov3.cfg --weights weights/yolov3.pt --classes 0 32 38 --iou-thres 0.1 --view-img

def detect(save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    zoom,zoom_object,out, source, weights, half, view_img, save_txt = opt.zoom,opt.zoom_object,opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize the device
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out)==False:
        os.makedirs(out)  # make new output folder


    # Initialize Tennis State
    gameState = TennisState(maxBalls = 8)
    readFrame = 0

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Export model to device Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        # print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size)
    
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, patch=opt.patch, overlap=opt.overlap)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # SlowMotion count
    count = 0
    # Smooth player speed
    speed = 0
    # player shifted frame
    player = None
    
    # last frame zoomed img centroid
    lastCentroid = (0,0)

    # motion weight 
    motionWeight = 0.9

    # Prediction on raw image
    if (opt.patch == 0):
        
        # Run inference
        t0 = time.time()
        # im0s -> real img
        # img -> transformed img (padded .. right size ...)
        for path, img, im0s, vid_cap in dataset:
            # Initialize Video config
            fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
            totalFrames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Detect tennis court on the first frame and Initialize the heart beart
            if gameState.courtIsDetected == False:
                gameState.court = detectTennisCourt(im0s)
                gameState.courtIsDetected = True
                gameState.scaleDistance = getEuclideanDistance(gameState.court[0][0], gameState.court[2][0])
                print("Scale distance :  {}".format(gameState.scaleDistance) )
                # gameState.leftHeartRates, gameState.rightHeartRates = readHeartRate(totalFrames, fps)
                print("Video total number of frames : {}".format(int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))))
                gameState.motionDetectionCorners = (200, int(7*im0s.shape[0] / 24), im0s.shape[1] -200, 6 * int(im0s.shape[0]/7))
                print("Motion Detection left-top corners : {}, {}".format(200,int(7*im0s.shape[0] / 24)))

            # Allocate the image tensor to the chosen device
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # Resize  the tensor
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            # print(pred)
            t2 = torch_utils.time_synchronized()

            # to float
            if half:
                pred = pred.float()
            # print("00000000   " , opt.classes)

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                    multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, s, im0 = path, '', im0s
                    zoom_im0 = im0.copy()

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):   
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    

                    im0Copy = im0.copy() # Image copy used by motion detection

                    leftPersons = [] # potential left player
                    rightPersons = [] # potential right player
                    
                    # Iterate over all detections
                    # *xyxy - bounding 
                    for *xyxy, conf, cls in det:
                        if names[int(cls)] == "person" and getArea(xyxy) > 10000:
                            # If the detected person is in the left side of the image
                            if (getRectCenter(xyxy))[0] < im0.shape[1] /  2:
                                leftPersons.append((xyxy,conf,cls))
                            else:
                                rightPersons.append((xyxy,conf,cls))

                        # Plot any other detected objects
                        elif names[int(cls)] != "person" and opt.show:
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                    # Update the ball position and trace
                    if( opt.tracing and readFrame > START_BALL_TRACK_FRAME):
                        startBall = torch_utils.time_synchronized()
                        contours = getMotionContours(im0Copy,True,gameState.motionDetectionCorners)
                        gameState.updateBallPositionFromMotion(im0, contours,readFrame, showContours = opt.contours)
                        endBall = torch_utils.time_synchronized()
                        print("[INFO] Updated ball position. ({:.3f}s)".format(endBall - startBall))

                    # Update the player last hit speed
                    gameState.updateHitSpeed(im0,readFrame)

                    # Update the player position and display the player bounding box on image
                    gameState.identifyPlayersAndPlot(im0,leftPersons, rightPersons, colors, False)
                    
                    # Update player heartbeat rate every second
                    # gameState.updateHeartRate(im0, readFrame,fps)

                    # Update time watch
                    gameState.updateTimeWatch(im0, fps)

                    if zoom:
                        zoom,im0,count,lastCentroid = zoomin(zoom,im0,gameState.players[0]['box'],count,lastCentroid, motionWeight) #crop image
                    elif zoom and zoom_object == 'ball':
                        zoom, zoom_im0,count = zoom_ball1(zoom,zoom_im0,count)
                    elif zoom and zoom_object == 'impact':
                        zoom,zoom_im0,count = zoom_impact(zoom,zoom_im0,count)
                    readFrame +=1
                    
                    """
                    # Write results
                    for *xyxy, conf, cls in det:

                        if save_txt:  # Write to file
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            if (gameState.clearDetections(xyxy, conf, names[int(cls)])):
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                    """
                    
                    # Update state after all detections processing
                    # print("[INFO] Update Game State ...")
                    # for index, distance in enumerate(gameState.distances):
                        # print("[INFO] Player {} : {} m".format(index,distance))
                    gameState.lastFrameBalls = gameState.currentFrameBalls
                    gameState.currentFrameBalls = []
                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    if zoom and opt.tracing:
                        im0_resized = imutils.resize(im0, width=1080)
                    elif zoom:
                        im0_resized = imutils.resize(zoom_im0, width=1080)
                    else:
                        im0_resized = imutils.resize(im0, width=1080)
                    cv2.imshow(p, im0_resized)
                    #cv2.waitKey(0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    count = 0 # writing count
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            SlowM_80 = 12
                            SlowM_20 = 48
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (zoom_im0.shape[1], zoom_im0.shape[0]))
                            if zoom and opt.tracing:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                write_traced_zoom = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (im0.shape[1], im0.shape[0]))
                        if zoom and opt.tracing:
                            write_traced_zoom.write(im0)
                        elif zoom:
                            vid_writer.write(zoom_im0)
                        else:
                            vid_writer.write(im0)

                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

        if save_txt or save_img:
            # print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + out + ' ' + save_path)
        # print('Done. (%.3fs)' % (time.time() - t0))

    # Prediction on image patches. Use --patch and --overlap arguments to use patch inference.
    else:
       
        # Run inference
        t0 = time.time()
        # im0s -> real img
        # img -> transformed img (padded .. right size ...)
        for path, im0s, vid_cap in dataset:
            
            # Initialize patches array
            patches, patchesRelativePos = imageToPatches(im0s, opt.patch, opt.overlap)
            # Initialize detection boxes array
            patchesDetections = []

            timeStartFullImg = time.time()
            for index, patch in enumerate(patches):
                # Patch processing (adapt to the the network size). See datasets.py.
                # Padded resize
                img = letterbox(patch, new_shape=img_size)[0]

                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)


                # Initialize inference
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = torch_utils.time_synchronized()
                pred = model(img, augment=opt.augment)[0]
                t2 = torch_utils.time_synchronized()

                # to float
                if half:
                    pred = pred.float()
        
                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0 = path[i], '%g: ' % i, im0s[i]
                    else:
                        p, s, im0 = path, '', patch

                    save_path = str(Path(out) / Path(p).name)
                    s += '%gx%g ' % img.shape[2:]  # print string
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        # Write results
                        for *xyxy, conf, cls in det:
                            if save_txt:  # Write to file
                                with open(save_path + '.txt', 'a') as file:
                                    file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                            if save_img or view_img:  # Add bbox to image
                                if gameState.clearPatchDetections(index,xyxy, conf, names[int(cls)]):
                                    label = '%s %.2f' % (names[int(cls)], conf)
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])


                    # Plot ball tracer
                    plot_ball_history(gameState.balls,im0)
                    # Print patch time (inference + NMS)
                    patches[index] = im0 #Update patches array
                    # print('Patch inference %d/%d %sDone. (%.3fs)' % (index,len(patches),s, t2 - t1))
                    # Update state after all detections processing
                    # print("[INFO] Update Game state ...")
                    gameState.lastFrameBallsPatch[index] = gameState.currentFrameBallsPatch[index]
                    gameState.currentFrameBallsPatch[index] = []
                    
            
            
            fullImg = patchesRelToImage(patches, patchesRelativePos,im0s.shape[0], im0s.shape[1])                   
            # print('Full Image Done. (%.3fs)' % (time.time() - timeStartFullImg))
            
            
            

            # Stream results
            if view_img:
                fullImg_resized = imutils.resize(fullImg, width=1080,height=1920)
                cv2.imshow(p, fullImg_resized)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, fullImg)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        SlowM_80 = 12
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(fullImg)

        if save_txt or save_img:
            # print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + out + ' ' + save_path)
        
        # print('Done. (%.3fs)' % (time.time() - t0))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS.')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--patch', type=int, default=0, help='patch size used for patches-based inference. ') # patch 1000 and overlap 200
    parser.add_argument('--overlap', type=int, help='overlap length used for patches-based inference. Should be greater or equal to the biggest relevant object')
    parser.add_argument('--tracing', action='store_true', help='Enable ball tracing')
    parser.add_argument('--show', action='store_true', help='Enable bounding boxes display from object detection')
    parser.add_argument('--contours', action='store_true', help='Enable contours display from motion detection')
    parser.add_argument('--zoom',action="store_true", help='zoom or not to zoom [True/False]')
    parser.add_argument('--zoom_object',type=str,default='object',help='enter object to zoom into[object/player/ball]')
    opt = parser.parse_args()
    # print(opt)

    with torch.no_grad():
        detect()
