import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

from lib_patches import *
import imutils


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



class TennisState:

    " Tennis Game state including ball, players position, ball tracer, court key points ..."
    def __init__(self, ballTracer):
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

        self.balls = Queue() # Save in memory last -ballTracer balls positions
        
        self.trainingBalls = [] #(xA, yA, xB, yB)
        self.lastFrameBalls = []
        self.currentFrameBalls = []

        self.lastFrameBallsPatch = 30 * [[]] # 30 patch # element i : lastFrameBalls of patch i
        self.currentFrameBallsPatch = 30 * [[]]

    def clearDetections(self, xyxy, conf, className):
        if className == "sports ball":

            # Check ball size
            # False positive rejection by area size
            if getArea(xyxy) > 10000:
                return False


            print("[INFO] New ball !")
            self.currentFrameBalls.append(xyxy)
            # Initialize training balls (non-moving ball in first frames)
            if self.lastFrameBalls == []:
                #self.trainingBalls.append(xyxy)
                return True
            else:
            # Check if detected ball is in the same position as ONE of the last frame balls
                for i,ball in enumerate(self.lastFrameBalls):
                    print("ball : {}, lastframe ball {} {}  . IoU = {}".format(tensorPointToList(xyxy), i,tensorPointToList(ball),bb_intersection_over_union(xyxy,ball)))
                    if bb_intersection_over_union(xyxy,ball) > 0.1:
                        return False
                
                print("[INFO] Real ball detected !")
                self.ball['box'] = xyxy
                self.ball['conf'] = conf
                return True

                
        elif className == "person":
            return True

        elif className == "tennis racket":
            return True

        return True

    def clearPatchDetections(self, indexPatch, xyxy, conf, className):
        if className == "sports ball":

            # Check ball size
            # False positive rejection by area size
            if getArea(xyxy) > 10000:
                return False


            print("[INFO] New ball !")
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


        elif className == "tennis racket":
            return True



        return True

def detect(save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out)==False:
        #shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    #os.makedirs(out)  # make nself.lastFrameBalls = []ew output folder


    # Initialize Tennis State
    gameState = TennisState(ballTracer = 5)

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

    # Eval mode
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
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
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


    if (opt.patch == 0):

        # Run inference
        t0 = time.time()
        # im0s -> real img
        # img -> transformed img (padded .. right size ...)
        for path, img, im0s, vid_cap in dataset:

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
            print("00000000   " , opt.classes)
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

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:

                        if save_txt:  # Write to file
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            if (gameState.clearDetections(xyxy, conf, names[int(cls)])):
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                    # Update state after all detections processing
                    print("[INFO] Update Game state ...")
                    gameState.lastFrameBalls = gameState.currentFrameBalls
                    gameState.currentFrameBalls = []
                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    im0_resized = imutils.resize(im0, width=1920)
                    cv2.imshow(p, im0_resized)
                    #cv2.waitKey(0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + out + ' ' + save_path)
        print('Done. (%.3fs)' % (time.time() - t0))

    # Patch inference
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



                    # Print patch time (inference + NMS)
                    patches[index] = im0 #Update patches array
                    print('Patch inference %d/%d %sDone. (%.3fs)' % (index,len(patches),s, t2 - t1))
                    # Update state after all detections processing
                    print("[INFO] Update Game state ...")
                    gameState.lastFrameBallsPatch[index] = gameState.currentFrameBallsPatch[index]
                    gameState.currentFrameBallsPatch[index] = []
                    
            
            
            fullImg = patchesRelToImage(patches, patchesRelativePos,im0s.shape[0], im0s.shape[1])                   
            print('Full Image Done. (%.3fs)' % (time.time() - timeStartFullImg))
            
            
            

            # Stream results
            if view_img:
                fullImg_resized = imutils.resize(fullImg, width=1920)
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
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(fullImg)

        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + out + ' ' + save_path)
        
        print('Done. (%.3fs)' % (time.time() - t0))



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
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--patch', type=int, default=0, help='patch size used for patches-based inference. ') # patch 1000 and overlap 200
    parser.add_argument('--overlap', type=int, help='overlap length used for patches-based inference. Should be greater or equal to the biggest relevant object')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
