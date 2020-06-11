import cv2
import time
import imutils
import numpy as np
from pylab import array,plot,show,axis,arange,figure,uint8


cap = cv2.VideoCapture('input_vid/serve_right1.mp4')
# Define the codec and create VideoWriter object 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

def make_1080():
    cap.set(3,1920)
    cap.set(4,1080)

def make_720p():
    cap.set(3,1280)
    cap.set(4,720)

def make_480p():
    cap.set(3,640)
    cap.set(4,480)

def rescale_frame(frame,percent=75):
    scale_percent = 75 
    width = int(frame.shape[1] *scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)

def shift_frame(frame):
    num_rows, num_cols = frame.shape[:2]
    translation_matrix = np.float32([[1,0,70],[0,1,110]])
    print(translation_matrix)
    img_translation = cv2.warpAffine(frame,translation_matrix,(num_cols,num_rows))
    return img_translation

resolution = [make_480p,make_720p,make_1080]

def zoomin(video,speed,y,x,w,h):
    ret,frame = video.read() 
    rows,cols,rgb = frame.shape
    stop = frame[w:,h:].shape 
    shift = 0
    while True:
        # print(rows, cols)
        ret,frame = video.read() 
        if ret:
            if rows > stop[0] and cols > stop[1]:
                shift += speed
            if y < 1056 and x < 2517:
                frame = frame[y:-shift,x:-shift*2]
            elif y > 1056 and x > 2517:
                frame = frame[shift:y,shift*2:x] 
            elif x > 2517 and y < 1056:
                frame = frame[y:-shift,shift*2:x]
            elif x < 2517 and y > 1056:
                frame = frame[shift:y,x:-shift*2]
            elif y > 1056:
                frame = frame[shift:,:]
            elif y < 1056:
                frame = frame[:-shift,:]
            elif x > 2517:
                frame = frame[:,shift:]
            elif x < 2517:
                frame = frame[:,:-shift]
            try:
                resized = cv2.resize(frame,(1280, 720))
                print(resized.shape)
            except Exception as e:
                print(e)
                pass
            rows, cols, rgb = frame.shape
            out.write(resized)
            cv2.imshow("frame", resized)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else:
            break
        
# zoomin(cap,21,2156,5034,2100,5000)
# zoomin(cap,21,2156,1,2100,5000)
# zoomin(cap,11,1,5034,2100,5000)
# zoomin(cap,21,1,1,2100,5000)
# zoomin(cap,21,1,2517,2100,5000)
# zoomin(cap,21,1056,1,2100,5000)
# zoomin(cap,11,2156,2517,2100, 5000)
# zoomin(cap,21,1056,5034,2100,5000)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=5.0, threshold=30):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def increase_brightness(img, value=10):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def jump(im0,y,x,h,w):
    im0 = im0[h:y,w:x]
    return im0

maxIntensity = 255.0
x = arange(maxIntensity)
phi = 1 
theta = 1 

# make_1080()
count = 0
while True:
    ret,frame = cap.read() 
    if ret:
        frame = frame[800:1300,2500:2800]
        h,w = frame.shape[:2]
        cv2.putText(frame,'50.55g',(w//12,h//3),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_4)
        if count == 0:
            out = cv2.VideoWriter('output.mp4', fourcc, 60,(w, h)) 
        try:
            resized = imutils.resize(frame,width=1080)
        except Exception as e:
            print(e)
            pass
        # frame = jump(frame,400,700,200,400)
        # frame = jump(frame,1000,1300,600,700)
        # frame = cv2.blur(frame, (2,2))
        # image = unsharp_mask(frame)
        # newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
        # newImage0 = array(newImage0,dtype=uint8)
        # frame = rescale_frame(sharpened_image,percent=75)
        newImage0 = increase_brightness(frame)
        cv2.imshow("frame", resized)
        out.write(newImage0)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        count += 1
    else:
        break
    



# release the cap object
out.release()
cap.release() 
# close all windows 
cv2.destroyAllWindows() 