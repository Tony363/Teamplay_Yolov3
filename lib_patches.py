# import the necessary packages
import cv2
import os 
import argparse
import numpy as np


import imutils
# ex cmd line :
# python3 imageToPatches.py images/football/frame1300.jpg s -s 640 -o 35


IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160


def readCommand():
    parser = argparse.ArgumentParser()
    # Positional arguments
    parser.add_argument('imagePath', help='Source path of image file')
    parser.add_argument('savePath',help='Destination saving path of patches')

    # Optional arguments
    parser.add_argument('-s', '--size',dest='size', type=int, default=640, help='size of patches')
    parser.add_argument('-o', '--overlap',dest='overlap', type=int, default=35, help='overlap length of pactches. Should be the size of smallest object in the original image.')
    parser.add_argument('-w', '--write',dest='write', type=int, default=1, help='Save image true(1)/false(0)')
    args = parser.parse_args()
    return args


"""
# Robust crop with black filling
def imcrop(img, bbox): 
    x1,y1,x2,y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2
"""

# Read image path and save all the patches.
# Overlap and patch size can be configured.
# NB : Overlap size should be equal or greater than the smallest object to detect

def readImageToPatches(imagePath, savePath, patchSize, overlapLength, write = False):
    image = cv2.imread(imagePath)
    cv2.imshow("original", image)
    cv2.waitKey(0)
    currentWidth = 0
    currentHeight = 0
    
    assert image.shape[0] > patchSize, 'Image height is too small in comparison with the patch size.'
    assert image.shape[1] > patchSize, 'Image width is too small in comparison with the patch size.'
    assert overlapLength > 0, 'Overlap lenght must be greather than 0.'

    IMAGE_HEIGHT = image.shape[0]
    IMAGE_WIDTH = image.shape[1]
    imageCounter = 0



    while(1):

        if( currentWidth + patchSize <= IMAGE_WIDTH and currentHeight + patchSize <= IMAGE_HEIGHT ):
            cropImage = image[currentHeight : currentHeight + patchSize, currentWidth : currentWidth + patchSize]
            patchesRelativePos.append((currentHeight, currentHeight + patchSize, currentWidth, currentWidth + patchSize))
            currentWidth = currentWidth + patchSize - overlapLength
        elif ( currentWidth + patchSize   > IMAGE_WIDTH and currentHeight + patchSize   <= IMAGE_HEIGHT):
            cropImage = image[currentHeight : currentHeight + patchSize, currentWidth : IMAGE_WIDTH]
            patchesRelativePos.append(currentHeight, currentHeight + patchSize, currentWidth, IMAGE_WIDTH)
            currentWidth = 0
            currentHeight = currentHeight + patchSize - overlapLength
        elif (currentWidth + patchSize   <=  IMAGE_WIDTH and currentHeight + patchSize  >  IMAGE_HEIGHT):
            cropImage = image[currentHeight : IMAGE_HEIGHT, currentWidth : currentWidth + patchSize ]
            patchesRelativePos.append(currentHeight, IMAGE_HEIGHT, currentWidth, currentWidth + patchSize)
            currentWidth = currentWidth + patchSize - overlapLength
        elif (currentWidth + patchSize    >  IMAGE_WIDTH and currentHeight + patchSize  >  IMAGE_HEIGHT):
            cropImage = image[currentHeight : IMAGE_HEIGHT, currentWidth : IMAGE_WIDTH ]
            patchesRelativePos.append(currentHeight, IMAGE_HEIGHT, currentWidth, IMAGE_WIDTH)

            #cv2.imshow("original", cropImage)
            #cv2.waitKey(0)

            if(write):
                name = savePath + '/frame' + str(imageCounter) + '.jpg'
                print ('Creating...' + name) 
                cv2.imwrite(name, cropImage) 
                imageCounter +=1

            break

        #cv2.imshow("original", cropImage)
        #cv2.waitKey(0)

        if(write):
            name = savePath + '/frame' + str(imageCounter) + '.jpg'
            print ('Creating...' + name) 
            cv2.imwrite(name, cropImage) 
            imageCounter +=1

    return patches, patchesRelativePos

# Get image and returns image patches with their relative positions in the original image
# Overlap and patch size can be configured.
# NB : Overlap size should be equal or greater than the biggest object to detect
def imageToPatches(image, patchSize, overlapLength):
    
    
    assert image.shape[0] > patchSize, 'Image height is too small in comparison with the patch size.'
    assert image.shape[1] > patchSize, 'Image width is too small in comparison with the patch size.'
    assert overlapLength > 0, 'Overlap lenght must be greather than 0.'


    currentWidth = 0
    currentHeight = 0
    IMAGE_HEIGHT = image.shape[0]
    IMAGE_WIDTH = image.shape[1]
    patches = []
    patchesRelativePos = []
    while(1):

        if( currentWidth + patchSize <= IMAGE_WIDTH and currentHeight + patchSize <= IMAGE_HEIGHT ):
            cropImage = image[currentHeight : currentHeight + patchSize, currentWidth : currentWidth + patchSize]
            patchesRelativePos.append((currentHeight, currentHeight + patchSize, currentWidth, currentWidth + patchSize))
            currentWidth = currentWidth + patchSize - overlapLength
        elif ( currentWidth + patchSize   > IMAGE_WIDTH and currentHeight + patchSize   <= IMAGE_HEIGHT):
            cropImage = image[currentHeight : currentHeight + patchSize, currentWidth : IMAGE_WIDTH]
            patchesRelativePos.append((currentHeight, currentHeight + patchSize, currentWidth, IMAGE_WIDTH))
            currentWidth = 0
            currentHeight = currentHeight + patchSize - overlapLength
        elif (currentWidth + patchSize   <=  IMAGE_WIDTH and currentHeight + patchSize  >  IMAGE_HEIGHT):
            cropImage = image[currentHeight : IMAGE_HEIGHT, currentWidth : currentWidth + patchSize ]
            patchesRelativePos.append((currentHeight, IMAGE_HEIGHT, currentWidth, currentWidth + patchSize))
            currentWidth = currentWidth + patchSize - overlapLength
        elif (currentWidth + patchSize    >  IMAGE_WIDTH and currentHeight + patchSize  >  IMAGE_HEIGHT): # end of crop
            cropImage = image[currentHeight : IMAGE_HEIGHT, currentWidth : IMAGE_WIDTH ]
            patchesRelativePos.append((currentHeight, IMAGE_HEIGHT, currentWidth, IMAGE_WIDTH))
            patches.append(cropImage)
            #cv2.imshow("original", cropImage)
            #cv2.waitKey(0)
            break
        
        patches.append(cropImage)
        
        #cv2.imshow("original", cropImage)
        #cv2.waitKey(0)
    
    print("\nImage has been successfully patched in {} sub-images.".format(len(patches)))          
    return patches, patchesRelativePos

# Merge patches into image
# Use detection object to check which image must be chosen in overlapping zone
# NB : if detection is None, the overlap zone is going to be decided by the patches order (from left to right then from top to the bottom of the image)
def patchesToImage(patches, overlapLength, imageWidth, imageHeight, detection = None):
    currentWidth = 0
    currentHeight = 0

    nbChannels = (patches[0].shape)[2]
    patchSize = patches[0].shape[1]
    image = np.zeros((imageHeight, imageWidth, nbChannels),np.uint8)
    

    if detection is None:
        for patch in patches:  

            if( currentWidth + patchSize <= imageWidth and currentHeight + patchSize <= imageHeight ):
                image[currentHeight : currentHeight + patchSize, currentWidth : currentWidth + patchSize] = patch
                currentWidth = currentWidth + patchSize - overlapLength
            elif ( currentWidth + patchSize   > imageWidth and currentHeight + patchSize   <= imageHeight):
                image[currentHeight : currentHeight + patchSize, currentWidth : imageWidth] = patch
                currentWidth = 0
                currentHeight = currentHeight + patchSize - overlapLength
            elif (currentWidth + patchSize   <=  imageWidth and currentHeight + patchSize  >  imageHeight):
                image[currentHeight : imageHeight, currentWidth : currentWidth + patchSize ] = patch
                currentWidth = currentWidth + patchSize - overlapLength
            elif (currentWidth + patchSize    >  imageWidth and currentHeight + patchSize  >  imageHeight): # end of crop
                image[currentHeight : imageHeight, currentWidth : imageWidth ] = patch
    
    else:
        pass

    
    #cv2.imshow("Merged patches", image)
    #cv2.imwrite("data/tennis/full_image/frame600_merged.jpg", image)
    #cv2.waitKey(0)
    
    return image

# Same function as patchesToImage but takes as argument the relative position of each patch
def patchesRelToImage(patches, patchesRelativePos,imageHeight, imageWidth, detection = None):
    nbChannels = (patches[0].shape)[2]
    image = np.zeros((imageHeight, imageWidth, nbChannels),np.uint8)
    for index, patch in enumerate(patches):
        y0,y1,x0,x1 = patchesRelativePos[index]
        image[y0:y1,x0:x1] = patch
    
    print("\nImage has been successfully merged using {} sub-images.".format(len(patches)))          
    #cv2.imshow("Merged patches", image)
    #cv2.imwrite("data/tennis/full_image/frame600_merged_2.jpg", image)
    #cv2.waitKey(0)
    
    return image

if __name__ == '__main__':
    
    args = readCommand()
    """
    patches, patchesRelativePos = readImageToPatches(args.imagePath, args.savePath, args.size, args.overlap,args.write)
    patchesRelToImage(patches,patchesRelativePos)
    """
    # Test frame reconstitution
    
    image = cv2.imread('data/tennis/full_image/frame600.jpg')
    h,w,c = image.shape
    print("Initial image size : {}x{}".format(w,h) )
    patchSize = 1000
    overlapLength = 200
    patches, patchesRelativePos = imageToPatches(image, patchSize, overlapLength)

    patchesRelToImage(patches,patchesRelativePos,h,w)
    