from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = np.array(Image.open('cat.jpg'))
plt.imshow(img)

def sobelKernel(img):
    x = np.array([[-1,0,1],
                  [-2,0,2],
                  [-1,0,1]])
    y = np.array([[1,2,1],
                  [0,0,0],
                  [-1,-2,-1]])
    if len(img.shape)==3:
        img = np.mean(img,axis=2)
    gradX = conv2d(img,x)
    gradY = conv2d(img,y)
    gMagnitude = np.sqrt(gradX**2+gradY**2)
    
    return gMagnitude

def prewittKernel(img):
    x = np.array([[-1,0,1],
                  [-1,0,1],
                  [-1,0,1]])
    y = np.array([[-1,-1,-1],
                  [0,0,0],
                  [1,1,1]])
    if len(img.shape)==3:
        img = np.mean(img,axis=2)
    gradX = conv2d(img,x)
    gradY = conv2d(img,y)
    gMagnitude = np.sqrt(gradX**2+gradY**2)
    
    return gMagnitude

def cannyKernel(img,lowthresh,hightresh):
    blur = cv2.GaussianBlur(img,(5,5),0)
    edge = cv2.Canny(blur,lowthresh,hightresh)
    return edge

def conv2d(img,kernel):
    kernel = np.flipud(np.fliplr(kernel))
    height,width = kernel.shape
    padding = np.pad(img,pad_width=((1,1),(1,1)),mode="constant",constant_values=0)
    outImg = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            outImg[i,j] = np.sum(padding[i:i+height,j:j+width]*kernel)
        
    return outImg

sobel = sobelKernel(img)
plt.title("SOBEL")
plt.imshow(sobel, cmap='gray')

prew = prewittKernel(img)
plt.title("PREWITT")
plt.imshow(prew,cmap="gray")

low_threshold = 30
high_threshold = 80
canny = cannyKernel(img,low_threshold,high_threshold)
plt.title("CANNY")
plt.imshow(canny.astype(np.uint8),cmap="gray")