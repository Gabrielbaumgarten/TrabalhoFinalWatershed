from math import floor
import numpy as np
import cv2
import imutils
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from tkinter import *
from tkinter import colorchooser
import colorsys

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

INPUT_IMAGE =  './quadros.jpg'

def selectBackground(event,x,y,flags, param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(sure_bg,(x,y),15,(0,0,0),-1)
            cv2.circle(sure_bg_mask,(x,y),15,(0,0,0),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(sure_bg,(x,y),15,(0,0,0),-1)
        cv2.circle(sure_bg_mask,(x,y),15,(0,0,0),-1)

def selectForeground(event,x,y,flags, param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(sure_fg,(x,y),15,(255,255,255),-1)
            cv2.circle(sure_fg_mask,(x,y),15,(255,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(sure_fg,(x,y),15,(255,255,255),-1)
        cv2.circle(sure_fg_mask,(x,y),15,(255,255,255),-1)


img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
# img = cv2.resize(img, (floor(img.shape[0]/2), floor(img.shape[1]/2)))
# img = img.astype(np.float)/255
cv2.imshow('Original', img)

sure_bg = img.copy()
sure_bg_mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
cv2.namedWindow('background')
cv2.setMouseCallback('background',selectBackground)

while(1):
    cv2.imshow('background',sure_bg)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 13 or k == 27:
        break

sure_bg_mask = np.where(sure_bg_mask != 0, 255, 0)
sure_bg_mask = sure_bg_mask.astype(np.uint8)
cv2.imshow('Sure BG', sure_bg_mask)

sure_fg = img.copy()
sure_fg_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
cv2.namedWindow('foreground')
cv2.setMouseCallback('foreground',selectForeground)

while(1):
    cv2.imshow('foreground',sure_fg)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 13 or k == 27:
        break

sure_fg_mask = np.where(sure_fg_mask != 255, 0, 255)
sure_fg_mask = sure_fg_mask.astype(np.uint8)
cv2.imshow('Sure FG', sure_fg_mask)


unknown = cv2.subtract(sure_bg_mask, sure_fg_mask)

cv2.imshow('Unknown', unknown.astype(np.uint8))


_, markers = cv2.connectedComponents(sure_fg_mask)
markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(img, markers)

# substituir essa parte pela parte do henrique
cor = colorchooser.askcolor(title = "Tkinter Color Chooser")
cor = np.array(np.array(np.array(cor[0])))
cor = cor.astype(np.uint8)
cor = cv2.cvtColor(cor, cv2.COLOR_BGR2HLS)


img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        if markers[row][col]==1 :
            img_hls[row][col] = [cor[0],(img_hls[row][col][1]),cor[2]]
#             # img_hsl[row][col] = [322 ,98,img_hsl[row][col][2]]

img_hls = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR)
cv2.imshow('changeColor', img_hls)


img[markers == -1] = [255, 0, 0]
cv2.imshow('Contornos', img)


cv2.waitKey ()
cv2.destroyAllWindows ()

