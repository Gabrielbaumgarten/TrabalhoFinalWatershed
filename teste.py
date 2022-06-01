import numpy as np
import cv2
import imutils
from torch import floor

INPUT_IMAGE =  './img.webp'

def main ():

    # img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    # # img = img.astype(np.float)/255
    # cv2.imshow('Original', img)
    img = cv2.imread("colourWall.jpg")
    img = cv2.resize(img, (round(img.shape[0]/2), round(img.shape[1]/2)))
    cImg = img.copy()
    img = cv2.blur(img, (5, 5))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    ret, thresh = cv2.threshold(grad, 10, 255, cv2.THRESH_BINARY_INV)

    c, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    areas = [cv2.contourArea(c1) for c1 in c]
    maxAreaIndex = areas.index(max(areas))

    cv2.drawContours(cImg, c, maxAreaIndex, (255, 0, 0), -1)
    cv2.imshow('output', cImg)
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()