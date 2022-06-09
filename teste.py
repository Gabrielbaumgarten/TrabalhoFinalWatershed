from statistics import mean
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
    # img = cv2.imread("quadros.jpg")
    img = cv2.resize(img, (round(img.shape[0]/2), round(img.shape[1]/2)))
    cImg = img.copy()
    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray_original, (5, 5))
    cv2.imshow("original", img)
    cv2.imshow("gray", gray)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    cv2.imshow('grad_x', abs_grad_x)
    cv2.imshow('grad_y', abs_grad_x)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv2.imshow('grad', grad)

    ret, thresh = cv2.threshold(grad, 13, 255, cv2.THRESH_BINARY_INV)
    # ret, thresh = cv2.threshold(grad, 10, 255, cv2.THRESH_BINARY)

    cv2.imshow('thresh', thresh)

    c, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    areas = [cv2.contourArea(c1) for c1 in c]
    maxAreaIndex = areas.index(max(areas))

    # for i in range(len(areas)):
    #     cv2.drawContours(cImg, c, i, (255, 0, 0), 1)
    gray_original = gray_original.astype(np.float32)/255
    cv2.drawContours(cImg, c, maxAreaIndex, (255, 0, 0), -1)
    # cv2.drawContours(cImg, c, maxAreaIndex, (255, 0, 0), -1)
    cv2.imshow('output', cImg)

    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if cImg[row][col][0] == 255 and cImg[row][col][1] == 0 and cImg[row][col][2] == 0 :
                img_hls[row][col] = [80,(img_hls[row][col][1]),183.6]
                # img_hls[row][col] = [80,((img_hls[row][col][1]*0.9) + (178.5*0.1)),183.6]
                # img_hls[row][col] = [80,204,183.6]

    img_hls = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR)
    cv2.imshow('changeColor', img_hls)



    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()