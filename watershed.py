from math import floor
import numpy as np
import cv2
import imutils
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

INPUT_IMAGE =  './colourWall.jpg'

def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    # img = cv2.resize(img, (floor(img.shape[0]/2), floor(img.shape[1]/2)))
    # img = img.astype(np.float)/255
    cv2.imshow('Original', img)

    
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.medianBlur(gray_img, 11)
    # _, filtered = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # cv2.imshow('Threshold', filtered)

    # # # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(filtered.copy(), cv2.MORPH_OPEN, kernel, iterations = 2)

    # # cv2.imshow('Opening', opening)

    # # # Find background area
    # sure_bg = cv2.dilate(opening, kernel, iterations = 3)
    sure_bg = cv2.imread ('./BG.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Sure BG', sure_bg)
    # cv2.imwrite ('BG.png', sure_bg)

    # # Find foreground area
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    sure_fg = cv2.erode(sure_bg, kernel, iterations = 7)

    cv2.imshow('Sure FG', sure_fg)

    # # Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    cv2.imshow('Unknown', unknown)

    # sure_fg = cv2.imread ('./quadros_mascara.jpg', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('Mask', sure_fg)
    # Add marker labels
    _, markers = cv2.connectedComponents(sure_fg)
    # markers = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(img, markers)

    # markers[0] = [1] * markers.shape[1]
    # markers[markers.shape[0]-1] = [1] * markers.shape[1]

    # for row in range(markers.shape[0]):
    #     markers[row][0] = 1
    #     markers[row][markers.shape[1]-1] = 1
    

    
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if markers[row][col]==1 :
    #             # print('areas azul')
                img_hls[row][col] = [80,(img_hls[row][col][1]),183.6]
    #             # img_hsl[row][col] = [322 ,98,img_hsl[row][col][2]]

    img_hls = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR)
    cv2.imshow('changeColor', img_hls)


    img[markers == -1] = [255, 0, 0]
    cv2.imshow('Contornos', img)


    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()