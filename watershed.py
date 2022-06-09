from math import floor
import numpy as np
import cv2
import imutils
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

INPUT_IMAGE =  './quadros.jpg'

def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    # img = cv2.resize(img, (floor(img.shape[0]/2), floor(img.shape[1]/2)))
    # img = img.astype(np.float)/255
    cv2.imshow('Original', img)

    
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.medianBlur(gray_img, 7)
    # _, filtered = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # # Remove noise
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(filtered.copy(), cv2.MORPH_OPEN, kernel, iterations = 2)

    # # Find background area
    # sure_bg = cv2.dilate(opening, kernel, iterations = 3)

    # # Find foreground area
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # # Find unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)

    sure_fg = cv2.imread ('./quadros_mascara.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Mask', sure_fg)
    # Add marker labels
    # _, markers = cv2.connectedComponents(sure_fg)
    markers = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # markers = markers + 1
    # markers[unknown == 255] = 0

    # Apply watershed
    # markers = cv2.watershed(img, markers)
    # img[markers == -1] = [255, 0, 0]

    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if sure_fg[row][col]==0 :
                # print('areas azul')
                img_hsl[row][col] = [100 ,100,20]
                # img_hsl[row][col] = [322 ,98,img_hsl[row][col][2]]

    img_hsl = cv2.cvtColor(img_hsl, cv2.COLOR_HLS2BGR)
    cv2.imshow('changeColor', img_hsl)


    cv2.imshow('Camera', img)


    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()