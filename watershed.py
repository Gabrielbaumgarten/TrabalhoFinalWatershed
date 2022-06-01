from math import floor
import numpy as np
import cv2
import imutils
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

INPUT_IMAGE =  './parede.jpg'

def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (floor(img.shape[0]/4), floor(img.shape[1]/4)))
    # img = img.astype(np.float)/255
    cv2.imshow('Original', img)

    
    deslocado = cv2. pyrMeanShiftFiltering ( img, 21 , 51 )
    cv2.imshow('Deslocado', deslocado)

    # gray = cv2.cvtColor(deslocado, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    thresh = cv2.threshold(gray,100, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)

    # cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL,
    #         cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Contornos", cnts)

    # D = ndimage.distance_transform_edt(thresh)
    # localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)

    
    # markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    # labels = watershed(-D, markers, mask=thresh)
    # print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    
    # for label in np.unique(labels):
    #     # if the label is zero, we are examining the 'background'
    #     # so simply ignore it
    #     if label == 0:
    #         continue
    #     # otherwise, allocate memory for the label region and draw
    #     # it on the mask
    #     mask = np.zeros(gray.shape, dtype="uint8")
    #     mask[labels == label] = 255
    #     # detect contours in the mask and grab the largest one
    #     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    #         cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = imutils.grab_contours(cnts)
    #     c = max(cnts, key=cv2.contourArea)
    #     # draw a circle enclosing the object
    #     ((x, y), r) = cv2.minEnclosingCircle(c)
    #     cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
    #     cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # # show the output image
    # cv2.imshow("Output", img)


    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()