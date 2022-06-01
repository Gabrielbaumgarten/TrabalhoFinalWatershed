from math import floor
import numpy as np
import cv2
import imutils
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

INPUT_IMAGE =  './colourWall.jpg'

def main ():

    frame = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    # frame = cv2.resize(frame, (floor(frame.shape[0]/4), floor(frame.shape[1]/4)))
    # frame = frame.astype(np.float)/255
    cv2.imshow('Original', frame)

    
    # deslocado = cv2. pyrMeanShiftFiltering ( img, 21 , 51 )
    # cv2.imshow('Deslocado', deslocado)

    # # gray = cv2.cvtColor(deslocado, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", gray)
    # thresh = cv2.threshold(gray,100, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow("Thresh", thresh)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.medianBlur(gray_frame, 7)
    # _, filtered = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    filtered = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(filtered.copy(), cv2.MORPH_OPEN, kernel, iterations = 2)

    # Find background area
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)

    # Find foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Add marker labels
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(frame, markers)
    frame[markers == -1] = [255, 0, 0]

    cv2.imshow('Camera', frame)
    cv2.imshow('Filtered', filtered)
    cv2.imshow('Unknown', unknown)
    cv2.imshow('BG', sure_bg)
    cv2.imshow('FG', sure_fg)


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