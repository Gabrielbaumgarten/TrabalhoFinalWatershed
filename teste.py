import numpy as np
import cv2
import imutils

INPUT_IMAGE =  './img.webp'

def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    # img = img.astype(np.float)/255
    cv2.imshow('Original', img)

    
    deslocado = cv2. pyrMeanShiftFiltering ( img, 21 , 51 )
    cv2.imshow('Deslocado', deslocado)

    gray = cv2.cvtColor(deslocado, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("[INFO] {} unique contours found".format(len(cnts)))
    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the contour
        ((x, y), _) = cv2.minEnclosingCircle(c)
        cv2.putText(img, "#{}".format(i + 1), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Image", img)

    # cv2.imwrite ('02 - out.png', chormaKey*255)


    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()