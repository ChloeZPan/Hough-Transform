import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
# get input by implementing canny edge detection
img_path = 'shoeprint/circle.png'
img = cv2.imread(img_path, 0)
bimg = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
high_thresh, thresh_im = cv2.threshold(bimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
low_thresh = 0.5*high_thresh
# hough circle transformation

for r in range(1, bimg.shape[0]):
    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, 1, 20, minRadius=r)
    if circles is not None:
        print(r)
        circles = np.uint16(np.around(circles))
        for i in circles[0]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()