import cv2
import numpy as np
from matplotlib import pyplot as plt

# get input by implementing canny edge detection
img_path = 'shoeprint/circle.png'
img = cv2.imread(img_path, 0)
img = cv2.medianBlur(img, 3)
high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
low_thresh = 0.5*high_thresh
img_canny = cv2.Canny(img, low_thresh, high_thresh)
# show edge img
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_canny, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

bimg = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(bimg,cv2.COLOR_GRAY2BGR)
# hough circle transformation
circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, 1, 20)
circles = np.uint16(np.around(circles))
for i in circles[0]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.imwrite('output_circle.png', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()



