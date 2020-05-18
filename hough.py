import cv2
import numpy as np
import matplotlib.pyplot as plt


def detectCircles(img,threshold,region,radius = None):
    (M,N) = img.shape
    if radius == None:
        R_max = np.max((M,N))
        R_min = 1
    else:
        [R_max,R_min] = radius

    R = R_max - R_min
    
#Initializing accumulator array.
    A = np.zeros((R_max,M+2*R_max,N+2*R_max))
    B = np.zeros((R_max,M+2*R_max,N+2*R_max))

    theta = np.arange(0,360)*np.pi/180
    edges = np.argwhere(img[:, :])
    
# vote in accumulator array
    for val in range(R):
        print(val)
        for x, y in edges:
            r = R_min+val
            for angle in theta:
                a = x - int(np.round(r*np.cos(angle)))
                b = y - int(np.round(r*np.sin(angle)))
                A[r, a, b] += 1
        A[r][A[r] < A[r].max()] = 0
        A[r][A[r] < threshold*len(theta)] = 0
        
# get the best in a region
    for r, a, b in np.argwhere(A):
        temp = A[r-region:r+region,a-region:a+region,b-region:b+region]
        try:
            r1, a1, b1 = np.unravel_index(np.argmax(temp),temp.shape)
        except:
            continue
        B[r + (r1-region),a+(a1-region),b+(b1-region)] = 1
    return B

def displayCircles(A):
    img = cv2.imread(file_path)
    fig = plt.figure()
    plt.imshow(img)
    circleCoordinates = np.argwhere(A)
    circle = []
    for r,x,y in circleCoordinates:
        circle.append(plt.Circle((y,x),r,color=(0,1,0),fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
    plt.show()

file_path = 'shoeprint/circle.png'
img = cv2.imread(file_path, 0)
img = cv2.medianBlur(img, 5)
high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
low_thresh = 0.5 * high_thresh
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
res = cv2.Canny(img, low_thresh, high_thresh)
res = detectCircles(res, 0.37, 15,radius=[70,20])
displayCircles(res)
