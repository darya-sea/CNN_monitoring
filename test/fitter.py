import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('00268N.tif',cv.IMREAD_ANYCOLOR) # queryImage
img2 = cv.imread('00268C.tif',cv.IMREAD_ANYCOLOR) # trainImage
cropped = img2[3:1533, 3:2045]
#cropped = img2[567:978, 78:565]
resized_cropped = cv.resize(cropped, (2048, 1536))
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(resized_cropped,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

#dimensions = img2.shape
#height = img2.shape[0]
#width = img2.shape[1]
#channels = img2.shape[2]
#print('Dimensions ', dimensions)
#print('Height ', height)
#print('Width ', width)
#print('Channels ', channels)
# cv.drawMatchesKnn expects list of lists as matches.
#img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(img3),plt.show()

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#h, status = cv.findHomography(src_pts, dst_pts)
#im_out = cv.warpPerspective(img1, h, (img2.shape[1],img2.shape[0]))
# Display images
#cv.imshow("Source Image", img1)
#cv.imshow("Destination Image", img2)
#cv.imshow("Warped Source Image", im_out)
 
#cv.waitKey(0)
#overlay = cv.add(img1,resized_cropped)
cv.imwrite('test.tif', resized_cropped)
