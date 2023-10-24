
import cv2
import numpy as np

import methods

testImgPath = '../test_images/solidWhiteCurve.jpg'
testImgPath2 = '../test_images/challange00136.jpg'

inputTestImg = cv2.imread(testImgPath2)

roiVertices = np.array([ [inputTestImg.shape[1]/3, inputTestImg.shape[0]/2],
                [0, inputTestImg.shape[0]],
                [inputTestImg.shape[1], inputTestImg.shape[0]],
                [2*inputTestImg.shape[1]/3, inputTestImg.shape[0]/2] ])

roiImage = methods.RegionOfInterest(inputTestImg, [roiVertices])

colorFilteredImg, colorMask = methods.filterByColor(roiImage)

# ne radi sobel
# gray = cv2.cvtColor(colorFilteredImg, cv2.COLOR_BGR2GRAY)
# sobel = np.int32(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=5))

lines = cv2.HoughLinesP(colorMask, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
for points in lines:
    x1, y1, x2, y2 = np.array(points[0])
    cv2.line(inputTestImg, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow('mask', inputTestImg)
cv2.waitKey(0)
