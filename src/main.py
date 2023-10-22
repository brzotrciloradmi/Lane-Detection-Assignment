# 1. camera calibration
# -> ROI
# -> Filtering by color? (Yellow and white line)
# -> Gray scale
# -> (High pass Filtering)
# 2. -> Edge detection (algorithm)
# 3. -> Perspective transform
# -> (Filtering)
# 4. -> Lane detection (curve/lane)
# 5. -> curvature radius & vehicle offset data aqusition
# 6 -> Perspective transform back and lane highlight

import cv2
import numpy as np
import glob

import methods

cameraCalibrationPath = '../camera_cal/calibration*.jpg'
CalibrationTestOutputPath = '../calibrationTest/'

chessRows = 6
chessCols = 9

#1 Camera calibration
mtx, dist, rvecs, tvecs, calibrationError = methods.CameraCalibration(chessRows, chessCols, cameraCalibrationPath)
print("Total Calibration Error: ", calibrationError)

#1.1 Undistorted img test
for path in glob.glob(cameraCalibrationPath):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    undistorted = methods.ImageUndistort(gray, mtx, dist, rvecs, tvecs)
    concatedImgs = np.hstack((gray, undistorted))

    calOutputFilePath = CalibrationTestOutputPath + path.split('/')[len(path.split('/')) - 1]
    cv2.imwrite(calOutputFilePath, concatedImgs)

#Input image (TODO: switch to video when done)
inputTestImg = cv2.imread('../test_images/solidWhiteCurve.jpg')
# cv2.imshow('input test' , inputTestImg)
# cv2.waitKey(1000)

undistortedImg = methods.ImageUndistort(inputTestImg, mtx, dist, rvecs, tvecs)
# cv2.imshow('undistorted test' , undistortedImg)
# cv2.waitKey(1000)

# ROI masking
roiVertices = np.array([[(0, undistortedImg.shape[0]),
               (undistortedImg.shape[1]/4, undistortedImg.shape[0]/2),
               (3*undistortedImg.shape[1]/4, undistortedImg.shape[0]/2),
               (undistortedImg.shape[1], undistortedImg.shape[0]),]], np.int32)

print(roiVertices)
roiFilteredImg = methods.RegionOfInterest(undistortedImg, roiVertices)
cv2.imshow('roi test' , roiFilteredImg)
cv2.waitKey(10000)

# Perspective warping
# warped = methods.Warper(undistortedImg, roiVertices, np.array([[[0, 0], [0, undistortedImg.shape[0]], [undistortedImg.shape[1], undistortedImg.shape[0]], [undistortedImg.shape[1], 0] ]], np.float32))
# cv2.imshow('roi test' , warped)
# cv2.waitKey(10000)





