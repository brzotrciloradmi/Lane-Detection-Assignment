import cv2
import numpy as np
import glob

def CameraCalibration(chessRows, chessCols, chessFolderPath):
    # Define the chess board rows and columns
    rows = chessRows
    cols = chessCols

    # Set the termination criteria for the corner sub-pixel algorithm
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Create the arrays to store the object points and the image points
    objectPointsArray = []
    imgPointsArray = []

    # Loop over the image files
    for path in glob.glob(chessFolderPath):
        # Load the image and convert it to gray scale
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        # Make sure the chess board pattern was found in the image
        if ret:
            # Refine the corner position
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Add the object points and the image points to the arrays
            objectPointsArray.append(objectPoints)
            imgPointsArray.append(corners)

    # Calibrate the camera and save the results
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)

    # Find Camera Calibration error
    calibrationError = 0

    for i in range(len(objectPointsArray)):
        imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
        calibrationError += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

    calibrationError = calibrationError / len(objectPointsArray)

    return mtx, dist, rvecs, tvecs, calibrationError


def ImageUndistort(img, mtx, dist, rvecs, tvecs):
    # Load one of the test images
    h, w = img.shape[:2]
    # Obtain the new camera matrix and undistort the image
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
    return undistortedImg



def RegionOfInterest(inputImg, roiVertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(inputImg)    # Retrieve the number of color channels of the image.
    channelCount = inputImg.shape[2]    # Create a match color with the same color channel counts.
    matchMaskColor = (255,) * channelCount

    # Fill inside the polygon
    cv2.fillPoly(mask, np.int32(roiVertices), matchMaskColor)

    # Returning the image only where mask pixels match
    maskedImage = cv2.bitwise_and(inputImg, mask)
    return maskedImage


def filterByColor(inputImg):
    inputImgHSV = cv2.cvtColor(inputImg, cv2.COLOR_BGR2HSV)

    #filter by white color
    lower_white = np.array([0, 0, 225])
    upper_white = np.array([255, 255, 255])
    whiteMask = cv2.inRange(inputImgHSV, lower_white, upper_white)

    #filter by yellow color
    lower_yellow = np.array([10, 50, 150])
    upper_yellow = np.array([50, 255, 255])
    yellowMask = cv2.inRange(inputImg, lower_yellow, upper_yellow)

    mask = whiteMask + yellowMask
    output = cv2.bitwise_and(inputImg, inputImg, mask = mask)

    # cv2.imshow('output', output)
    # cv2.waitKey(0)

    return output, mask

def Warper(img, src, dst):
    # Compute and apply perpective transform
    imgSize = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    warped = cv2.warpPerspective(img, M, imgSize, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped