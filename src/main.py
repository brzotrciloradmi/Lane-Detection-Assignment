# 1. + camera calibration
# -> ROI /
# + -> Filtering by color  (Yellow and white line)
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
import matplotlib.pyplot as plt

import methods

cameraCalibrationPath = '../camera_cal/calibration*.jpg'
CalibrationTestOutputPath = '../calibrationTest/'
testVideoPath = '../test_videos/project_video01.mp4'


chessRows = 6
chessCols = 9

def videoPlayer(VideoPath):
    video = cv2.VideoCapture(VideoPath)
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            frame = detectLanes(frame)
            cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()


leftx_base_prev = 0
rightx_base_prev= 0
def detectLanes(inputImg):
    #Input image (TODO: switch to video when done)
    # inputImg = cv2.imread('../test_images/solidWhiteCurve.jpg')

    undistortedImg = methods.ImageUndistort(inputImg, mtx, dist, rvecs, tvecs)

    # ROI masking
    # print(undistortedImg.shape) #shape returns (width, height, channelNum)
    # Max shape needs to be decremented by 1 because of 0 start counting... fml
    roiVertices = np.int32([ [2*undistortedImg.shape[1]/5 + 50 - 1, 4*undistortedImg.shape[0]/6 -1],
                    [1*undistortedImg.shape[1]/5 + 50 - 1, 6*undistortedImg.shape[0]/7 - 1],
                    [4*undistortedImg.shape[1]/5 + 50 - 1, 6*undistortedImg.shape[0]/7 - 1],
                    [3*undistortedImg.shape[1]/5 + 50 - 1, 4*undistortedImg.shape[0]/6 - 1] ])

    dstVertices = np.int32([ [0, 0],
                    [0, undistortedImg.shape[0] - 1],
                    [undistortedImg.shape[1] - 1, undistortedImg.shape[0] - 1],
                    [undistortedImg.shape[1] - 1, 0] ])

    # for val in roiVertices:
    #     cv2.circle(undistortedImg,(val[0],val[1]),5,(0,255,0),-1)

    # for val in dstVertices:
    #     cv2.circle(undistortedImg,(val[0],val[1]),5,(255,0,0),-1)


    # Perspective warping
    warped = methods.Warper(undistortedImg, roiVertices, dstVertices)

    # Continue here
    colorFilteredImg, colorMask = methods.filterByColor(warped)


    #histogram of pixel density by x(widht)-axis
    #get midpoints of two lines and find their spread on histogram
    #use the mid point and spread to determine image zone of the line
    #find nonzero pixels in the zone and use them to polyfit a curve

    hist = np.sum(colorMask[colorMask.shape[0]//2:,:], axis=0)
    midpoint = int(hist.shape[0]/2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint

    #filter single midpoint peak noise
    global leftx_base_prev
    global rightx_base_prev
    if(leftx_base == 0):
        leftx_base = leftx_base_prev
    if(rightx_base == 0):
        rightx_base = rightx_base_prev
    leftx_base_prev = leftx_base
    rightx_base_prev = rightx_base

    cv2.circle(warped,(leftx_base, 0),5,(0,255,0),-1)
    cv2.circle(warped,(rightx_base, 0),5,(0,255,0),-1)

    #find image borders that define line zones
    farLeft_leftx_point = leftx_base
    farRight_leftx_point = leftx_base
    farLeft_rightx_point = rightx_base
    farRight_rightx_point = rightx_base

    while(hist[farLeft_leftx_point] != 0):
        farLeft_leftx_point -= 1
    while(hist[farRight_leftx_point] != 0):
        farRight_leftx_point += 1
    while(hist[farLeft_rightx_point] != 0):
        farLeft_rightx_point -= 1
    while(hist[farRight_rightx_point] != 0):
        farRight_rightx_point += 1

    farLeft_leftx_point -= 50
    farRight_leftx_point += 50
    farLeft_rightx_point -= 50
    farRight_rightx_point += 50

    cv2.circle(warped,(farLeft_leftx_point, 0),5,(0,0,255),-1)
    cv2.circle(warped,(farRight_leftx_point, 0),5,(0,0,255),-1)
    cv2.circle(warped,(farLeft_rightx_point, 0),5,(0,0,255),-1)
    cv2.circle(warped,(farRight_rightx_point, 0),5,(0,0,255),-1)

    #extract lines
    nonzero = colorMask.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    leftLaneX = []
    leftLaneY = []

    # index = 0
    # for x in np.nditer(nonzerox):
    #     if(x > farLeft_leftx_point and x < farRight_leftx_point):
    #     # if(x < midpoint):
    #         leftLaneX.append(nonzerox[index])
    #         leftLaneY.append(nonzeroy[index])
    #     index += 1


    good_left_inds = ((nonzerox >= farLeft_leftx_point) & (nonzerox < farRight_leftx_point)).nonzero()[0]
    for i in good_left_inds:
        leftLaneX.append(nonzerox[i])
        leftLaneY.append(nonzeroy[i])

    if(len(leftLaneX) > 0 and len(leftLaneY) > 0):
        leftLineCoeffs = np.polyfit(leftLaneX, leftLaneY, 2)
        print(leftLineCoeffs)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = leftLineCoeffs[0]*ploty**2 + leftLineCoeffs[1]*ploty + leftLineCoeffs[2]

    # leftLinePlot = []
    # for i in range(len(ploty)):
    #     leftLinePlot.append([left_fitx[i], ploty[i]])

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])

    # leftLinePlot = np.int32(leftLinePlot)
    # leftLinePlot = leftLinePlot.reshape((-1, 1, 2))

    cv2.polylines(warped, np.int32([pts_left]), False, (0, 255, 0), 2)

    return warped

if __name__ == '__main__':
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

    #start reading the test video and parse data
    videoPlayer(testVideoPath)






