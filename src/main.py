import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

import methods

cameraCalibrationPath = '../camera_cal/calibration*.jpg'
CalibrationTestOutputPath = '../calibrationTest/'
testVideoPath = '../test_videos/project_video01.mp4'
# testVideoPath = '../test_videos/challenge01.mp4'

chessRows = 6
chessCols = 9

#predefined ROIs
def findROI(imgShape):
    #special case for this one
    if testVideoPath == '../test_videos/project_video03.mp4':
        roiVertices = np.int32([ [2*imgShape[1]/5 + 10, 4*imgShape[0]/6 -1],
                [1*imgShape[1]/5 + 70 - 1, 6*imgShape[0]/7 - 1],
                [4*imgShape[1]/5 + 90 - 1, 6*imgShape[0]/7 - 1],
                [3*imgShape[1]/5 + 20 - 1, 4*imgShape[0]/6 - 1] ])
    else:
        roiVertices = np.int32([ [2*imgShape[1]/5 + 43 - 1, 4*imgShape[0]/6 -1],
                [1*imgShape[1]/5 + 50 - 1, 6*imgShape[0]/7 - 1],
                [4*imgShape[1]/5 + 50 - 1, 6*imgShape[0]/7 - 1],
                [3*imgShape[1]/5 + 43 - 1, 4*imgShape[0]/6 - 1] ])
    return roiVertices


def VideoPlayer(VideoPath):
    video = cv2.VideoCapture(VideoPath)
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            frame, meanCurve, centerOffset = DetectLanes(frame)
            cv2.putText(frame, 'Lane Curvature: {:.0f} m'.format(meanCurve), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, 'Vehicle offset: {:.4f} m'.format(centerOffset), (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()


leftx_base_prev = 0
rightx_base_prev= 0
def DetectLanes(inputImg):

    undistortedImg = methods.ImageUndistort(inputImg, mtx, dist)

    # ROI masking
    # print(undistortedImg.shape) #shape returns (width, height, channelNum)
    roiVertices = findROI(undistortedImg.shape)

    # Max shape needs to be decremented by 1 because of 0 start counting... gaah
    dstVertices = np.int32([ [0, 0],
                    [0, undistortedImg.shape[0] - 1],
                    [undistortedImg.shape[1] - 1, undistortedImg.shape[0] - 1],
                    [undistortedImg.shape[1] - 1, 0] ])


    # for val in roiVertices:
    #     cv2.circle(undistortedImg,(val[0],val[1]),5,(0,255,0),-1)

    # Perspective warping
    warped = methods.Warper(undistortedImg, roiVertices, dstVertices)


    # Binary mask extract from color filtering
    colorFilteredImg, colorMask = methods.FilterByColor(warped)


    # Histogram Peak Detection
    hist, leftx_base, rightx_base, histogram_image = methods.PeakHistogram(colorMask)

    # Filter single midpoint peak noise
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

    # Find image borders that define line zones
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

    # Spread the zones to be sure we get the entire lane
    farLeft_leftx_point -= 50
    farRight_leftx_point += 50
    farLeft_rightx_point -= 50
    farRight_rightx_point += 50

    #Visualize zones
    # cv2.circle(warped,(farLeft_leftx_point, 0),5,(0,0,255),-1)
    # cv2.line(warped, (farLeft_leftx_point, 0), (farLeft_leftx_point, warped.shape[0]-1), (0, 0, 255), 3)
    # cv2.circle(warped,(farRight_leftx_point, 0),5,(0,0,255),-1)
    # cv2.line(warped, (farRight_leftx_point, 0), (farRight_leftx_point, warped.shape[0]-1), (0, 0, 255), 3)
    # cv2.circle(warped,(farLeft_rightx_point, 0),5,(0,0,255),-1)
    # cv2.line(warped, (farLeft_rightx_point, 0), (farLeft_rightx_point, warped.shape[0]-1), (0, 0, 255), 3)
    # cv2.circle(warped,(farRight_rightx_point, 0),5,(0,0,255),-1)
    # cv2.line(warped, (farRight_rightx_point, 0), (farRight_rightx_point, warped.shape[0]-1), (0, 0, 255), 3)

    # Extract lines
    nonzero = colorMask.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    leftLaneX = []
    leftLaneY = []
    rightLaneX = []
    rightLaneY = []

    # Find candidates for lane fitting
    good_left_inds = ((nonzerox >= farLeft_leftx_point) & (nonzerox < farRight_leftx_point)).nonzero()[0]
    good_right_inds = ((nonzerox >= farLeft_rightx_point) & (nonzerox < farRight_rightx_point)).nonzero()[0]
    for i in good_left_inds:
        leftLaneX.append(nonzerox[i])
        leftLaneY.append(nonzeroy[i])
    for i in good_right_inds:
        rightLaneX.append(nonzerox[i])
        rightLaneY.append(nonzeroy[i])

    # Find right and left polynom lane coeffs
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    if(len(leftLaneX) > 0 and len(leftLaneY) > 0):
        leftLineCoeffs = np.polyfit(leftLaneY, leftLaneX, 2)
    if(len(rightLaneX) > 0 and len(rightLaneY) > 0):
        rightLineCoeffs = np.polyfit(rightLaneY, rightLaneX, 2)

    # In case no lane is found, return default undistorted img
    if(len(rightLaneX) == 0 or len(rightLaneY) == 0 or len(leftLaneX) == 0 or len(leftLaneY) == 0):
        return undistortedImg, -1, -1

    # Calculate all lane pixes inside the image
    left_fitx   = leftLineCoeffs[0]*ploty**2 + leftLineCoeffs[1]*ploty + leftLineCoeffs[2]
    right_fitx  = rightLineCoeffs[0]*ploty**2 + rightLineCoeffs[1]*ploty + rightLineCoeffs[2]

    # Plot the detected lines
    leftLinePlot = []
    for i in range(len(ploty)):
        leftLinePlot.append([left_fitx[i], ploty[i]])

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])

    left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(warped, np.int_(points), (50,150,55))

    cv2.polylines(warped, np.int32([pts_left]), False, (0, 255, 0), 40)
    cv2.polylines(warped, np.int32([pts_right]), False, (0, 255, 0), 40)

    # Unwarp the result and merge it with the undistorted input image
    unwarped = methods.Warper(warped, dstVertices, roiVertices)
    undistortedImg = cv2.fillPoly(undistortedImg, [roiVertices], (0, 0, 0))
    outputImg = undistortedImg + unwarped

    curverad = methods.GetCurve(warped, left_fitx, right_fitx)
    meanCurve = np.mean([curverad[0], curverad[1]])
    centerOffset = curverad[2]
    return outputImg, meanCurve, centerOffset

if __name__ == '__main__':
    #1 Camera calibration
    mtx, dist, calibrationError = methods.CameraCalibration(chessRows, chessCols, cameraCalibrationPath)
    print("Total Calibration Error: ", calibrationError)

    #1.1 Undistorted img test
    for path in glob.glob(cameraCalibrationPath):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        undistorted = methods.ImageUndistort(gray, mtx, dist)
        concatedImgs = np.hstack((gray, undistorted))

        calOutputFilePath = CalibrationTestOutputPath + path.split('/')[len(path.split('/')) - 1]
        cv2.imwrite(calOutputFilePath, concatedImgs)

    #start reading the test video and parse data
    VideoPlayer(testVideoPath)






