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

import methods

cameraCalibrationPath = '../camera_cal/calibration*.jpg'
CalibrationTestOutputPath = '../calibrationTest/'
# testVideoPath = '../test_videos/project_video02.mp4'
testVideoPath = '../test_videos/challenge01.mp4'

chessRows = 6
chessCols = 9

#predefined ROIs
def findROI(imgShape):
    if testVideoPath == '../test_videos/project_video03.mp4':
        print('AAA')
        roiVertices = np.int32([ [2*imgShape[1]/5 + 10, 4*imgShape[0]/6 -1],
                [1*imgShape[1]/5 + 70 - 1, 6*imgShape[0]/7 - 1],
                [4*imgShape[1]/5 + 70 - 1, 6*imgShape[0]/7 - 1],
                [3*imgShape[1]/5 + 40 - 1, 4*imgShape[0]/6 - 1] ])
    else:
        roiVertices = np.int32([ [2*imgShape[1]/5 + 43 - 1, 4*imgShape[0]/6 -1],
                [1*imgShape[1]/5 + 50 - 1, 6*imgShape[0]/7 - 1],
                [4*imgShape[1]/5 + 50 - 1, 6*imgShape[0]/7 - 1],
                [3*imgShape[1]/5 + 43 - 1, 4*imgShape[0]/6 - 1] ])
    return roiVertices


def videoPlayer(VideoPath):
    video = cv2.VideoCapture(VideoPath)
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            frame = detectLanes(frame)
            cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

def detectLanes(inputImg):
    #Input image (TODO: switch to video when done)
    # inputImg = cv2.imread('../test_images/solidWhiteCurve.jpg')

    undistortedImg = methods.ImageUndistort(inputImg, mtx, dist, rvecs, tvecs)

    # ROI masking
    print(undistortedImg.shape) #shape returns (width, height, channelNum)
    # Max shape needs to be decremented by 1 because of 0 start counting... fml
    roiVertices = findROI(undistortedImg.shape)
    # roiVertices = np.int32([ [2*undistortedImg.shape[1]/5 + 50 - 1, 4*undistortedImg.shape[0]/6 -1],
    #                 [1*undistortedImg.shape[1]/5 + 50 - 1, 6*undistortedImg.shape[0]/7 - 1],
    #                 [4*undistortedImg.shape[1]/5 + 50 - 1, 6*undistortedImg.shape[0]/7 - 1],
    #                 [3*undistortedImg.shape[1]/5 + 50 - 1, 4*undistortedImg.shape[0]/6 - 1] ])

    dstVertices = np.int32([ [0, 0],
                    [0, undistortedImg.shape[0] - 1],
                    [undistortedImg.shape[1] - 1, undistortedImg.shape[0] - 1],
                    [undistortedImg.shape[1] - 1, 0] ])

    # Perspective warping
    warped = methods.Warper(undistortedImg, roiVertices, dstVertices)

    # Continue here
    colorFilteredImg, colorMask = methods.filterByColor(warped)

    lines = cv2.HoughLinesP(colorMask, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=1000)

    if(type(lines) != type(None)):
        for points in lines:
            x1, y1, x2, y2 = np.array(points[0])
            cv2.line(warped, (x1, y1), (x2, y2), (0, 255, 0), 10)

    unwarped = methods.Warper(warped, dstVertices, roiVertices)
    undistortedImg = cv2.fillPoly(undistortedImg, [roiVertices], (0, 0, 0))
    outputImg = undistortedImg + unwarped

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






