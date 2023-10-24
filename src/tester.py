
import cv2
import numpy as np

import methods

testImgPath = '../test_images/solidWhiteCurve.jpg'
testImgPath2 = '../test_images/challange00136.jpg'
testVideoPath = '../test_videos/project_video03.mp4'


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


def detectLanes(inputImg):
    roiVertices = np.array([ [4*inputImg.shape[1]/9, inputImg.shape[0]/2],
                    [0, inputImg.shape[0]],
                    [inputImg.shape[1], inputImg.shape[0]],
                    [5*inputImg.shape[1]/9, inputImg.shape[0]/2] ])

    roiImage = methods.RegionOfInterest(inputImg, [roiVertices])

    colorFilteredImg, colorMask = methods.filterByColor(roiImage)

    # ne radi sobel
    # gray = cv2.cvtColor(colorFilteredImg, cv2.COLOR_BGR2GRAY)
    # sobel = np.int32(cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=5))

    lines = cv2.HoughLinesP(colorMask, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)
    for points in lines:
        x1, y1, x2, y2 = np.array(points[0])
        cv2.line(inputImg, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return inputImg
    # cv2.imshow('mask', inputTestImg)
    # cv2.waitKey(0)

if __name__ == "__main__":
    videoPlayer(testVideoPath)
