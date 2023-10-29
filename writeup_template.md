# Lane finding project

---

The goals / steps of this project are the following:

* 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* 2. Apply a distortion correction to raw images.
* 3. Apply a perspective transform to rectify binary image ("birds-eye view").
* 4. Extract color mask of pixels of interest to create a thresholded binary image.
* 5. Find lane zones of interest (left and right lane).
* 6. Detect lane pixels and fit to find the lane boundary.
* 7. Determine the curvature of the lane and vehicle position with respect to center.
* 8. Warp the detected lane boundaries back onto the original image.
* 9. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./calibrationTest/calibration2.jpg "Undistorted"
[image2]: ./mdImages/distortUndistort.jpg "Input/Output of image distortion"
[imageROI]: ./mdImages/ROIVertices.jpg "ROI vertices on undistorted img"
[imageWarp]: ./mdImages/WarpExample.jpg "Input/Output warp example"
[ColorMask]: ./mdImages/ColorMask.jpg "Input/Output colorMask example"
[PeakHist]: ./mdImages/PeakHistogram.jpg "Input/Output histogram example"
[LaneZones]: ./mdImages/ZonePts.jpg "Lane Zones"
[DetectedLane]: ./mdImages/DetectedLane.jpg "Detected Lane"
[curveFormula]: ./mdImages/curveFormula.jpg "Curve Formula"
[Output]: ./mdImages/Output.jpg "Output"
---

### Writeup / README

##How to Run the project
Position the terminal window into the `src/` folder and run `python3 main.py`. To change the video that is being parsed, open `src/main.py` file and change `testVideoPath` string variable. The Algorithm runs reasonably well in all test videos and `challenge01.mp4` video.
## 1. Camera Calibration and testing

#### 1. Camera calibration and extraction of calibration data

Before any image processing for lane detection is done, first camera calibration step is executed to get the parameters needed to undistort images. This is done so because each camera will introduce distortion (radial and tangentual) upon gathered images.

The first thing that is done is cameraCalibration(chessRows, chessCols, chessFolderPath) method is called upon the start of the program. The role of this step is to get camera matrix and distortion coefficients. These parameters are essential to be able to undistort captured images. First and second parameter of the method represents how many chess rows and columns on the chessboard from calibration images, and chessFolderPath as the name suggests is the path to the calibration iamges folder.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

The output of the CameraCalibration(chessRows, chessCols, chessFolderPath) method is the camera matrix, distortion coefficients array and calibration error.

#### 2. Calibration testing
After the `CameraCalibration` method is finshed, image undistortion is done upon all chessboard images inside camera_cal folder. The output is saved inside `calibrationTest` folder. The next image is one of the undistorted images.
![alt text][image1]

### Pipeline (single images)

After we have gathered the data needed to undistort images captured from the camera, we can now start lane detection processing. The pipeline for the lane detection is implemented inside `DetectLanes(inputImg)` method.

#### 1. Distortion-correction

The first thing that is done to the `inputImg` is image undistortion:

![alt text][image2]

#### 2. Region of interest for warping
After image undistortion, the next step is to warp the image into the bird eye view. For the warping to be successful, first region of interest (ROI) vertices need to be defined. This is done inside the `findROI(imageShape)` method. The method is relatively trivial that tries to find relative ROI vertices of the road and return them. Because I could not define one general set of ROI vertices for all test videos, I made a special case for the test3 video. Image warping would still be possible with only one set of ROI vertices but I wanted to have a right angle of lanes when warping is executed.

![alt text][imageROI]

#### 3. Image warping
Now that we have our ROI vertices, we can do a bird eye perspective transform using method `Warper(img, src, dst)`. The input of the method is the image to be warped, source vertices (in our case ROI vertices) and destination vertices from which the perpective transformation matrix is calculated. The method calculates the perpective transform matrix and with it warps the input image. The result example is shown below:

![alt text][imageWarp]

#### 4. Color transform and binary image extraction

After warping, the warped image is the sent to the `FilterByColor(inputImg)` method which takes the warped image as the input and returns a binary color mask. The input image is transformed into the HSV color space from where all pixels that are not in the yellow or white HSV color range are filtered out. The binary mask that retains the information of parts of the image that contain the yellow and white colors is returned as output. For the given test videos, after running the pipeline with only the `colorFiltering` as the main method to get a binary image the lane was succesfully detected so no further binary filtering was done.
The example input/output of this step can be seen below:

![alt text][colorMask]

#### 4. Histogram peak detection

Now that we have a binary image with only the lane information, the next part is the bread and butter of this project.
After we have gained the binary image, we need to somehow decide and group pixels from the left lane and right lane into two separate sets. After that, we can use the pixels from those two sets to polyfit a curve onto each lane.
To get the left and right lane pixel sets, we first need to find the zones of the binary image where the lanes are located. To do so, we'll be getting a histogram of the colorMask binary image with respect to the x-axis. Each portion of the histogram below displays how many white pixels are in each column of the image. We then take the highest peaks of each side of the image, one for each lane line. Here's what the histogram looks like, next to the binary image:

![alt text][PeakHist]

This step is implemented inside PeakHistogram(colorMask) method which returns histogram data, left lane x-axis peak coordinate and right lane x-axis peak coordinate.

####5. Zoning of the lanes
Now that we have left lane x-axis peak coordinate and right lane x-axis peak coordinate, we can do lane zoning.
The peak x-coordinates of left and right lane correlate one to one with the binary color mask that was the input into the peak histogram calculation. This means that we can define lane zones around these peak x-axis coordinates.
The algorithm for finding the x-axis range of these zones is trivial. We initiate two points at the left (and right) lane peak and move one point to the left and one point to the right for each nonzero x-axis pixel on the histogram graph. After we encounter a zero value pixel, we stop moving the point.

The result of the algorithm looks something like this:

![alt text][LaneZones]

#### 5. Lane pixel extraction and lane fitting

The idea is this, after we have defined the left and right lane zones, lets take nonzero pixels inside these zones and do a polyfitting algorithm upon them. Simplest way to describe the algortihm is to show you the code:
```python
    # Extract line points
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
```

Inside `nonzero` array are located coordinates of all nonzero points of `colorMask` image. To differentiate between left lane and right lane nonzero points, we use `good_left_inds` and `good_right_inds` arrays which store indexes for the left and right nonzero pixels. The extracted coordinates for nonzero pixels for the left and right lane are located in `leftLaneX`, `leftLaneY`, `rightLaneX`, `rightLaneY` arrays.

These arrays of coordinates are used as input to the `np.polyfit` method which returns the coefficients that describe a polynomial curve of the 2nd order.

Using the right and left coeffients have been found, we can visualize the result on the warped image and then unwarp the image back by inverting the `src` and `dst` input arguments of `Warper` method (meaning we put `ROIVertices` as `dst` argument):

![alt text][DetectedLane]

#### 6. Lane curvature and vehicle lane offset extraction

With the detected left and right lane curves it is possible to calculate lane curvature and vehicle offset position from the lane center.

The formula for the lane curvature is shown in the image below:

![alt text][curveFormula]

The curvature of the lanes f(y) are calculated by using the formula R(curve), so the vehicle position is calculated as the difference between the image center and the lane center.
Source: https://www.researchgate.net/figure/The-curvature-of-the-lanes-fy-are-calculated-by-using-the-formula-Rcurve-so-the_fig4_344123734

The implementation of curvature calculation and vehicle offset calculation is done inside the `GetCurve(img, leftx, lefty)` method where `img` is the image from which we derived detected lanes, `leftx` and `lefty` represent arrays of x,y coordinates upon which polyfitting is done.

```python
def GetCurve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/720 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, center)
```
Because we need curvature to be in meters, we need to scale our detected curves into meters per pixel first. So we fit the new curves with the scale factor. After that we implement the curvature formula.
For the lane center, we just take the most bottom two points of the curves and calculate a mean from the two points. The Car position can be calculated simply by dividing the width of the input image by two, assuming the camera is located on center of the car. Then relative offset from the lane center is derived by substracting car position from the lane center.

At the end, `left_curverad`, `right_curverad`, and `center` parameters are returned which represent left and right lane curvature and center represent car offset from the lane center.

Visualization of the data on processed output image looks like this:
![alt text][Output]


### Pipeline (video)

#### 1. Video output link.

Here's a [link to my video result.](./mdImages/output.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Because only color filtering is used to binarize the image, and no special filtering is done to get better image prior or after the color filter, other then the test videos I expect poor performance. But because for the test videos color filtering gave good enough results, no further filtering was done. If I were to continue this project, I'd probably first think on how to get a more robust binary extraction algorithm for lane detection.

It also seems that in certain conditions the memory or cpu load jumps implying that theres room for algorithm optimization.

I'd also probably switch up the core lane detection (lane zoning) because I fear that in "hard" lane curves the algorithm would perform poorly.