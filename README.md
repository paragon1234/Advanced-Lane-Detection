## Advanced Lane Detection Project

### Goals

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration3.jpg "Distorted"
[image2]: ./undist_calibration3.jpg "Un-distorted"
[image3]: ./test_images/test1.png "Distorted Road Image"
[image4]: ./outImg3/3_0undist.png "Undistorted Road Image"
[image5]: ./outImg3/3_20.png "Warped1 Image"
[image6]: ./outImg3/3_21.png "Masked Warped1 Image"
[image7]: ./outImg3/3_1mask.png "Mask1 Image"
[image8]: ./outImg3/3_22.png "Warped Mask1 Image"
[image9]: ./outImg3/5_2warp.png "Warp Example"
[image10]: ./outImg3/3_3mask.png "Road Estimate"
[image11]: ./outImg3/3_4road.png "Fit Visual"
[image12]: ./outImg3/3_5Lane.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## Camera Calibration

### Compute the camera matrix and distortion coefficients

The code for this step is contained in lines 7 through 42 of the file  AdvancedLaneDetection.py.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function

![alt text][image1]

and obtained this result: 

![alt text][image2]

### Pipeline

#### 1. An example of a distortion-corrected image.

I applied the distortion correction to one of the test images using the `cv2.undistort()` function and following image was obtained:

![alt text][image4]


#### 2. Perspective transform, Color transforms, gradients to create a thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image. I used a 2 way approach, where I first applied perspective transform and then masking (perspective transform at lines 236 through 240, thresholding steps at lines 242 through 245 in AdvancedLaneDetection.py). 

For perspective transform, I chose to hardcode the source and destination points in the following manner:
    src = np.float32([[(250, 690), (1060, 690), (684, 450), (595, 450)]])     # define source points
    dst = np.float32([[(350, 690), (950, 690), (950, 0), (350, 0)]])     # define dst points
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

For masking I used combination of sobel and color masks

Here's an example of my output for this step.  

![alt text][image5]          

![alt text][image6]


Next, I performed these two operations in reverse order: first application of masking (Combine sobel and color masks) on test image, followed by perspective transform (mask-thresholding steps at lines 249 through 252, perspective transform at lines 254,  in AdvancedLaneDetection.py).  Here's an example of my output for this step.  

![alt text][image7]           

![alt text][image8]


Finally, I combined the 2 images. As the image obtained in second step contains noise, I reduced its magnitude by quarter (line 255).

![alt text][image9]


#### 4. Identificaton of lane-line pixels and fit their positions with a polynomial

We divided image into 9 vertical segments and applied convolution with a mask of '1' on each segment, to get the position of the maximum magnitude of the convolved signal. This position signifies the actual lane line in that segment (line 264 in AdvancedLaneDetection.py)

![alt text][image10]

We next fit a quadratic function with independent variable 'x' and dependent variable 'y' to the points within the line mask using numpy's polyfit function (line 265 in AdvancedLaneDetection.py)

![alt text][image11]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 269 through 283 in my code in AdvancedLaneDetection.py

The radius of curvature is based upon [this website](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) (line 278-283 in AdvancedLaneDetection.py)

ym_per_pixel is the factor used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters.

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code ((line 269-275 in AdvancedLaneDetection.py))

lane_center_position = (right_fitx + left_fitx) /2
center_dist = (car_position - lane_center_position) * xm_per_pix

#### 6. Example image of result plotted back down onto the road such that the lane area is identified clearly.

A polygon is generated based on plots of the left and right fits, warped back to the perspective of the original image using the inverse perspective matrix Minv(line 266 in my code in AdvancedLaneDetection.py) and overlaid onto the original image (line 287 in my code in AdvancedLaneDetection.py).  Here is an example of my result on a test image:

![alt text][image12]

---

### Pipeline (video)

#### Final video output.  
The pipeline performed reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

To make the lines on video more robust, I have used values across the frames viz. left_center, right_center, and average distance between them. 
1) If the distance between left_center and right_center of the first level of the frame is greater (by atleast some threshold) than the corresponding distance of the previous frame, then we correct the left_center and/or right_center so that it is close to the corresponding value in the previous frame (line 66-75 of tracker.py).
2) If for a particular level and lane, the lane-pixels are not identified (due to missing markings), then it is adjusted as a function of average distance between lane and the position of the other lane (line 103-106 of tracker.py). This case fixes the issue of missing lane on one side as in the case of discrete lane markings
3) If for a particular level, the distance between left and right lanes is different from the averge distance by a threshold, and the lane position is drifted by more than a threshold of the previous frame, then we apply correction to it (line 110-115 in tracker.py). This case fixes the issue of missing lane on one side and instead some noise is detected (like tyres of car in next lane)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This was a very tedious project involving tuning of several parameters. My approach also invalidates fits if the left and right base points aren't a certain distance apart (within some threshold) under the assumption that the lane width will remain relatively constant.
I couldn’t get the approach to work on the harder-challenge video, mainly because the lanes had large curvature, and as a result the lanes went outside the region of interest we chose for perspective transform. 
