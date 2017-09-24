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
[image3]: ./test_images/test3.jpg "Distorted Road Image"
[image4]: ./outImg3/5_0undist.jpg "Undistorted Road Image"
[image5]: ./outImg3/5_20.jpg "Warped1 Image"
[image6]: ./outImg3/5_21.jpg "Masked Warped1 Image"
[image7]: ./outImg3/5_1mask.jpg "Mask1 Image"
[image8]: ./outImg3/5_22.jpg "Warped Mask1 Image"
[image9]: ./outImg3/5_2warp.jpg "Warp Example"
[image10]: ./outImg3/5_3mask.jpg "Road Estimate"
[image11]: ./outImg3/5_4road.jpg "Fit Visual"
[image12]: ./outImg3/5_5Lane.jpg "Output"
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

I used a combination of color and gradient thresholds to generate a binary image. I used a 2 way approach, where I first applied perspective transform and then masking (perspective transform at lines 236 through 240, thresholding steps at lines 242 through 245 in AdvancedLaneDetection.py). For perspective transform, I chose to hardcode the source and destination points in the following manner:
    src = np.float32([[(250, 690), (1060, 690), (684, 450), (595, 450)]])     # define source points
    dst = np.float32([[(350, 690), (950, 690), (950, 0), (350, 0)]])     # define dst points
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Here's an example of my output for this step.  

![alt text][image5]          

![alt text][image6]


Next, I performed these two operations in reverse order: I first applied masking and then perspective transform on the test image (thresholding steps at lines 249 through 252, perspective transform at lines 254,  in AdvancedLaneDetection.py).  Here's an example of my output for this step.  

![alt text][image7]           

![alt text][image8]


Finally, I combined the 2 images. As the image obtained in second step contains noise, I reduced its magnitude by quarter (line 255).

![alt text][image9]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 269 through 283 in my code in AdvancedLaneDetection.py

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### Final video output.  
The pipeline performed reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
