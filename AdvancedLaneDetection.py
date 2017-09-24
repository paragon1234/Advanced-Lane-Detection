import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tracker import tracker

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        #cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey(500)


# Test undistortion on an image
img = cv2.imread('camera_cal/calibration2.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(5, 100)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient=='y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

#Not Used
# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output

#Not used
# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(grad)
    binary_output[(grad >= thresh[0]) & (grad <= thresh[1])] = 1
    return binary_output

#Applies color thresholding on s channel of HLS color space, and v channel of HSV color space
def color_threshold(img, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel=hls[:,:,2]
    binary_s = np.zeros_like(s_channel)
    binary_s[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel=hsv[:,:,2]
    binary_v = np.zeros_like(v_channel)
    binary_v[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    binary_output = np.zeros_like(s_channel)
    binary_output[(binary_s == 1) & (binary_v == 1)] = 1
    return binary_output

#Not USed
#Applies Sobel thresholding on the x and y channel
def abs_sobel_thresh2(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient=='x':
        img_s = cv2.Sobel(img,cv2.CV_64F, 1, 0)
    else:
        img_s = cv2.Sobel(img,cv2.CV_64F, 0, 1)
    img_abs = np.absolute(img_s)
    img_sobel = np.uint8(255*img_abs/np.max(img_abs))

    binary_output = 0*img_sobel
    binary_output[(img_sobel >= thresh[0]) & (img_sobel <= thresh[1])] = 1
    return binary_output

#Not USed
#Applies sobel thresholding on the HLS color scheme
def apply_sobel_on_HLS(img):
    # Convert image to HLS scheme
    image_HLS = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

    # Apply sobel filters on L and S channels.
    img_gs = image_HLS[:,:,1]
    img_abs_x = abs_sobel_thresh2(img_gs,'x',5,(50,225))
    img_abs_y = abs_sobel_thresh2(img_gs,'y',5,(50,225))
    wraped2 = np.copy(cv2.bitwise_or(img_abs_x,img_abs_y))

    img_gs = image_HLS[:,:,2]
    img_abs_x = abs_sobel_thresh2(img_gs,'x',5,(50,255))
    img_abs_y = abs_sobel_thresh2(img_gs,'y',5,(50,255))
    wraped3 = np.copy(cv2.bitwise_or(img_abs_x,img_abs_y))

    # Combine sobel filter information from L and S channels.
    return cv2.bitwise_or(wraped2,wraped3)

#Not Used
#Computes points that satisfies the yellow and white color in an image using HSV color scheme
def color_mask(img):
    image_HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Define color ranges and apply color mask
    yellow_hsv_low  = np.array([ 0, 100, 100])
    yellow_hsv_high = np.array([ 50, 255, 255])

    white_hsv_low  = np.array([  20,   0,   180])
    white_hsv_high = np.array([ 255,  80, 255])

    # get yellow and white masks 
    # Takes in low and high values and returns mask
    mask_yellow = cv2.inRange(image_HSV,yellow_hsv_low,yellow_hsv_high)
    mask_white = cv2.inRange(image_HSV,white_hsv_low,white_hsv_high)
    return cv2.bitwise_or(mask_yellow,mask_white)


#Computes the curvature of the road
def get_curvature(pol_a,y_pt):
    # Returns curvature of a quadratic
    A = pol_a[0]
    B = pol_a[1]
    R_curve = (1+(2*A*y_pt+B)**2)**1.5/2/A
    return R_curve

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


#Computes the points that lies on the left and right road lanes.
def draw_road_bounding_box(binary_mask, window_centroids, window_width, window_height, leftx, rightx):
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(binary_mask)
    r_points = np.zeros_like(binary_mask)

    # Go through each level and draw the windows
    for level in range(0,len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width, window_height, binary_mask, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, binary_mask, window_centroids[level][1], level)
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((binary_mask, binary_mask, binary_mask)),np.uint8) # making the original road pixels 3 color channels
    return cv2.addWeighted(warpage, 0.5, template, 0.5, 0.0) # overlay the orignal road image with window results


#Fits a second order polynomial to points on the rad lanes.
#Also, draws the road lines on the image
def road_lines(img, warped, win_ht, win_wd, res_yvals1, leftx, rightx, left_fitx1, right_fitx1):
    yvals = range(win_ht, warped.shape[0])
    res_yvals = np.arange(warped.shape[0]-win_ht/2, 0, -win_ht)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)


    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)
	
    left_fitx1.extend(left_fitx)
    right_fitx1.extend(right_fitx)
    res_yvals1.extend(res_yvals)

    left_lane = np.array(list(zip(np.concatenate((left_fitx-win_wd/2, left_fitx[::-1]+win_wd/2), axis=0), np.concatenate((yvals,yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-win_wd/2, right_fitx[::-1]+win_wd/2), axis=0), np.concatenate((yvals,yvals[::-1]), axis=0))), np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx+win_wd/2, right_fitx[::-1]-win_wd/2), axis=0), np.concatenate((yvals,yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [inner_lane], color=[160, 82, 45])
    return road


#Function that does actual computation
def display_road_lines(img):
    img_size = (img.shape[1], img.shape[0])

    # Function to undistort image
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    
    #first apply perspective transform and then apply thresholding
    # Apply perspective transform
    src = np.float32([[(250, 690), (1060, 690), (684, 450), (595, 450)]])     # define source points
    dst = np.float32([[(350, 690), (950, 690), (950, 0), (350, 0)]])     # define dst points
    M = cv2.getPerspectiveTransform(src, dst)
    warped1 = cv2.warpPerspective(undist_img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #Preprocess Image to Apply color, sobel Thresholding and generate binary image
    sv_color_mask = color_threshold(warped1, sthresh=(100,255), vthresh=(50,255))
    gradx = abs_sobel_thresh(warped1, orient='x', sobel_kernel=3, thresh=(12, 100))
    binary_mask1 = np.zeros_like(img[:,:,0])
    binary_mask1[ (gradx==1) | (sv_color_mask==1)] = 255

    #first apply thresholding and then apply perspective transform
    #Preprocess Image to Apply color, sobel Thresholding and generate binary image
    sv_color_mask = color_threshold(undist_img, sthresh=(100,255), vthresh=(50,255))
    gradx = abs_sobel_thresh(undist_img, orient='x', sobel_kernel=3, thresh=(12, 100))
    binary_mask = np.zeros_like(img[:,:,0])
    binary_mask[ (gradx==1) | (sv_color_mask==1)] = 255
    # Apply perspective transform
    warped = cv2.warpPerspective(binary_mask, M, img_size, flags=cv2.INTER_LINEAR)
    warped = warped/4 + binary_mask1


    window_centroids = curve_center.find_window_centroids(warped)
    rightx = []
    leftx = []
    res_yvals = []
    left_fitx=[]
    right_fitx=[]
    output = draw_road_bounding_box(warped, window_centroids, window_width, window_height, leftx, rightx)
    road = road_lines(img, warped, window_height, window_width, res_yvals, leftx, rightx, left_fitx, right_fitx)
    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)

    #calculate offset of car on road and print it
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center - warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'
    vehicle_offset = 'vehicle is ' + str(abs(round(center_diff,3))) + 'm ' + side_pos + ' of center'
    cv2.putText(road_warped, vehicle_offset , (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	
	#calculate curvature of road and print it
    left_fit = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx, np.float32)*xm_per_pix, 2)
    right_fit = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(rightx, np.float32)*xm_per_pix, 2)
    left_curve = get_curvature(left_fit,img_size[0]/2*ym_per_pix)
    Right_curve = get_curvature(right_fit,img_size[0]/2*ym_per_pix)
    str_curv = 'Curvature: Right = ' + str(np.round(Right_curve,2)) + 'm, Left = ' + str(np.round(left_curve,2)) +'m'
    cv2.putText(road_warped, str_curv , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


    alpha = 0.6
    result = cv2.addWeighted(undist_img, alpha, road_warped, 1-alpha, 0)
    return result

# window settings
window_width = 50
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 50 # How much to slide left and right for searching
ym_per_pix = 30/720
xm_per_pix = 3.7/700
curve_center = tracker(window_height, window_width, margin, ym_per_pix, xm_per_pix, smooth_factor=15)		
alpha = 0.6

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
project_output = 'project_video_output.mp4'

  
clip1 = VideoFileClip("project_video.mp4");
white_clip = clip1.fl_image(display_road_lines) #NOTE: this function expects color images!!
white_clip.write_videofile(project_output, audio=False);
	
