import numpy as np
import cv2

class tracker():
    def __init__(self, winHt, winWd, margin, ym=1, xm=1, smooth_factor=15):
        self.recent_centres = []
        self.window_width = winWd
        #Break image into vertical layers. Layers=imageHeight/window_height
        self.window_height = winHt

        #How much to slide left and right for searching
        self.margin = margin

        #meters per pixel
        self.ym_per_pix = ym
        self.xm_per_pix = xm
        self.smooth_factor = smooth_factor

        #Variables that are used in the next frame to check the correctness of the next frame.
        self.avg_pixel_dist = 0
        self.compute_first = False
        self.l_center = 0
        self.r_center = 0


    def find_window_centroids(self, warped):

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
       # value = [0.75, 0.75]
       # l_cent=np.zeros(len(value))
       # r_cent=np.zeros(len(value))
       # average=np.zeros(len(value))
       # for i,ratio in enumerate(value):
       #     l_sum = np.sum(warped[int(ratio*warped.shape[0]):,:int(warped.shape[1]/2)], axis=0)
       #     l_cent[i] = np.argmax(np.convolve(window,l_sum))-window_width/2
       #     r_sum = np.sum(warped[int(ratio*warped.shape[0]):,int(warped.shape[1]/2):], axis=0)
       #     r_cent[i] = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
       #     average[i] = r_cent[i]-l_cent[i]
       # if (np.abs(average[0] - average[1])) > 50:
       #     l_center = l_cent[0]
       #     r_center = r_cent[0]
       #     avg_pixel_dist = average[0]
       # else:
       #     l_center = l_cent[1]
       #     r_center = r_cent[1]
       #     avg_pixel_dist = average[1]

        ratio = 0.75
        l_sum = np.sum(warped[int(ratio*warped.shape[0]):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(ratio*warped.shape[0]):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
        avg_pixel_dist = r_center-l_center


        pix_threshold = 40
        if self.compute_first & (abs(avg_pixel_dist - self.avg_pixel_dist) > pix_threshold):
            if (abs(l_center - self.l_center) < pix_threshold):
                self.l_center = l_center
            if (abs(r_center - self.r_center) < pix_threshold):
                self.r_center = r_center
        else:
            self.avg_pixel_dist = avg_pixel_dist
            self.l_center = l_center
            self.r_center = r_center

        if ~self.compute_first:
            self.compute_first = True

        threshold = 1000
        
        # Add what we found for the first layer
        window_centroids.append((self.l_center,self.r_center))

        level_count = (int)(warped.shape[0]/window_height)
        # Go through each layer looking for max pixel locations
        for level in range(1,level_count):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

            #Interpolate the current reading from the other lane, if the max value is less then threshold.
            if max(conv_signal[l_min_index:l_max_index]) < threshold:
                l_center = max(r_center - self.avg_pixel_dist, 0)
            if max(conv_signal[r_min_index:r_max_index]) < threshold:
                r_center = min(l_center + self.avg_pixel_dist, warped.shape[1])
            
            #interpolate the current reading from the other lane, if the current window is more than pix_threshold pixel apart from the previous window
            #We do this only for the bottom half of the image, as it handles the case where there is no lane line on one side, at the beginning of the frame.
            if (level < 0.8 * level_count):
                if abs(r_center - l_center - self.avg_pixel_dist) > pix_threshold:
                    if abs(l_center - window_centroids[level-1][0]) > pix_threshold:
                        l_center = max(r_center - self.avg_pixel_dist, 0)
                    if abs(r_center - window_centroids[level-1][1]) > pix_threshold:
                        r_center = min(l_center + self.avg_pixel_dist, warped.shape[1])
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))


        self.recent_centres.append(window_centroids)    
        return window_centroids
