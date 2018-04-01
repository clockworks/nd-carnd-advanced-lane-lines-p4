import numpy as np
import cv2

class tracker():
    
    #when starting a new instance please be sure to specify all unassigned variables
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym = 1, My_xm =1, Mysmooth_factor = 15):
        #list that stores all the past (left, right) center set values used for smoothing the output
        self.recent_centers = []
        
        #the window pixel width of the center values, used to count pixels inside center windows to determine curve values 
        self.window_width = Mywindow_width
        
        #the window pixel height of the center values, used to count pixels inside center windows to determine curve values
        #break the image into vertical levels 
        self.window_height = Mywindow_height
        
        #the pixel distance in both directions to slide ( left_window + right_window ) template for searching
        self.margin = Mymargin
        
        self.ym_per_pix = My_ym # meter per pixel in vertical axis
        
        self.xm_per_pix = My_xm # meter per pixel in horizontal axis
        
        self.smooth_factor = Mysmooth_factor
        
    # the main tracking function for finding and storing lane segment positions 
    def find_window_centroids(self, warped):
        
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        
        window_centroids = [] # store the (left, right) window centroid positions per level
        window = np.ones(window_width) # create our window template that we will use for convolutions
        
        # first find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template
        
        # sum quarter buttom of image to get slice, could use a different ratio 
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):, :int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2 #TODO
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):, int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2 + int(warped.shape[1]/2)
        
        #add what we found for the first layer 
        window_centroids.append((l_center, r_center))
        
        # go through the layer looking for max pixel locations 
        for level in range(1, (int)(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0) # TODO
            conv_signal = np.convolve(window, image_layer)
            
            # Find the best left centroids by using past left center as a reference
            # Use window_width / 2 as offset because convolotion signal reference at right side of window, not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            
            #Find the best right centroids by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin, 0))
            r_max_index = int(min(r_center+offset+margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            
            #add what we found for that layer
            window_centroids.append((l_center, r_center))
            
        self.recent_centers.append(window_centroids)
        
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)
        