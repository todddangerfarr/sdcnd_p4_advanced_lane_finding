
**Advanced Lane Finding Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

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

[image1]: ./writeup_images/camera_calibration.png "Camera Calibration"
[image2]: ./writeup_images/undistorted_image.png "Road Undistorted"
[image3]: ./writeup_images/gradient_thresholding.png "Gradient Examples"
[image4]: ./writeup_images/color_channels.png "Test Images Color Channels"
[image5]: ./writeup_images/gradient_color_threshold.png "Binary Example"
[image6]: ./writeup_images/birdseye_warp.png "Warp Example"
[image7]: ./writeup_images/binary_warp_test_images.png "Binary Warped"
[image8]: ./writeup_images/sliding_window_search.png "Sliding Window Search"
[image9]: ./writeup_images/fit_from_previous.png "Fit From Previous Fit"
[image10]: ./writeup_images/lane_overlay.png "Lane Overlay"

[video9]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### In this writeup I'll consider each of the rubric points individually and describe how I addressed each of them in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.   

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is located in Section 1 of IPython notebook "./advanced_lane_line_detection.ipynb"

The first step of camera calibration is to create a list of real world "object points" for each individual chessboard corner, where each corner is represented by an (x, y, z) value. It is assumed that the chessboard is fixed on the (x, y) plane at z=0, such that "object points" remains a constant for each calibration image.  Therefore, the variable `objp` is just a replicated array of each possible combination of the coordinates (`chess_rows`, `chess_cols`, 0) that's appended to the list `objpoints` every time all chessboard corners are detected in a calibration image.  In parallel `imgpoints` will be appended with the (x, y) pixel locations in the image plane for each of the detected corners as well.  

Once the `objpoints` and `imgpoints` lists are populated with the respective object and image plane coordinates from all successful calibration images, the camera calibration matrix and distortion coefficients are calculated using the OpenCV `cv2.calibrateCamera()` function.  Then using the resulting matrix and distortion coefficients any image captured from that camera can be undistorted using the `cv2.undistort()` function.  The results for this process and code applied to the calibration images can be seen on the example below:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Now that the camera calibration matrix and distortion coefficients have been calculated for the given camera, any image from that camera can be corrected.  For this purpose the OpenCV function `cv2.undistort()` was wrapped in a function called `undistort_image`.  This function takes three arguments: `img` - the image to undistort, `mtx` - the associated camera calibration matrix and `dist` - the associated distortion coefficients.  The purpose for this is to simply just help to abstract the `cv2.undistort` function to one that's easier to remember.  The code for this function can be seen in Section 2.1 of the Ipython notebook.

The results of an undistorted camera image from the road using this function can be seen below:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for all thresholding can be seen in the IPython notebook in Sections 2.3, 2.4 & 2.5.

To create the thresholded binary image a combination of gradient thresholds and color transform thresholds were combined for the final solution.  For the gradient threshold it was an `&` combination of an absolute Sobel thresholding oriented in the x-direction from 0 - 255 and a directional Sobel thresholding from 0 - π/2 to specifically target vertical (lane) lines. An example of the tested gradients can be seen in the image below:

![alt text][image3]

For color thresholding both RGB and HLS colorspaces were used and more specifically the R, G, L and S channels in those colors spaces yielded the best results for having the most contrast for the lane lines even in varying lighting conditions as seen below:  

![alt text][image4]

After the best channels from these two colorspaces were identified for the varying road conditions a simple grayscale threshold was added to each channel with the following ranges:

| Channel  |  Lower Threshold  | Upper Threshold  |
|:--------:|:-----------------:|:----------------:|
| R        | 0                 | 150              |
| G        | 0                 | 150              |
| L        | 0                 | 100              |
| S        | 0                 | 120              |

Finally all thresholds were combined using the following logic:

`(R | G & L) & (S | gradients)`

The resulting thresholds and combination can be seen in the image below:

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective warping actually came before thresholding in my pipeline development and can be seen in Section 2.2 of the IPython Notebook.

There are 2 main functions associated with warping the image to the bird's eye perspective.  The first function `get_warp_pts` uses the size of and incoming image to return arrays of the source and destination points that will be used for Perspective Warping.  Ideally this function would be more dynamic to the incoming image but for this project most of the points where empirically hardcoded.  Both the source `src` and destination `dst` points consist of bottom left `bl`, bottom right `br`, top right `tr` and top left `tl` coordinates and the equations for these points can be seen below:

```python
x_offset = 440
bottom_offset = 170
top_offset = 220

src_bl = [img_width/2 - x_offset, img_height]
src_br = [img_width/2 + x_offset, img_height]
src_tr = [700, 460]
src_tl = [580, 460]
src = np.float32([src_bl, src_br, src_tr, src_tl])

dst_bl = [src_bl[0] + bottom_offset, img_height]
dst_br = [src_br[0] - bottom_offset, img_height]
dst_tr = [src_tr[0] + top_offset, 0]
dst_tl = [src_tl[0] - top_offset, 0]
dst = np.float32([dst_bl, dst_br, dst_tr, dst_tl])
```

This resulted in the following source and destination points:

| Location      | Source        | Destination   |
|:-------------:|:-------------:|:-------------:|
| bl            | 200, 720      | 370, 720      |
| br            | 1080, 720     | 910, 720      |
| tr            | 700, 460      | 920, 0        |
| tl            | 580, 460      | 360, 0        |

Once the source and destination points are calculated a Perspective Transformation is as simple as first getting the perspective transformation Matrix `M` using OpenCV's `cv2.getPerspectiveTransform()` function.  Once we have this matrix the warped image can be created using the `cv2.warpPerspective()` function.  I again wrapped both of these function in my own function `birdseye_warp` and the result on the test image can be seen below:

![alt text][image6]

Finally, just for the sake of being thorough I ran both the binary_thresholding and warping on all the test images to check for any obvious issues.

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are two ways that polynomials are fit to the lane lines for this project.  

The first method is a brute force sliding window search through the image.  In this method a histogram of the bottom half of the binary warped image is created and the max peaks on the left and right side of the image are considered the starting points of the lane lines and sliding window search.  Search windows of `margin` wide pixels and `img_height` / `n_windows` high pixels are centered on each of these peaks and all the nonzero pixels indices in these windows are recorded.  If the number of 'good' pixels captured in a window is greater than `minpix` the next search window in the vertical step direction is centered on the mean of these pixels from the previous window.  The code for the sliding window search can be seen in Section 2.7.3 in the IPython Code, and a visualization for the code can be seen below:

![alt text][image8]

The second method for finding lane lines is to fit from the previous polynomial fit.  This function assumes we have already fit using the brute force sliding window method in the previous frame and now we can use that fit to se a new search region of `margin` wide pixels centered on the previous polynomial line.  This makes sense because we'd expect only subtle differences in lane line polynomials frame to frame.  After the search window is created we again scan the area for all nonzero pixels and refit a 2nd order polynomial to each lane line.  The code for this method can be seen in Section 2.7.4 in the IPython Notebook and the resulting visualization can be seen below:

![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Both the functions for calculating radius of curvature and line position respect to center live in the Line Class found in Section 2.7.2 of the IPython Notebook.  

The radius of curvature calculation comes from information contained on this <a href="https://www.intmath.com/applications-differentiation/8-radius-curvature.php">website</a> and the resulting code can be seen below. 

```python
curve_rad = (((1 + (2*fit[0]*y_eval*meters_per_pix_y + fit[1])**2)**1.5) / np.absolute(2*fit[0]))

```

For the final lane position of the vehicle with respect to center both of the current individual off center values from each lane line are stored in the class instance and multipled by `meters_per_pix_x` which is 3.7/700.  It's important to convert the value from pixels to meters so we have a real world representation of the offset for display.  At the time of updating the vehicle position the right line's value is subtracted from the left line's value and divided by two to get the offset.  If the value is negative we can assume the car is positioned to the left, however if the value if positive we can assume the offset is towards the right of the lane.  

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

For overlaying the lane back down onto the road image, it's as easy as applying the inverse Perspective Transformation to the previously transformed image.  For this step, since my functions are already in wrapper functions this means just swapping the src and dst points from the `get_warp_pts` function in the input for the `birdseye_warp` function.

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](.output_videos/project_video_processed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of the problems faced during this project were due to the getting the appropriate thresholds setup to accurately isolate the lane lines in the varying conditions.  Even with the amount of effort put in it's still hardcoded and mostly empirical.  The pipeline is likely to break at this point because, although it works for the project video, it's not a good representation of all the lighting, road and lane conditions that are likely encountered in the real world.  In fact, on the first challenge video the pipeline does explode and has some really strange polynomial fitting on the left lane line before they both converge to the right side.  Unfortunately I don't have the time to really investigate what's happening here, but I hope to revisit this at a later time.  

Another challenge I faced with this project was the Line class.  I haven't had to do much OOP recently and I was pretty rusty in terms of how best to update the lines and what functions/information should be nested into this class.  

To make the pipeline more robust, I think my first step would be too look at some sort of dynamic thresholding function based on the overall brightness of the image.  I could also investigate some other colorspaces to see if I can further reduce the effects of changing conditions that are faced daily in real world driving.  Finally, as mentioned before in the challenge video the lane lines actually converge to both identify as the right lane line, I could look at adding some intelligence in regards to this issue keeping constraints on lines being roughly parallel fits and at a minimum distance apart.  
