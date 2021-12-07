import os
import argparse
import numpy as np
import cv2

def create_hue_mask(image, lower_color, upper_color):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)
 
    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    output_image = cv2.bitwise_and(image, image, mask = mask)
    return output_image

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-i", "--image_path", dest = 'image_path', required = True)
    parser.add_argument("-o", "--output_dir", dest = 'output_dir', required = False)
    parser.add_argument("-s", "--show", dest = 'show_pic', required = False)
    
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image_path)

    # Blur image to make it easier to detect objects
    blur_image = cv2.medianBlur(image, 3)
    if args.output_dir:
        result_image_path = os.path.join(args.output_dir, "blur_image.png")
        cv2.imwrite(result_image_path, blur_image)
    ''' debug code for excluding blurring
    blur_image = image
    '''

    # Convert to HSV in order to 
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
    if args.output_dir:
        result_image_path = os.path.join(args.output_dir, "hsv_image.png")
        cv2.imwrite(result_image_path, hsv_image)

    # Get lower orange hue, from 2021-12-04
    lower_orange_hue = create_hue_mask(hsv_image, [0, 0, 255], [0, 255, 255])
    if args.output_dir:
        result_image_path = os.path.join(args.output_dir, "lower_orange_hue.png")
        cv2.imwrite(result_image_path, lower_orange_hue)

    # Get higher orange hue
    higher_orange_hue = create_hue_mask(hsv_image, [0, 0, 255], [38, 255, 255])    

    if args.output_dir:
        result_image_path = os.path.join(args.output_dir, "higher_orange_hue.png")
        cv2.imwrite(result_image_path, higher_orange_hue)

    # Merge the images
    full_image = cv2.addWeighted(lower_orange_hue, 1.0, higher_orange_hue, 1.0, 0.0)
    if args.output_dir:
        result_image_path = os.path.join(args.output_dir, "full_image.png")
        cv2.imwrite(result_image_path, full_image)

    # Blur the final image to reduce noise from image
    full_image = cv2.GaussianBlur(full_image, (9, 9), 2, 2)
    if args.output_dir:
        result_image_path = os.path.join(args.output_dir, "full_image_blur.png")
        cv2.imwrite(result_image_path, full_image)

    ''' note we are actually converting hsv to gray
    # Convert image to gray in order to find circles in the image
    image_gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    if args.output_dir:
        result_image_path = os.path.join(args.output_dir, "full_image_gray.png")
        cv2.imwrite(result_image_path, image_gray)
    '''

    h, s, image_gray = cv2.split(full_image)
    image_gray_neg = cv2.bitwise_not(image_gray) # black on white background is
                                                # needed for SimpleBlobDetection
    
    if args.output_dir:
        result_image_path = os.path.join(args.output_dir, "full_image_gray.png")
        cv2.imwrite(result_image_path, image_gray)
        result_image_path = os.path.join(args.output_dir, "full_image_gray_inv.png")
        cv2.imwrite(result_image_path, image_gray_neg)
    
    ''' 2021-12-05 this doesn't work and is replaced by cv2.SimpleBlobDetector()
    # Find circles in the image
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1.2, 17)

    # If we didn't find circles, the aircond status is "OFF"
    if circles is None:
        print("Study aircond is OFF")
        return

    # If we did find circles, the aircond is "ON"
    print("Study aircond is ON")

    if args.output_dir:
        # Draw the circles on the original image
        circles = np.round(circles[0, :]).astype("int")
        for (center_x, center_y, radius) in circles:
            cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 4)
        result_image_path = os.path.join(args.output_dir, "original_image_with_circles.png")
        cv2.imwrite(result_image_path, image)
    '''

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 255
    
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    
    # Create a detector with the parameters
    # OLD: detector = cv2.SimpleBlobDetector(params)
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(image_gray_neg)

    if not keypoints :
        print("Study aircond is OFF")
        return
    else :
        print("Study aircond is ON")
        im_with_keypoints = image.copy()
        for marker in keypoints:
            im_with_keypoints = cv2.drawMarker(im_with_keypoints, tuple(int(i) for i in marker.pt), color=(0,255,0) )
        if args.output_dir:
            result_image_path = os.path.join(args.output_dir, "original_image_with_blob.png")
            cv2.imwrite(result_image_path, im_with_keypoints)

         
        # Show blobs
        if args.show_pic :
            cv2.imshow("Keypoints", im_with_keypoints)
            cv2.waitKey(0)

if __name__ == '__main__':
    main()
