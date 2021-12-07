import os
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

  key = 0
  AirCond = "Unknown state"
  cap = cv2.VideoCapture(0, cv2.CAP_V4L)

  # Setup SimpleBlobDetector parameters.
  params = cv2.SimpleBlobDetector_Params()
  
  # Change thresholds
  params.minThreshold = 1
  params.maxThreshold = 255
  
  
  # Filter by Area.
  params.filterByArea = True
  params.minArea = 60
  
  # Filter by Circularity
  params.filterByCircularity = True
  params.minCircularity = 0.1
  
  # Filter by Convexity
  params.filterByConvexity = True
  params.minConvexity = 0.87
  
  # Filter by Inertia
  params.filterByInertia = True
  params.minInertiaRatio = 0.01
 
  while True :
    # Load image
    ret, image = cap.read()

    # Blur image to make it easier to detect objects
    blur_image = cv2.medianBlur(image, 3)

    # Convert to HSV in order to 
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

    # Get lower orange hue, from 2021-12-04
    lower_orange_hue = create_hue_mask(hsv_image, [0, 0, 255], [0, 255, 255])

    # Get higher orange hue
    higher_orange_hue = create_hue_mask(hsv_image, [0, 0, 255], [38, 255, 255])    

    # Merge the images
    full_image = cv2.addWeighted(lower_orange_hue, 1.0, higher_orange_hue, 1.0, 0.0)

    # Blur the final image to reduce noise from image
    full_image = cv2.GaussianBlur(full_image, (9, 9), 2, 2)

    h, s, image_gray = cv2.split(full_image)
    image_gray_neg = cv2.bitwise_not(image_gray) # black on white background is
                                              # needed for SimpleBlobDetection
  
 
    # Create a detector with the parameters
    # OLD: detector = cv2.SimpleBlobDetector(params)
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(image_gray_neg)

    if not keypoints :
      if AirCond != 'Off' :
        AirCond = 'Off'
        print(f"Study aircond is {AirCond}")
    else :
      if AirCond != 'On' :
        im_with_keypoints = image.copy()
        for marker in keypoints:
          im_with_keypoints = cv2.drawMarker(im_with_keypoints, tuple(int(i) for i in marker.pt), color=(0,255,0) )
          position = tuple(int(i) for i in marker.pt)
          print(f"{position}, x={position[0]}, y={position[1]}", end = " ")
          if (position[0]) > 300 and (position[1] > 100) : # filter off false alarms
            AirCond = 'On'
          else :
            pass
        else:
          print() 
        print(f"Study aircond is {AirCond}")
      # Show blobs
      cv2.imshow("Keypoints", im_with_keypoints)
    key = cv2.waitKey(1)
    if key == ord("q") :
      break

if __name__ == '__main__':
  main()
