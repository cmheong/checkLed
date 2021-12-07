import cv2
import sys

# total arguments
n = len(sys.argv)
print("Total arguments passed:", n)

# Arguments passed
print("\nName of Python script:", sys.argv[0])

print("\nArguments passed:", end = " ")
for i in range(1, n):
    print(sys.argv[i], end = " ")
print("")

if n >= 2 :
  cap = cv2.VideoCapture(0, cv2.CAP_V4L)
  ret, image = cap.read()
  blur_image = cv2.medianBlur(image, 3)
  hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
  cv2.imwrite(sys.argv[1], image)
  cv2.imwrite('hsv_'+sys.argv[1], hsv_image)
