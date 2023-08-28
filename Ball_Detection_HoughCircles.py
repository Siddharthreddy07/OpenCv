import cv2 as cv
import numpy as np
import imutils
greenLower = (20,70,80)
greenUpper = (38,255,255)
vid = cv.VideoCapture(0)
while(1):
	frame = vid.read()[1]
	blur = cv.GaussianBlur(frame,(11,11),0)
	hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
	mask = cv.inRange(hsv, greenLower, greenUpper)
	mask = cv.erode(mask, None, iterations=2)
	mask = cv.dilate(mask, None, iterations=2)
	blur1= cv.GaussianBlur(mask, (9, 9),10)
	res = cv.bitwise_and(frame, frame, mask=mask)
	circles = cv.HoughCircles(blur1.copy(),cv.HOUGH_GRADIENT,1,60,param1=60,param2=30,minRadius=10,maxRadius=100)
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for circles in circles[0, :]:
			x, y, r = circles[0], circles[1], circles[2]
			cv.circle(frame, (int(x), int(y)), int(r), (255, 25, 255), 3)
			cv.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
	cv.imshow("Frame", frame)
	cv.imshow("mask", mask)
	if cv.waitKey(1) == ord('q'):
		break
vid.release()
cv.destroyAllWindows