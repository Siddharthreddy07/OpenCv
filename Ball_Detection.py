import cv2 as cv
import numpy as np
import imutils
greenLower = (20,70,80)
greenUpper = (38,255,255)
vid = cv.VideoCapture(0)
while (1):
	frame = vid.read()[1]
	blur = cv.GaussianBlur(frame,(11,11),10)
	hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
	mask = cv.inRange(hsv, greenLower, greenUpper)
	mask = cv.erode(mask, None, iterations=2)
	mask = cv.dilate(mask, None, iterations=2)
	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	if len(cnts) > 0:
		c = max(cnts, key=cv.contourArea)
		((x, y), radius) = cv.minEnclosingCircle(c)
		M = cv.moments(c)
		print(x,y)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		if radius > 30:
			cv.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv.circle(frame, center, 5, (0, 0, 255), -1)
	cv.imshow("Frame", frame)
	cv.imshow("mask", mask)
	key = cv.waitKey(1) & 0xFF
	if key == ord("q"):
		break
vid.release()
cv.destroyAllWindows()
