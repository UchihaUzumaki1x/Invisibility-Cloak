import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

time.sleep(3)
background=0

for i in range(60):
	ret,background = cap.read()

background = np.flip(background,axis=1)

while(cap.isOpened()):
	ret, img = cap.read()
	img = np.flip(img,axis=1)
	
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # lower mask (0-10)   H  S   V
	lower_red = np.array([0,120,70])
	upper_red = np.array([10,255,255])
	mask1 = cv2.inRange(hsv,lower_red,upper_red)
	#cv2.imshow('mask1', mask1)

        # upper mask (170-180) H   S   V
	lower_red = np.array([170,120,70])
	upper_red = np.array([180,255,255])
	mask2 = cv2.inRange(hsv,lower_red,upper_red)
	#cv2.imshow('mask2', mask2)

	mask1 = mask1+mask2
	kernel = np.ones((3,3), np.uint8)

	# Refining mask with the detected red color
	# removing false positive from the background
	mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel,iterations=2)
	#cv2.imshow('mask1',mask1)
	mask1 = cv2.dilate(mask1,kernel,iterations = 1)
	mask2 = cv2.bitwise_not(mask1)

	res1 = cv2.bitwise_and(background,background,mask=mask1)
	res2 = cv2.bitwise_and(img,img,mask=mask2)
	final_output = cv2.addWeighted(res1,1,res2,1,0)

	cv2.imshow('Output',final_output)
	#cv2.imshow('res1',res1)
	#cv2.imshow('res2',res2)
	#cv2.imshow('mask1',mask1)
	#cv2.imshow('mask2',mask2)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
