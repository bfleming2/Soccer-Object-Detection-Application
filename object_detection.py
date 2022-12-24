# Import necessary packages
import cv2
import os
import numpy as np

#Setting Different Coloring Ranges

#yellow Range
yellow_low = np.array([5, 100, 100])
yellow_high = np.array([40, 255, 255])

#black range
black_low = np.array([110,50,50])
black_high = np.array([130,255,255])
#green range
green_low = np.array([40,40, 40])
green_high = np.array([70, 255, 255])
#blue range
blue_low = np.array([110,50,50])
blue_high = np.array([130,255,255])

#Red range
lower_red = np.array([0,31,255])
upper_red = np.array([176,255,255])

#white range
lower_white = np.array([0,0,200])
upper_white = np.array([0,0,255])

# #white range
# lower_white = np.array([200,200,200])
# upper_white = np.array([255,255,255])

#Reading the video
capture = cv2.VideoCapture('input_video.mp4')
temp_success,im = capture.read()

# A playback speed of 10 means that it will be around 3x fast forward
# A playback speed of 1 means that it will be a little slower than 1x speed
# I've felt that a playbackspeed of 5 was a good medium
playbackSpeed = 5
#Read the video frame by frame
while (capture.isOpened()):

    # Combining our image with a mask in the green range
	combined_image_hsv = cv2.bitwise_and(im, im, mask=cv2.inRange(cv2.cvtColor(im,cv2.COLOR_BGR2HSV), green_low, green_high))
	# This will convert from hsv to gray scale
	combined_image_gray = cv2.cvtColor(combined_image_hsv,cv2.COLOR_BGR2GRAY)

	# This will set the gray threshold
	current_threshhold = cv2.threshold(combined_image_gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	threshhold_morph = cv2.morphologyEx(current_threshhold, cv2.MORPH_CLOSE, (np.ones((13,13), np.uint8)))
	
    # Find contours in threshold image
	array_contour,temp = cv2.findContours(threshhold_morph,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	for current_contour in array_contour:
		#Finds the bounding rectange for the current contour
		(x,y,width,height) = cv2.boundingRect(current_contour)
		
		#This is the Ball Detection Code

		#I tried height = 30 and width = 30 it does look a little better
		if((height >= 1 and width >= 1) and (height <= 20 and width <= 20)):
			#Finds the part of the image containing the ball
			ball_img = im[y: y + height,x : x + width]
			ball_hsv = cv2.cvtColor(ball_img,cv2.COLOR_BGR2HSV)

			#Set the mask to the white range for the white ball
			ball_combined_image = cv2.bitwise_and(ball_img, ball_img, mask=cv2.inRange(ball_hsv, lower_white, upper_white))
			ball_combined_image = cv2.cvtColor(ball_combined_image,cv2.COLOR_HSV2BGR)
			ball_combined_image = cv2.cvtColor(ball_combined_image,cv2.COLOR_BGR2GRAY)
	
			if(cv2.countNonZero(ball_combined_image) >= 0.25):
				#This will detect the soccer ball
				cv2.putText(im, 'soccer ball', (x-2, y-2), cv2.FONT_ITALIC, 0.8, (0,255,0), 2, cv2.LINE_AA)
				cv2.rectangle(im,(x,y),(x + width,y + height),(0,255,0),2)


		# This is the Player Detection code

		if(width > 19 and height >= 20) and (height >= (1.6) * width):
			# Finds the part of the image containing the player being identified
			current_player_image = im[y: y + height,x: x + width]
			current_player_hsv = cv2.cvtColor(current_player_image,cv2.COLOR_BGR2HSV)
			
			# One could repeat this step for all of the color ranges to generalize to every soccer game
			# If you were to do that simply cp[y every set of three conversion lines (line 91-93) and the if else statements (lnes 96-98).
			# Simply change the lower_red and upper_red or blue_low and blue_high to the color jersey you want to have it check for

			# Identifies the player with a blue jersey
			player_combined_image_hsv = cv2.bitwise_and(current_player_image, current_player_image, mask = cv2.inRange(current_player_hsv, blue_low, blue_high))
			player_combined_image_rgb = cv2.cvtColor(player_combined_image_hsv,cv2.COLOR_HSV2BGR)
			player_combined_image_gray = cv2.cvtColor(player_combined_image_rgb,cv2.COLOR_BGR2GRAY)
			
			# Identifies the player with a red jersey
			player2_combined_image_hsv = cv2.bitwise_and(current_player_image, current_player_image, mask = cv2.inRange(current_player_hsv, lower_red, upper_red))
			player2_combined_image_rgb = cv2.cvtColor(player2_combined_image_hsv,cv2.COLOR_HSV2BGR)
			player2_combined_image_gray = cv2.cvtColor(player2_combined_image_rgb,cv2.COLOR_BGR2GRAY)
			
			# Counts the non zero entries from the red jersey team and identifies
			# May change to 15 later it does seem better
			if(cv2.countNonZero(player2_combined_image_gray) >= 15):
				# Putting a rectangle on the red jersey players
				# The tuple (85,0,130) changes the color of the rectangle and the number 2 is the thickness of the rectangle
				# im is the image it is placing the rectangle on and (x,y) is the lower left hand corner of the rectangle
				# and the upper right hand corner of the array is x+width, y + height
				cv2.rectangle(im,(x,y),(x + width,y + height),(85,0,130), 2)
			else:
				pass
			
			# Counts the non zero entries from the blue jersey team and identifies
			# May change to 15 later it does seem better
			if(cv2.countNonZero(player_combined_image_gray) >= 20):
				# Putting a rectangle on the blue jersey players
				cv2.rectangle(im,(x,y),(x + width,y + height),(110,50,20), 2)
			else:
				pass
	cv2.imshow('Soccer',im)
    
	# Press c on your keyboard to break out of the video
    # Change the speed of the outputted video by changing the value in waitKey
    # If playbackSpeed == 1 it will be around 4x speed and 10 is around 1x speed
    
	if cv2.waitKey(playbackSpeed) & 0xFF == ord('c'):
		break
	temp_success,im = capture.read()
    
capture.release()
cv2.destroyAllWindows()

