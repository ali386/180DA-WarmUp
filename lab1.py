#REFERENCES
#Webcam video: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
#Changing color spaces and object tracking: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#Drawing contour: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

#IMPROVEMENTS
#Some improvements I made to the baseline code included denoising the mask a little by only considering contours found that met a certain size. This was to make sure that I was tracking the actual object as opposed to random noise contours. One factor to consider is that in order to create a contour of a certain size, the target object must actually be close enough to the camera in order to be recognized as the target and not just random noise. Another improvement I made was playing around with the lower and upper threshold colors as a way to narrow down the actual target color and not random noise in the background.

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_thresh = np.array([110, 50, 50]) #light blue
    upper_thresh = np.array([130, 255, 255]) #dark blue
   
    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in contours:
        #get rid of noise first by calculating area
        area = cv2.contourArea(i)
        if area > 5000:
            #cv2.drawContours(frame, [i], -1, (0, 255, 0), 2)
            x, y, width, height = cv2.boundingRect(i)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

    # show the image edges on the newly created image window
    cv2.imshow('frame with detection', frame)

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
