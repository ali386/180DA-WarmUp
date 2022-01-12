#REFERENCES
#Webcam video: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
#K-Means: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
#Drawing contours: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

#IMPROVEMENTS
#Some improvements I made included finding the dominant color for a continuous video shot using a laptop webcam whereas the original code only printed out the dominant color graphic for a single image. 
#However, in order to improve the speed of the program, only a 100x100 pixel box is processed to determine the dominant color. The box is also printed on the video feed in order to show the user where it is taking the color input from.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    roi = img[310:410, 590:690]
    roi = roi.reshape((roi.shape[0] * roi.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(roi)

    plt.ion()
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    
    img = cv2.rectangle(img, (410, 310), (690, 590), (0, 255, 0), 3)
    cv2.imshow('webcam image', img)
    
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

