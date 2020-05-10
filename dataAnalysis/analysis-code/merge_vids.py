import cv2
import numpy as np
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('move.mp4')
cap2 = cv2.VideoCapture('moveFeat.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 25
filename = 'moveStacked.avi'

if (cap.isOpened() == False):
    print("Error opening video stream or file")
if (cap2.isOpened() == False):
    print("Error opening video stream or file")

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL) 
idx = 0
while(cap.isOpened() | cap2.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret, frame1 = cap2.read()
    if idx == 0:
        w = 800
        h = 1200
        fid = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        idx = 1
    if ret == True:
        newFrame = cv2.resize(frame, (800, 600))
        newFrame1 = cv2.resize(frame1, (800, 600))
        stackedFrame = np.vstack((newFrame, newFrame1))
        fid.write(stackedFrame)
        #  cv2.imshow('Preview', stackedFrame)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break

cap.release()
cap2.release()
fid.release()
cv2.destroyAllWindows()
