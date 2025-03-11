import cv2
import numpy as np
video=cv2.VideoCapture(0)
while True:
    isTrue, frame=video.read()
    cv2.imshow("video",frame)
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break
video.release()
cv2.destroyAllWindows()