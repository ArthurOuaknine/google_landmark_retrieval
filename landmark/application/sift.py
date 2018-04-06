"""Computing sift features """
import os
import numpy as np
import cv2
from landmark.infrastructure.landmark_dataset import Landmark

if __name__ == "__main__":
    landmark_home = os.environ["LANDMARK_HOME"]
    config_file = os.path.join(landmark_home, "config.ini")
    train_data = Landmark(config_file).train[:100]
    img = cv2.imread(train_data["path"][0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    img_kp = cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,outImage=np.array([]))
    cv2.imshow("test", img_kp)
