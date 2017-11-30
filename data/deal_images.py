depth_file = open("./depths.csv", "r")
t = depth_file.readline()
d1 = [float(d) for d in t.split(",")]

img_file = open("./images.csv", "r")
t2 = img_file.readline()
d2 = [int(d) for d in t2.split(",")]

depth_file.close();
img_file.close();

import cv2

import numpy as np

depth_img = np.array(d1)
img_img = np.array(d2, dtype=np.uint8)


mx = np.max(depth_img)
mi = np.min(depth_img)

print mx, mi

depth_img = (depth_img - mi) / (mx - mi)
depth_img = np.reshape(depth_img, [640, 480])
depth_img = np.transpose(depth_img)
cv2.imshow("test", depth_img)

img = np.reshape(img_img, [3, 640, 480])
img = np.transpose(img, axes=[2, 1, 0])
cv2.imshow("img", img)



while True:
    cv2.waitKey(0)
