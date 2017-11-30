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
depth_img = np.reshape(depth_img, [640 * 480])

img = np.reshape(img_img, [3, 640, 480])
img = np.transpose(img, axes=[2, 1, 0])
img = np.reshape(img, [480*640, 3])
img = img * (1.0 / 256)


import random

k = random.uniform(0.7, 1.0)
A = np.array([k,k,k])
beta = random.uniform(-1.5, -0.5)

tx = np.reshape(np.exp(depth_img * beta), (480*640, 1))

print A

img_out = tx * img + A * (1-tx)
# img_out = A * (1-tx)

img_out = np.reshape(img_out, (480, 640, 3))

cv2.imshow("imgout", img_out)


while True:
    cv2.waitKey(0)
