import numpy as np
import random
import cv2

def generate_dataset(depth_data, img_data):
    depth_img = np.array(depth_data)
    img_img = np.array(img_data, dtype=np.uint8)

    depth_img = np.reshape(depth_img, [640, 480])
    depth_img = np.transpose(depth_img)
    depth_img = np.reshape(depth_img, [640 * 480])

    img = np.reshape(img_img, [3, 640, 480])
    img = np.transpose(img, axes=[2, 1, 0])
    img = np.reshape(img, [480*640, 3])

    k = random.uniform(0.6, 0.8) * 255
    A = np.array([k,k,k])
    beta = random.uniform(-1.0, -0.5) / 4

    tx = np.reshape(np.exp(depth_img * beta), (480*640, 1))

    img_out = tx * img + A * (1-tx)

    orig_img = np.reshape(img, (480, 640, 3))
    img_out = np.reshape(img_out, (480*640*3))

    img_out_show = np.reshape(img_out, (480, 640, 3))
    tx_out = np.reshape(tx, (480, 640, 1)) * 255.0
    tx_out = tx_out.repeat(3, axis=2)
    # cv2.imshow("img", img_out_show / 255)
    
    return img_out_show, orig_img, tx_out 

depth_file = open("./depths.csv", "r")
img_file = open("./images.csv", "r")

# hazed_img_file = open("./hazed_images.csv", "w")
# original_img_file = open("./original_images.csv", "w")

# TOT_DATA = 2248
# REPEAT_TIMES = 5;

TOT_DATA = 2248
REPEAT_TIMES = 2

tot = 0
for i in range(TOT_DATA):
    
    t = depth_file.readline()
    d1 = [float(d) for d in t.split(",")]
    t2 = img_file.readline()
    d2 = [int(d) for d in t2.split(",")]

    for j in range(REPEAT_TIMES):
        l1, l2, l3 = generate_dataset(d1, d2)

        # hazed_img_file.write(str(i) + "," + str(j) + "," + l1 + "\n")
        
        cv2.imwrite("./trainset/" + str(i) + "_" + str(j) + ".jpg", np.concatenate([l2, l1, l3], axis=1))

        # if (j == 0):
            # original_img_file.write(str(i) + "," + l2 + "\n")

        tot += 1
        print(tot)
    
depth_file.close()
img_file.close()
