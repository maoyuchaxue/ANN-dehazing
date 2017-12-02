import numpy as np
import random

def generate_dataset(depth_data, img_data):
    depth_img = np.array(depth_data)
    img_img = np.array(img_data, dtype=np.uint8)

    depth_img = np.reshape(depth_img, [640, 480])
    depth_img = np.transpose(depth_img)
    depth_img = np.reshape(depth_img, [640 * 480])

    img = np.reshape(img_img, [3, 640, 480])
    img = np.transpose(img, axes=[2, 1, 0])
    img = np.reshape(img, [480*640, 3])

    k = random.uniform(0.6, 1.0) * 256
    A = np.array([k,k,k])
    beta = random.uniform(-1.5, -0.5)

    tx = np.reshape(np.exp(depth_img * beta), (480*640, 1))

    img_out = tx * img + A * (1-tx)

    orig_img = np.reshape(img, (480*640*3))
    img_out = np.reshape(img_out, (480*640*3))

    img_out_show = np.reshape(img_out, (480, 640, 3))
    
    return [",".join([str(int(k)) for k in img_out.tolist()]), ",".join([str(int(k)) for k in orig_img.tolist()])]

depth_file = open("./depths.csv", "r")
img_file = open("./images.csv", "r")
hazed_img_file = open("./hazed_images_mini.csv", "w")
original_img_file = open("./original_images_mini.csv", "w")

TOT_DATA = 10
REPEAT_TIMES = 5

tot = 0
for i in range(TOT_DATA):
    
    t = depth_file.readline()
    d1 = [float(d) for d in t.split(",")]
    t2 = img_file.readline()
    d2 = [int(d) for d in t2.split(",")]

    for j in range(REPEAT_TIMES):
        l1, l2 = generate_dataset(d1, d2)

        hazed_img_file.write(str(i) + "," + str(j) + "," + l1 + "\n")
        if (j == 0):
            original_img_file.write(str(i) + "," + l2 + "\n")

        tot += 1
        print(tot)
    
depth_file.close()
img_file.close()
hazed_img_file.close()
original_img_file.close()
