import numpy as np
import os
import cv2
import random

class DataSet(object):
    def __init__(self, data_dir, batch_size, is_test, max_size=-1):
        self.DATA_SIZE = 224
        self.data_dir = data_dir
        self.is_test = is_test
        self.batch_size = batch_size
        self.max_size = max_size
        self.gen_image_list()
        self.shuffle_data()
        self.cur_index = 0

    def gen_image_list(self):
        self.image_list = []
        self.total_images = 0
        for file_name in os.listdir(self.data_dir):
            if (file_name.endswith("jpg")):
                self.image_list.append(file_name)
                self.total_images += 1
                if (self.max_size > 0 and self.total_images >= self.max_size):
                    break

        self.total_batches = (self.max_size // self.batch_size) if self.max_size > 0 else (self.total_images // self.batch_size)

    def read_image(self, image_name):
        image_path = os.path.join(self.data_dir, image_name)
        img_data = cv2.imread(image_path)
        height, weight, channels = img_data.shape

        if (self.is_test):
            # no t(x) is provided for test set

            dif = (weight//2 - height) // 2
            original_img_large = img_data[0:height, dif:(weight//2-dif), 0:channels]
            hazed_img_large = img_data[0:height, (weight//2+dif):(weight-dif), 0:channels]
            # print original_img_large.shape, hazed_img_large.shape

            original_img = cv2.resize(original_img_large, (self.DATA_SIZE, self.DATA_SIZE)) / 255.0
            hazed_img = cv2.resize(hazed_img_large, (self.DATA_SIZE, self.DATA_SIZE)) / 255.0
            original_img = original_img * 2 - 1
            hazed_img = hazed_img * 2 - 1

            return original_img, hazed_img

        else:
            
            dif = (weight//3 - height) // 2
            original_img_large = img_data[0:height, dif:(weight//3-dif), 0:channels]
            hazed_img_large = img_data[0:height, (weight//3+dif):(2*weight//3-dif), 0:channels]
            tx_large = img_data[0:height, (2*weight//3+dif):(weight-dif), 0:channels]
            # print original_img_large.shape, hazed_img_large.shape

            original_img = cv2.resize(original_img_large, (self.DATA_SIZE, self.DATA_SIZE)) / 255.0
            hazed_img = cv2.resize(hazed_img_large, (self.DATA_SIZE, self.DATA_SIZE)) / 255.0
            tx = cv2.resize(tx_large, (self.DATA_SIZE, self.DATA_SIZE)) / 255.0
            tx = np.reshape(tx[0:self.DATA_SIZE,0:self.DATA_SIZE,0], (self.DATA_SIZE, self.DATA_SIZE, 1)) 

            original_img = original_img * 2 - 1
            hazed_img = hazed_img * 2 - 1

            return original_img, hazed_img, tx

    def shuffle_data(self):
        random.shuffle(self.image_list)

    def next_batch(self):
        self.end_index = min([self.cur_index + self.batch_size, self.total_images])
        array_hazed_img = []
        array_original_img = []
        array_tx = []
        for i in range(self.cur_index, self.end_index):
            # print(self.image_list[i])
            if (self.is_test):
                original_img, hazed_img = self.read_image(self.image_list[i])
            else:
                original_img, hazed_img, tx = self.read_image(self.image_list[i])
                array_tx.append(tx)

            array_hazed_img.append(hazed_img)
            array_original_img.append(original_img)

        self.cur_index += self.batch_size
        if (self.cur_index >= self.total_images):
            self.cur_index = 0
            self.shuffle_data()
            
        if (self.is_test):
            return np.array(array_hazed_img), np.array(array_original_img)
        else:        
            return np.array(array_hazed_img), np.array(array_original_img), np.array(array_tx)

if __name__ == "__main__":
    dataset = DataSet("../data/trainset/", 2, False)

    for i in range(10):
        hazed_img, original_img, tx = dataset.next_batch()
        print(hazed_img.shape, original_img.shape, tx.shape)
        cv2.imshow("test", np.reshape(hazed_img[0,0:dataset.DATA_SIZE,0:dataset.DATA_SIZE,0:3], (224,224,3)))
        cv2.waitKey(0)
        cv2.imshow("test", np.reshape(original_img[0,0:dataset.DATA_SIZE,0:dataset.DATA_SIZE,0:3], (224,224,3)))
        cv2.waitKey(0)
        cv2.imshow("test", np.reshape(tx[0,0:dataset.DATA_SIZE,0:dataset.DATA_SIZE], (224,224)))
        cv2.waitKey(0)
