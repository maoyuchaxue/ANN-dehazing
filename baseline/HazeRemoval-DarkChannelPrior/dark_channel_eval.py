import numpy as np
import os
import cv2
import random
from functions import *
from evaluation_funcs import *

from dataset import DataSet


def deHaze(imageRGB):
    # imageRGB = imageRGB / 255.0
    # cv2.imwrite('output/' + fileName + '_imageRGB.jpg', imageRGB)

    

    # print('Getting Dark Channel Prior')
    darkChannel = getDarkChannel(imageRGB);
    # cv2.imwrite('output/' + fileName + '_dark.jpg', darkChannel * 255.0)

    # print('Getting Atmospheric Light')
    atmLight = getAtmLight(imageRGB, darkChannel);

    # print('Getting Transmission')
    transmission = getTransmission(imageRGB, atmLight);
    # cv2.imwrite('output/' + fileName + '_transmission.jpg', transmission * 255.0)

    # print('Getting Scene Radiance', transmission.shape)
    radiance = getRadiance(atmLight, imageRGB, transmission);
    # cv2.imwrite('output/' + fileName + '_radiance.jpg', radiance * 255.0)

    # print('Apply Soft Matting')
    mattedTransmission = performSoftMatting(imageRGB, transmission);
    # cv2.imwrite('output/' + fileName + '_refinedTransmission.jpg', mattedTransmission * 255.0)

    # print('Getting Scene Radiance')
    betterRadiance = getRadiance(atmLight, imageRGB, mattedTransmission);
    # cv2.imwrite('output/' + fileName + '_refinedRadiance.jpg', betterRadiance * 255.0)

    return radiance, betterRadiance



def save_test_results(PSNRs, SSIMs, UQIs):
    print("Test result for dark channel")
    print("PSNR: mean {:f}, var {:f}".format(np.mean(PSNRs), np.std(PSNRs)))
    print("SSIM: mean {:f}, var {:f}".format(np.mean(SSIMs), np.std(SSIMs)))
    print("UQI: mean {:f}, var {:f}".format(np.mean(UQIs), np.std(UQIs)))

    test_result_log_file_name = "./result/metrics.csv" 
    log_file = open(test_result_log_file_name, "a")
    log_file.write("{:d},{:f},{:f},{:f},{:f},{:f},{:f}\n".format(epoch, np.mean(PSNRs), np.std(PSNRs),
        np.mean(SSIMs), np.std(SSIMs), np.mean(UQIs), np.std(UQIs)))
    log_file.close()

batch_size = 4
test_set = DataSet("../testset/", batch_size, True)

num_batches = test_set.total_batches

PSNRs = []
SSIMs = []
UQIs = []

log_res_file = "./result/log.csv"
# get test batch data
for idx in range(0, num_batches):
    batch_hazed_img, batch_ground_truth = test_set.next_batch()
    if (batch_hazed_img.shape[0] < batch_size):
        break
    for i in range(batch_size):
        rad, refined = deHaze(batch_hazed_img[i])
        tPSNR, tSSIM, tUQI = test_image(batch_ground_truth[i], refined)
        # print(PSNR, SSIM, UQI)
        
        log_f = open(log_res_file, "a")
        log_f.write("{:d},{:f},{:f},{:f}\n".format(idx, tPSNR, tSSIM, tUQI))
        log_f.close()

        PSNRs.append(tPSNR)
        SSIMs.append(tSSIM)
        UQIs.append(tUQI)
    
save_test_results(PSNRs, SSIMs, UQIs)
