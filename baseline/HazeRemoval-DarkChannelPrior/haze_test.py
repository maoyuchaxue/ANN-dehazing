from functions import *
import argparse
from skimage import transform
import os
parser = argparse.ArgumentParser(description='Remove image haze using dark channel prior method')
parser.add_argument('-s','--scale', action="store", dest="scale", default=1, type=float, help="Scaling factor for images")
parser.add_argument('-f','--folder', action="store", dest="folder", default='beijing1', help="folder name")
parser.add_argument('-n','--file', action="store", dest="file", default='IMG_8763', help="file name")

args = parser.parse_args()

scalingFactor = args.scale
folder = args.folder
fileName = args.file

def deHaze(imageRGB, fileName):
    # cv2.imwrite('output/' + fileName + '_imageRGB.jpg', imageRGB)
    # imageRGB = imageRGB / 255.0

    print('Getting Dark Channel Prior')
    darkChannel = getDarkChannel(imageRGB);
    # cv2.imwrite('output/' + fileName + '_dark.jpg', darkChannel * 255.0)

    print('Getting Atmospheric Light')
    atmLight = getAtmLight(imageRGB, darkChannel);

    print('Getting Transmission')
    transmission = getTransmission(imageRGB, atmLight);
    # cv2.imwrite('output/' + fileName + '_transmission.jpg', transmission * 255.0)

    print('Getting Scene Radiance', transmission.shape)
    radiance = getRadiance(atmLight, imageRGB, transmission);
    # cv2.imwrite('output/' + fileName + '_radiance.jpg', radiance * 255.0)

    print('Apply Soft Matting')
    mattedTransmission = performSoftMatting(imageRGB, transmission);
    # cv2.imwrite('output/' + fileName + '_refinedTransmission.jpg', mattedTransmission * 255.0)

    print('Getting Scene Radiance')
    betterRadiance = getRadiance(atmLight, imageRGB, mattedTransmission);
    # cv2.imwrite('output/' + fileName + '_refinedRadiance.jpg', betterRadiance * 255.0)

    return radiance, betterRadiance

hdr = "D:\\workspace\\tmp_workspace\\Middlebury_Hazy"
odr = "D:\\workspace\\tmp_workspace\\Middlebury_GT"


psnrs = []
ssims = []
uqis = []

for nm in os.listdir(hdr):
    if (not nm.endswith(".bmp")):
        continue
    print(nm)
    original_img = cv2.imread(os.path.join(hdr, nm))
    print original_img.shape
    (h, w, c) = original_img.shape

    hazed_img =  cv2.imread(os.path.join(odr, nm.replace("_Hazy.bmp","_im0.png")))

    if (h * 640.0 / 480 <= w):
        new_w = int(h * 640.0 / 480)
        dif_w = w - new_w
        print (w, dif_w, new_w)
        original_img = original_img[:, dif_w/2 : (dif_w/2 + new_w), :]
        hazed_img = hazed_img[:, dif_w/2 : (dif_w/2 + new_w), :]
    else:
        new_h = int(w / 640.0 * 480)
        dif_h = h - new_h
        print (h, dif_h, new_h)
        original_img = original_img[dif_h/2 : (dif_h/2 + new_h), :, :]
        hazed_img = hazed_img[dif_h/2 : (dif_h/2 + new_h), :, :]
        
    original_img = cv2.resize(original_img, (640, 480)) * 1.0 / 255
    hazed_img = cv2.resize(hazed_img, (640, 480)) * 1.0 / 255

    print(original_img[0][0], hazed_img[0][0])
    # original_img = original_img / 255.0
    # hazed_img =  hazed_img / 255.0

    cv2.imwrite("output/" + nm + ".jpg", original_img * 255)
    cv2.imwrite("output/" + nm + ".hazed.jpg", hazed_img * 255)

    # print hazed_ind, hazed_sub_ind, original_img.shape, hazed_img.shape
    rad, refined = deHaze(hazed_img, nm.replace("_Hazy.bmp",""))

    psnrs.append(PSNR(original_img, refined))
    ssims.append(SSIM(original_img, refined))
    uqis.append(UQI(original_img, refined))
    print(psnrs, ssims, uqis)

res_f = open("./baseline_test.csv", "w")

for i in range(len(psnrs)):
    res_f.write(str(psnrs[i]) + "," + str(ssims[i]) + "," + str(uqis[i]) + "\n")

res_f.close()

