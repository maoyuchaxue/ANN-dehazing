from functions import *
import argparse
from skimage import transform

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
    imageRGB = imageRGB / 255.0

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

original_image_f = open("../../data/original_images_mini.csv", "r")
hazed_image_f = open("../../data/hazed_images_mini.csv", "r")

psnrs = []
ssims = []
uqis = []

i = 0
for original_l in original_image_f:
    original_arr = [int(t) for t in original_l.split(",")]
    original_img = np.reshape(np.array(original_arr[0:]), (480, 640, 3)) / 255.0
    hazed_ind = i
    i += 1
    # cv2.imwrite("output/" + str(original_ind) + ".jpg", original_img)

    for j in range(5):

        hazed_l = hazed_image_f.readline()
        hazed_arr = [int(t) for t in hazed_l.split(",")]

        hazed_sub_ind = j
        hazed_img = np.reshape(np.array(hazed_arr[0:]), (480, 640, 3))
        
        # print hazed_ind, hazed_sub_ind, original_img.shape, hazed_img.shape
        rad, refined = deHaze(hazed_img, str(hazed_ind)+"_"+str(hazed_sub_ind))

        psnrs.append(PSNR(original_img, refined))
        ssims.append(SSIM(original_img, refined))
        uqis.append(UQI(original_img, refined))
        print(psnrs, ssims, uqis)


original_image_f.close()
hazed_image_f.close()



res_f = open("./baseline_res.csv", "w")

for i in range(len(psnrs)):
    res_f.write(str(psnrs[i]) + "," + str(ssims[i]) + "," + str(uqis[i]) + "\n")

res_f.close()

