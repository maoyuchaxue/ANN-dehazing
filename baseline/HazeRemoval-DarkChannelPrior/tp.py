
from functions import *
import os

def deHaze(imageRGB, fileName):
    # imageRGB = imageRGB / 255.0


    print('Getting Dark Channel Prior')
    darkChannel = getDarkChannel(imageRGB);
    cv2.imwrite('output/' + fileName + '_dark.jpg', darkChannel * 255.0)

    print('Getting Atmospheric Light')
    atmLight = getAtmLight(imageRGB, darkChannel);

    mattedTransmission = cv2.imread("./output/816_2_input.jpg_refinedTransmission.jpg", 0)
    # mattedTransmission = np.reshape(mattedTransmission, (600,600,3))

    print('Getting Scene Radiance')
    betterRadiance = getRadiance(atmLight, imageRGB, mattedTransmission / 255.0);
    cv2.imwrite('output/' + fileName + '_refinedRadiance.jpg', betterRadiance * 255.0)

    return betterRadiance

nms = ["./result/result/816_2_input.jpg"]

for nm in nms:
    hazed_img =  cv2.imread(nm)
    refined = deHaze(hazed_img, nm.replace("/result/result",""))
    # cv2.imwrite(nm.replace(".jpg", "_baseline.jpg"), refined)

