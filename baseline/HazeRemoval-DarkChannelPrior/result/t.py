from numpy import genfromtxt

import numpy as  np
d = genfromtxt("./log.csv", delimiter=',')

print d.shape
PSNRs = d[0:20, 1]
SSIMs = d[0:20, 2]
UQIs = d[0:20, 3]


print("PSNR: mean {:f}, var {:f}".format(np.mean(PSNRs), np.std(PSNRs)))
print("SSIM: mean {:f}, var {:f}".format(np.mean(SSIMs), np.std(SSIMs)))
print("UQI: mean {:f}, var {:f}".format(np.mean(UQIs), np.std(UQIs)))
