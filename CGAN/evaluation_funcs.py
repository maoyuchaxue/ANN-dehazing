import numpy as np
import skimage

# used to evaluate the result of the model.
def PSNR(origin, recovered):
    return skimage.measure.compare_psnr(origin, recovered)

def SSIM(origin, recovered):
    return skimage.measure.compare_ssim(origin, recovered, multichannel=True)

def UQI(origin, recovered):
    ori = origin.flatten()
    rec = recovered.flatten()

    mean_ori = np.mean(ori)
    mean_rec = np.mean(rec)

    std_ori = np.std(ori, ddof=1)
    std_rec = np.std(rec, ddof=1)

    cov = np.cov(ori, rec, ddof=1)

    return 4 * cov[0][1] * mean_ori * mean_rec / ((std_ori*std_ori + std_rec*std_rec) * (mean_ori**2 + mean_rec**2))

def test_image(origin, recovered, batch_size):
    PSNRs = []
    SSIMs = []
    UQIs = []
    for i in range(batch_size):
        origin_img = (origin[i] + 1) / 2
        recovered_img = (recovered[i] + 1) / 2
        recovered_img = recovered_img.astype(np.float64)
        print(origin_img.dtype, recovered_img.dtype)
        PSNRs.append(PSNR(origin_img, recovered_img))
        SSIMs.append(SSIM(origin_img, recovered_img))
        UQIs.append(UQI(origin_img, recovered_img))
        
    return PSNRs, SSIMs, UQIs
