
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        output = np.zeros(img.shape, np.float32)
        y, x = np.mgrid[-self.pad_w:self.pad_w+1, -self.pad_w:self.pad_w+1]
        g_spatial = np.exp(-(x**2 + y**2) / (2 * self.sigma_s**2))
        print(x)
        print("y:")
        print(y)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # Extract local regions
                i_padded, j_padded = i + self.pad_w, j + self.pad_w
                patch_img = padded_img[i_padded-self.pad_w:i_padded+self.pad_w+1, j_padded-self.pad_w:j_padded+self.pad_w+1]
                patch_guidance = padded_guidance[i_padded-self.pad_w:i_padded+self.pad_w+1, j_padded-self.pad_w:j_padded+self.pad_w+1]

                # Compute Gaussian range weights
                d_intensity = patch_guidance - padded_guidance[i_padded, j_padded]
                g_range = np.exp(-(d_intensity**2) / (2 * self.sigma_r**2))

                # Calculate the weights
                weights = g_spatial * g_range
                weighted_sum = np.sum(patch_img * weights)
                sum_weights = np.sum(weights)

                # Assign the computed value to the output image
                output[i, j] = weighted_sum / sum_weights



        return np.clip(output, 0, 255).astype(np.uint8)
