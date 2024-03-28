
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
        output = np.zeros(img.shape, np.uint8)
        y, x = np.mgrid[-self.pad_w:self.pad_w+1, -self.pad_w:self.pad_w+1]
        print(x)
        print("y:")
        print(y)
        g_spatial = np.exp(-(x**2 + y**2) / (2 * self.sigma_s**2))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                
                i_padded, j_padded = i+self.pad_w , j+self.pad_w
                # patch_img = padded_img[i_padded-self.pad_w:i_padded+self.pad_w+1, j_padded-self.pad_w:j_padded+self.pad_w+1]
                patch_img_r = padded_img[i_padded-self.pad_w:i_padded+self.pad_w+1, j_padded-self.pad_w:j_padded+self.pad_w+1,0]
                patch_img_g = padded_img[i_padded-self.pad_w:i_padded+self.pad_w+1, j_padded-self.pad_w:j_padded+self.pad_w+1,1]
                patch_img_b = padded_img[i_padded-self.pad_w:i_padded+self.pad_w+1, j_padded-self.pad_w:j_padded+self.pad_w+1,2]
                
                if padded_guidance.ndim == 3: # rgb
                # Extract local regions
                    patch_guidance_r = padded_guidance[i_padded-self.pad_w:i_padded+self.pad_w+1, j_padded-self.pad_w:j_padded+self.pad_w+1,0]
                    patch_guidance_g = padded_guidance[i_padded-self.pad_w:i_padded+self.pad_w+1, j_padded-self.pad_w:j_padded+self.pad_w+1,1]
                    patch_guidance_b = padded_guidance[i_padded-self.pad_w:i_padded+self.pad_w+1, j_padded-self.pad_w:j_padded+self.pad_w+1,2]
                    d_intensity_r = patch_guidance_r - padded_guidance[i_padded, j_padded,0]
                    d_intensity_g = patch_guidance_g - padded_guidance[i_padded, j_padded,1]
                    d_intensity_b = patch_guidance_b - padded_guidance[i_padded, j_padded,2]
                    g_range = np.exp(-(( d_intensity_r/255) **2 + ( d_intensity_g/255) **2 + ( d_intensity_b/255) **2 ) / (2 * self.sigma_r**2))
                    
                    
                else:
                    patch_guidance = padded_guidance[i_padded-self.pad_w:i_padded+self.pad_w+1, j_padded-self.pad_w:j_padded+self.pad_w+1]
                    d_intensity = patch_guidance - padded_guidance[i_padded, j_padded]
                    g_range = np.exp(-(( d_intensity/255) **2) / (2 * self.sigma_r**2))
                
                # Calculate the weights
                # for k in range(3):
                weights = g_spatial * g_range
                weighted_sum_r = np.sum(patch_img_r * weights)
                weighted_sum_g = np.sum(patch_img_g * weights)
                weighted_sum_b = np.sum(patch_img_b * weights)
                sum_weights = np.sum(weights)

                    # Assign the computed value to the output image
                output[i, j, 0] = weighted_sum_r / sum_weights
                output[i, j, 1] = weighted_sum_g / sum_weights
                output[i, j, 2] = weighted_sum_b / sum_weights
                
                
                    
                
                



        return np.clip(output, 0, 255).astype(np.uint8)
