import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)

        gaussian_images = [[],[]]
        tmp_image = image
        for octave in range(self.num_octaves):
            if octave == 1: 
                height, width = tmp_image.shape[:2]
                resized_image = cv2.resize(tmp_image, (int(width/2), int(height/2)),interpolation = cv2.INTER_NEAREST)
                tmp_image = resized_image
            gaussian_images[octave].append(tmp_image)
            for i in range(1,self.num_guassian_images_per_octave): # 1 ~ 4
                gaussian_images[octave].append(cv2.GaussianBlur(tmp_image ,ksize= (0, 0), sigmaX= self.sigma**i, sigmaY=self.sigma**i))
                if i == 4:
                    tmp_image = cv2.GaussianBlur(tmp_image ,ksize = (0, 0),  sigmaX= self.sigma**i, sigmaY=self.sigma**i)
        # DEBUG
        # print(len(gaussian_images[0]))
        # for i in range( len(gaussian_images[0]) ):
        #     cv2.imshow('My Image', gaussian_images[0][i].astype(np.uint8))
        #     cv2.waitKey(0)
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
                    
        dog_images = [[],[]]
        for octave in range(self.num_octaves):
            for i in range(1,self.num_guassian_images_per_octave): # 1 ~ 4
                dog_images[octave].append( cv2.subtract( gaussian_images[octave][i], gaussian_images[octave][i-1] ) )

        # for i in range( len(dog_images[1]) ):
        #     cv2.imshow('My Image', dog_images[1][i].astype(np.uint8))
        #     cv2.waitKey(0)
        # # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        # #         Keep local extremum as a keypoint

        keypoints = [[],[]]
        for octave in range(self.num_octaves):

            for i in range(1, len(dog_images[octave]) - 1):
                for x in range(1, dog_images[octave][i].shape[0] - 1):
                    for y in range(1, dog_images[octave][i].shape[1] - 1):
                        patch_current = dog_images[octave][i][x-1:x+2, y-1:y+2]
                        patch_below = dog_images[octave][i-1][x-1:x+2, y-1:y+2]
                        patch_above = dog_images[octave][i+1][x-1:x+2, y-1:y+2]

                        if patch_current[1][1] == np.max([patch_current, patch_below, patch_above]):
                            if abs(patch_current[1,1]) <= self.threshold:
                                # print("lower than threshold")
                                continue
                            if octave == 1:
                                keypoints[octave].append([2*x, 2*y])
                            else:
                                keypoints[octave].append([x, y])
                            
                            
                        elif patch_current[1,1] == np.min([patch_current, patch_below, patch_above]):
                            if abs(patch_current[1,1]) <= self.threshold:
                                # print("lower than threshold")
                                continue
                            if octave == 1:
                                keypoints[octave].append([2*x, 2*y])
                            else:
                                keypoints[octave].append([x, y])
        # print(keypoints)
        # # Step 4: Delete duplicate keypoints
        # # - Function: np.unique
        # keypoints[0].pop(6)
        # keypoints[1].pop(6)                  

        keypoints = np.concatenate((keypoints[0],keypoints[1]), axis=0)        
        keypoints = np.array(keypoints)

        print(np.size(keypoints,0))  
        print(keypoints)          
        keypoints = np.unique(keypoints,axis=0) 
        print(np.size(keypoints,0))
        # sort 2d-point by y, then by x
        print("DEBUG")
        print( np.lexsort((keypoints[:,1],keypoints[:,0])) )
        keypoints = keypoints[ np.lexsort((keypoints[:,1],keypoints[:,0])) ] 
        print(keypoints)
        # keypoints = np.concatenate((keypoints[0],keypoints[1]), axis=0)
        return keypoints
