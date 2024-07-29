'''
This script was an implementation approach to utilise color values of pixels to group the object related pixels to detect it properly.

Author: Dheeraj Singh
Email: dheeraj.singh@rwth-aachen.de, singh97.dheeraj@gmail.com
'''


import sys
import os
import cv2
import yaml
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import rgbddata
from src.classicalsam_overlayed import binary_edge_extractor


try:
    config_path = '/home/dheeraj/unnon97/Object3DPose_project/config/pose.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("File not found. Check the path variable and filename")
    exit()


datadir = config["datadirectory"]
samples = config["numsamples"]

def kmean_segmentation(image,k=3):
    centerpixelx,centerpixely = image.shape[0]//2, image.shape[1]//2
    if len(image.shape) == 3:
        pixel_vals = image.reshape((-1,3))
    else:
        pixel_vals = image.reshape((-1,1))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
    # then perform k-means clustering with number of clusters defined as 3
    #also random centres are initially choosed for k-means clustering
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))

    segmented_image[segmented_image != segmented_image[centerpixelx,centerpixely]] = 0
    # cv2.imshow("segmented_im",segmented_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return segmented_image

def kmeans_in_depth(datadir,sample):
    depth_img = cv2.imread(datadir+f"{sample}/depth.png", cv2.IMREAD_UNCHANGED) [240:340,275:605]

    depth_img = cv2.normalize(depth_img, None, 0, 65535, cv2.NORM_MINMAX)
    # depth_img = cv2.GaussianBlur(depth_img,(1,11),0)
    return kmean_segmentation(depth_img)

def kmeans_in_rgb(datadir,sample):
    rgb_img = cv2.imread(datadir+f"{sample}/color.png", cv2.IMREAD_COLOR) [240:340,275:605]
    rgb_img = cv2.GaussianBlur(rgb_img,(7,7),0)
    return kmean_segmentation(rgb_img)

def kmeans_in_luv(datadir, sample):
    rgb_img = cv2.imread(datadir+f"{sample}/color.png", cv2.IMREAD_COLOR) [240:340,275:605]
    rgb_img = cv2.GaussianBlur(rgb_img,(7,7),0)
    luv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LUV)
    return kmean_segmentation(luv_img)

def kmeans_in_hsv(datadir, sample):
    rgb_img = cv2.imread(datadir+f"{sample}/color.png", cv2.IMREAD_COLOR) [240:340,275:605]
    rgb_img = cv2.GaussianBlur(rgb_img,(7,7),0)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return kmean_segmentation(hsv_img)


def main():
    for sample in range(11):
        print("SAMPLE")
        print(sample)
        segmented_im = kmeans_in_rgb(datadir,sample)
        cv2.imshow("segmented_im",segmented_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# def kmean_segmentation_with_position(image,k=3):
#     y, x = np.mgrid[:image.shape[0], :image.shape[1]]
#     xy = np.stack([x, y], axis=-1)
#     feature_im = np.concatenate([image, xy], axis=-1)
#     data = feature_im.reshape(-1, 5).astype(float)
    
    
#     centerpixelx,centerpixely = image.shape[0]//2, image.shape[1]//2
#     pixel_vals = image.reshape((-1,5))
#     pixel_vals = np.float32(pixel_vals)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
#     # then perform k-means clustering with number of clusters defined as 3
#     #also random centres are initially choosed for k-means clustering
#     retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     print("ce",centers)
#     print("la",labels)
#     cv2.imshow("labels",labels)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     segmented_im = retval[labels][..., :3].reshape(image.shape).astype(np.uint8)
#     # convert data into 8-bit values
#     centers = np.uint8(centers)
#     segmented_data = centers[labels.flatten()]
    
#     # reshape data into the original image dimensions
#     segmented_image = segmented_data.reshape((image.shape))

#     segmented_image[segmented_image != segmented_image[centerpixelx,centerpixely]] = 0
#     cv2.imshow("segmented_im",segmented_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return segmented_image
