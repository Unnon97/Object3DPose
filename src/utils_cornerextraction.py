'''
This script is contains the relevant functions for extracting keypoints from the input images.

Author: Dheeraj Singh
Email: dheeraj.singh@rwth-aachen.de, singh97.dheeraj@gmail.com
'''


import sys
import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

from scipy import stats, ndimage as nd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



try:
    config_path = '/home/dheeraj/unnon97/Object3DPose_project/config/pose.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("File not found. Check the path variable and filename")
    exit()


projectdir = config["projectdirectory"]
datadir = config["datadirectory"]
samples = config["numsamples"]

def plotcornersonimage(rgb_image,vertices):
    cv2.imshow("IMAGWEWITH OUT in frame",rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for corner in vertices:
        x, y = corner.ravel()
        # print("xx",x,"yy",y)
        cv2.circle(rgb_image, (int(x), int(y)), 5, (255, 0, 0), -1)

    cv2.imshow("Features in frame",rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def structuring_corners(corners):
    if corners.shape[0] > 5:
        reference_point = np.array(corners[0].tolist())
        distances = np.linalg.norm(corners - reference_point, axis=1)
        closest_index = np.argsort(distances)[:5]
        corners = corners[closest_index]   

    centerpoint = corners[0].tolist()
    final_features = [centerpoint]
    condition1 = (corners[:,0]>centerpoint[0]) & (corners[:,1]<centerpoint[1])
    condition2 = (corners[:,0]<centerpoint[0]) & (corners[:,1]<centerpoint[1])
    condition3 = (corners[:,0]<centerpoint[0]) & (corners[:,1]>centerpoint[1])
    condition4 = (corners[:,0]>centerpoint[0]) & (corners[:,1]>centerpoint[1])

    final_features.append(corners[condition1][0].tolist())
    final_features.append(corners[condition2][0].tolist())
    final_features.append(corners[condition3][0].tolist())
    final_features.append(corners[condition4][0].tolist())
    final_features = np.array(final_features)

    return final_features

def corners_with_radius(image, sample, threshold=0.15, k=0.04, block_size=15, aperture_size=3):

    
    gray = image
    corners_harris = cv2.cornerHarris(gray, blockSize=block_size, ksize=aperture_size, k=k)
    
    dilated_corners = cv2.dilate(corners_harris, None)
    
    _, threholded_corners = cv2.threshold(dilated_corners, threshold * dilated_corners.max(), 255, 0)
    threholded_corners = np.uint8(threholded_corners)

    _, _, _, centroids = cv2.connectedComponentsWithStats(threholded_corners,4)
    # plotcornersonimage(image,centroids)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1, -1), criteria)
    
    
    corners = structuring_corners(corners)

    # rgb_img = cv2.imread(datadir+f"{sample}/color.png", cv2.IMREAD_COLOR)
    # corners[:,0]+=275
    # corners[:,1]+=240
    # for corner in corners:
    #     x, y = corner[0],corner[1]
    #     cv2.circle(rgb_img, (int(x), int(y)), 5, (0, 0, 255), -1)
    # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    # plt.title("INSIDE DETECT")
    # plt.imshow(rgb_img)
    # plt.axis('off')
    # plt.show()
    # plt.imsave(projectdir+"output_images/keypoints_on_brick/"+f"{sample}_brick_corners.jpg", rgb_img)
    # plt.imsave(projectdir+"output_images/keypoints_on_real_image/"+f"{sample}_keypoint_overlays.jpg", rgb_img)

    return corners

def binary_object_edges(narrow_rgbimage, depthimage):
    rgb_image = narrow_rgbimage
    depth_image = depthimage[240:340,275:605]
    normalized_depth = cv2.normalize(depth_image, None, 0, 65535, cv2.NORM_MINMAX)

    centerpixelx,centerpixely = rgb_image.shape[0]//2, rgb_image.shape[1]//2
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # cv2.imshow("hsv",hsv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mask = cv2.inRange(hsv, (100,20,20),(130,150,150))
    closedmask = nd.binary_closing(mask,np.ones((37,37)))

    lower_bound = normalized_depth[centerpixelx,centerpixely] - 650
    upper_bound = normalized_depth[centerpixelx,centerpixely] + 750
    mask_depth = (normalized_depth >= lower_bound) & (normalized_depth <= upper_bound)

    depth_masked_image = np.zeros_like(normalized_depth)
    depth_masked_image[mask_depth] = normalized_depth[mask_depth]



    combined_detect = closedmask*depth_masked_image

    gray = combined_detect
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)

    edges = np.uint8(edges / edges.max() * 255) 

    _, binary_edges = cv2.threshold(edges, 20, 150, cv2.THRESH_BINARY)

    # cv2.imshow("binary_edges",binary_edges)
    # cv2.waitKey(0)
    # cv2.imshow("sobelx",sobel_x)
    # cv2.waitKey(0)
    # cv2.imshow("sobely",sobel_y)
    # cv2.waitKey(0)
    # cv2.imshow("edges",edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return binary_edges

def main():
    for sample in range(10,11):
        print("SAMPLE: ", sample)
        rgb_img = cv2.imread(datadir+f"{sample}/color.png", cv2.IMREAD_COLOR) [240:340,275:605]
        rgb_img = cv2.GaussianBlur(rgb_img,(21,21),0)
        depth_img = cv2.imread(datadir+f"{sample}/depth.png", cv2.IMREAD_UNCHANGED)


        binary_borders = binary_object_edges(rgb_img,depth_img)

        # cv2.imshow('binary_borders',binary_borders)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # plt.imshow(binary_borders)
        # plt.show()

        binary_borders = cv2.GaussianBlur(binary_borders,(17,17),0)

        skeleton = skeletonize(binary_borders / 255)
        skeleton = (skeleton * 255).astype(np.uint8)

        # plt.title("skeleton")
        # plt.imshow(skeleton)
        # plt.show()
        centerx,centery = rgb_img.shape[0]//2,rgb_img.shape[1]//2
        corners = corners_with_radius(skeleton,sample)
        print("corners",corners)


        rectanlgecorners = []
        for cor in corners:
            x, y = int(cor[0]),int(cor[1])
            if not (x-5 <= centery <= x+5 and y-5 <= centerx <= y+5):
                rectanlgecorners.append([x,y])

        rectanlgecorners = np.array(rectanlgecorners)
        
        cv2.polylines(rgb_img, [rectanlgecorners], isClosed=True, color=(0, 255, 0), thickness=2)

        # cv2.imshow("Rectangle", rgb_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()