'''
This script is the main program used to detect and predict the 3D object pose of the object in frame of the camera. 

Author: Dheeraj Singh
Email: dheeraj.singh@rwth-aachen.de, singh97.dheeraj@gmail.com
'''


import sys
import os
import yaml
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils_cornerextraction import corners_with_radius, binary_object_edges

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


def draw_origin(img, origin, projection):
    img = cv2.line(img, origin, tuple(projection[0].ravel()), (0,0,255), 2)
    img = cv2.line(img, origin, tuple(projection[1].ravel()), (0,255,0), 2)
    img = cv2.line(img, origin, tuple(projection[2].ravel()), (255,0,0), 2)
    return img


def plotcornersonimage(rgb_image,vertices):
    for corner in vertices:
        x, y = corner.ravel()
        cv2.circle(rgb_image, (int(x), int(y)), 2, (255, 0, 0), -1)
    cv2.imshow("Features in frame",rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_in3d(points_3D):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_3D[:,0],points_3D[:,1],points_3D[:,2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def plotcornersonimage(rgb_image,vertices):
    for corner in vertices:
        x, y = corner.ravel()
        cv2.circle(rgb_image, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.imshow("Corners in RGB Image",rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def vertices_of_object(narrow_rgbimage, depthimage,sample):
    binary_edge_image = binary_object_edges(narrow_rgbimage, depthimage)

    binary_borders = cv2.GaussianBlur(binary_edge_image,(17,17),0)
    skeleton = skeletonize(binary_borders / 255)
    skeleton = (skeleton * 255).astype(np.uint8)
    vertices = corners_with_radius(skeleton,sample)
    vertices[:,0] += 275
    vertices[:,1] += 240
    vertices = vertices.astype(int)
    return vertices

def rot_trans_calculation(rgbimage, depthimage, metadata, object_dimensions,sample):
    fx, fy = metadata['fx'], metadata['fy']
    px, py = metadata['px'], metadata['py']
    dist_coeff = metadata["dist_coeffs"]
    dist_coeff = np.array(dist_coeff)
    width, height = metadata['width'], metadata['height']
    cx, cy = px, py

    fullrgb_image = rgbimage
    narrow_rgb_image = rgbimage[240:340,275:605]
    depth_image = depthimage
    corners = vertices_of_object(narrow_rgb_image, depth_image,sample)
    
    pointsin2d = corners

    # 3D points of all properly orientated objects in frame of the corners detected in image, first element is the center point and next 4 are corners of the object 
    # going anti-clockwise from the right top corner to right bottom corner
    object3dpoints_gt = np.array([[0.0, -object_dimensions[1]/2, 0.0], 
                                 [object_dimensions[0]/2,-object_dimensions[1]/2,object_dimensions[2]/2],
                                 [-object_dimensions[0]/2,-object_dimensions[1]/2,object_dimensions[2]/2], 
                                 [-object_dimensions[0]/2,-object_dimensions[1]/2,-object_dimensions[2]/2],
                                 [object_dimensions[0]/2,-object_dimensions[1]/2,-object_dimensions[2]/2]])
    object_axes = np.array([[1,0,0],[0,1,0],[0,0,1]])
    centerpoint = corners[0,:]
    
    # 3D points of object orientated 90 degrees rotated from ideal placing format. The ordering of the points is same as before.
    if (np.abs(corners[:,0]-centerpoint[0]) < 100).all() & (np.abs(corners[:,1]-centerpoint[1]) < 50).all() :
        object3dpoints_gt = np.array([[-object_dimensions[0]/2, 0.0, 0.0], 
                                     [-object_dimensions[0]/2, -object_dimensions[1]/2, object_dimensions[2]/2],
                                     [-object_dimensions[0]/2, object_dimensions[1]/2, object_dimensions[2]/2], 
                                     [-object_dimensions[0]/2, object_dimensions[1]/2, -object_dimensions[2]/2],
                                     [-object_dimensions[0]/2, -object_dimensions[1]/2,-object_dimensions[2]/2]])
        object_axes = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    # plotcornersonimage(fullrgb_image, corners)

    # plot_in3d(object3dpoints_gt)
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
        ], dtype=np.float32)
        
    object_origin = corners[0]
    success, rvec, tvec ,inliers = cv2.solvePnPRansac(object3dpoints_gt.astype('float32'),pointsin2d.astype('float32'),camera_matrix, dist_coeff, iterationsCount = 200,confidence = 0.99,reprojectionError=3.0,)
  
    projection, ipts = cv2.projectPoints(object_axes.astype('float32'), rvec, tvec, camera_matrix, dist_coeff)

    fullrgb_image = draw_origin(fullrgb_image, object_origin, projection.astype(int))

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    tvec = tvec.reshape((3,))

    # print("DEPTH",depth_image[object_origin[0],object_origin[1]])
    # cv2.imshow('object origin',fullrgb_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(projectdir+"output_images/object_poses/"+f"{sample}_object_pose.jpg",fullrgb_image)
    return tvec, euler_angles



def main():
    for sample in range(11):
        print("Sample: ",sample)
        metadata = datadir + f"{sample}/cam.json"
        with open(metadata) as f:
            metadata = json.load(f)
        
        object_dimensions = [config['objectdimension']['length'],
                            config['objectdimension']['breadth'],
                            config['objectdimension']['height']]

        fullrgb_image = cv2.imread(datadir+f"{sample}/color.png", cv2.IMREAD_COLOR)
        depth_image = cv2.imread(datadir+f"{sample}/depth.png", cv2.IMREAD_UNCHANGED)

        position, orientation = rot_trans_calculation(fullrgb_image,depth_image,metadata, object_dimensions,sample)
        position = position.reshape((3,))
        print("object position : ", position)
        print("object orientation : ", orientation)
        print()

if __name__ == '__main__':
    main()