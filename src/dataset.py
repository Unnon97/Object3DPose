'''
This script provides an easier accessing setup for dataset of color.png, depth.png and cam.json 
that can be used in case of training models or performing 3D pose estimation.

Author: Dheeraj Singh
Email: dheeraj.singh@rwth-aachen.de, singh97.dheeraj@gmail.com
'''

import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import yaml
import json
try:
    config_path = '/home/dheeraj/unnon97/Object3DPose_project/config/pose.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("File not found. Check the path variable and filename")
    exit()


datadir = config["datadirectory"]
samples = config["numsamples"]


class rgbddata(Dataset):
    def __init__(self, targ_dir=datadir):
        self.path = targ_dir
        self.rgbimg_data = []
        self.depthimg_data = []
        self.jsondata = []
        self.data = {}
        for img_path in glob.glob(self.path + "*/color.png"):
            self.rgbimg_data.append(img_path)
        
        for img_path in glob.glob(self.path + "*/depth.png"):
            self.depthimg_data.append(img_path)

        for camdata in glob.glob(self.path + "*/cam.json"):
            self.jsondata.append(camdata)
        print("Number of rgb images: ", len(self.rgbimg_data))
        print("Number of depth images: ", len(self.depthimg_data))

    
    def load_image(self, index: int):
        rgb_img = self.rgbimg_data[index]
        return cv2.imread(rgb_img)
    
    def load_depth(self, index: int):
        depthimg = self.depthimg_data[index]
        return cv2.imread(depthimg)
    
    def load_metadata(self, index: int):
        metadata = self.jsondata[index]
        with open(metadata) as f:
            d = json.load(f)
    
    def __len__(self):
        return len(self.imgdata)
    
    def __getitem__(self, idx):
        img = self.load_image(idx)
        depth = self.load_depth(idx)
        metadata = self.load_metadata(idx)
        return img, depth, metadata
    
    def visualise(self, data):
        cv2.imshow("visual",data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__=='__main__':
    customData = rgbddata()
    for i in range(11):
        image, depthimage, metad = customData[i]
        customData.visualise(image)
        print("Depth image values",depthimage.shape,np.unique(depthimage),np.max(depthimage), np.min(depthimage))
        customData.visualise(depthimage)