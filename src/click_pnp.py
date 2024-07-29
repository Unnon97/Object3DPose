'''
This script contains the implementation of using mouse clicks to specifically obtain the pixel locations from an image.

Author: Dheeraj Singh
Email: dheeraj.singh@rwth-aachen.de, singh97.dheeraj@gmail.com
'''


import sys
import os
import yaml
import cv2
import json

from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


try:
    config_path = '/home/dheeraj/unnon97/Object3DPose_project/config/pose.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("File not found. Check the path variable and filename")
    exit()


datadir = config["datadirectory"]
samples = config["numsamples"]
current_sample = config["sample_image"]
projectdir = config["projectdirectory"]

sample = 0
print("Sample: ",sample)
metadata = datadir + f"{sample}/cam.json"
with open(metadata) as f:
    metadata = json.load(f)

def click_event(event, x, y, flags, params): 
  
    if event == cv2.EVENT_LBUTTONDOWN: 
        print(x, ' ', y) 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('image', img) 
       
    if event==cv2.EVENT_RBUTTONDOWN: 
        print(x, ' ', y)   
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r), 
                    (x,y), font, 1, 
                    (255, 255, 0), 2) 
        cv2.imshow('image', img) 
  
if __name__=="__main__": 
    img = cv2.imread(datadir+f"{sample}/color.png", cv2.IMREAD_COLOR)
    cv2.imshow('image', img) 
    cv2.setMouseCallback('image', click_event) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 