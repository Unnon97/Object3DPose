'''
This script creates a request to be sent to the app by providing input data images of color, 
depth and camera parameters to obtain the response of position and orientation of the object

Author: Dheeraj Singh
Email: dheeraj.singh@rwth-aachen.de, singh97.dheeraj@gmail.com
'''


import requests

import yaml
try:
    config_path = '/home/dheeraj/unnon97/Object3DPose_project/config/pose.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("File not found. Check the path variable and filename")
    exit()


im = config["sample_image"]
projectdir = config["projectdirectory"]
url = 'http://127.0.0.1:5000/poseestimation'
files = {'rgb': open(f'{projectdir}/data/{im}/color.png', 'rb'),
         'depth':open(f'{projectdir}/data/{im}/depth.png', 'rb'),
         'metadata':open(f'{projectdir}/data/{im}/cam.json', 'rb'),}
response = requests.post(url, files = files)

print(response.json())