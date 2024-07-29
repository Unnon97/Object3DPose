'''
This script runs the app for providing 3D pose estimate of a object in the frame through HTTP server requests.

Author: Dheeraj Singh
Email: dheeraj.singh@rwth-aachen.de, singh97.dheeraj@gmail.com
'''


import sys
import os
from flask import Flask, request, jsonify
import numpy as np
import cv2
import json
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.object_pose_estimation import rot_trans_calculation

try:
    config_path = '/home/dheeraj/unnon97/Object3DPose_project/config/pose.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("File not found. Check the path variable and filename")
    exit()



app = Flask(__name__)
   

@app.route("/poseestimation", methods = ['POST'])
def estimation():
    if 'rgb' not in request.files:
        return jsonify({'error':'No rgb image given'}), 400
    if 'depth' not in request.files:
        return jsonify({'error':'No depth image given'}), 400
    if 'metadata' not in request.files:
        return jsonify({'error':'No metadata provided'}), 400
    
    rgbfile = request.files['rgb']
    depthfile = request.files['depth']
    jsonfile = request.files['metadata']

    object_dimensions = [config['objectdimension']['length'],
                        config['objectdimension']['breadth'],
                        config['objectdimension']['height']]
    rgbimage = np.frombuffer(rgbfile.read(), np.uint8)
    rgbimage = cv2.imdecode(rgbimage, cv2.IMREAD_COLOR)

    depthimage = np.frombuffer(depthfile.read(), np.uint8)
    depthimage = cv2.imdecode(depthimage, cv2.IMREAD_UNCHANGED)

    jsondata = jsonfile.read().decode('utf-8')
    metadata = json.loads(jsondata)

    position, orientation = rot_trans_calculation(rgbimage, depthimage, metadata, object_dimensions, sample=0)

    response = {
        'position': position.tolist(),
        'orientation': orientation.tolist()
    }
    
    return jsonify(response)


if __name__== '__main__':
    app.run(debug=True)