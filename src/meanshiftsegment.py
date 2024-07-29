'''
This script contains an implementation approach to utilise color values of pixels to group the object related pixels to detect it properly.
This script uses meanshift based clustering for object segmentation.

Author: Dheeraj Singh
Email: dheeraj.singh@rwth-aachen.de, singh97.dheeraj@gmail.com
'''


import sys
import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt


try:
    config_path = '/home/dheeraj/unnon97/Object3DPose_project/config/pose.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("File not found. Check the path variable and filename")
    exit()


datadir = config["datadirectory"]
samples = config["numsamples"]

def luv_color(datadir, sample):
    rgb_img = cv2.imread(datadir+f"{sample}/color.png", cv2.IMREAD_COLOR) [240:350,250:630]
    
    im_luv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LUV)
    plt.imshow(im_luv)
    plt.show()

def find_peak_opt(data, query, radius, c=3):
    is_near_search_path = np.zeros(len(data), dtype=bool)
    
    shift = np.inf
    while shift > 0:
        dist = np.linalg.norm(data - query, axis=1)
        query_old = query
        query = np.mean(data[dist <= radius], axis=0)
        shift = np.linalg.norm(query - query_old)
        is_near_search_path[dist <= radius/c] = True
   
    return query, is_near_search_path

def mean_shift_opt2(data, radius):
    labels = np.full(len(data), fill_value=-1, dtype=int)
    
    peaks = np.empty(data.shape)
    n_peaks = 0
    
    for idx, query in enumerate(data):
        if labels[idx] != -1:
            continue
            
        peak, is_near_search_path = find_peak_opt(data, query, radius)
        label = None
        
        if n_peaks > 0:
            dist = np.linalg.norm(peaks[:n_peaks] - peak, axis=1)
            label_of_nearest_peak = np.argmin(dist)
            
            if dist[label_of_nearest_peak] <= radius / 2:
                label = label_of_nearest_peak
        
        if label is None:
            label = n_peaks
            peaks[label] = peak
            n_peaks += 1
            
            dist = np.linalg.norm(data - peak, axis=1)
            labels[dist <= radius] = label
            
        labels[is_near_search_path] = label
    
    peaks = peaks[:n_peaks]
    
    return peaks, labels

def mean_shift_segment_luv_pos(im, radius, pos_weight=1):
    im_luv = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)
    y, x = np.mgrid[:im.shape[0], :im.shape[1]]
    xy = np.stack([x, y], axis=-1)
    feature_im = np.concatenate([im_luv, pos_weight*xy], axis=-1)
    data = feature_im.reshape(-1, 5).astype(float)
    
    peaks, labels = mean_shift_opt2(data, radius)
    
    segmented_im = peaks[labels][..., :3].reshape(im.shape).astype(np.uint8)
    segmented_im = cv2.cvtColor(segmented_im, cv2.COLOR_LUV2RGB)
    return data, peaks, labels, segmented_im

def mean_shift_segment_luv(im, radius):
    im_luv = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)
    data = im_luv.reshape(-1, 3).astype(float)
    
    peaks, labels = mean_shift_opt2(data, radius)
    
    segmented_im = peaks[labels].reshape(im.shape).astype(np.uint8)
    segmented_im = cv2.cvtColor(segmented_im, cv2.COLOR_LUV2RGB)
    return data, peaks, labels, segmented_im

def find_peak_opt(data, query, radius, c=3):
    is_near_search_path = np.zeros(len(data), dtype=bool)
    
    shift = np.inf
    while shift > 0:
        dist = np.linalg.norm(data - query, axis=1)
        query_old = query
        query = np.mean(data[dist <= radius], axis=0)
        shift = np.linalg.norm(query - query_old)
        is_near_search_path[dist <= radius/c] = True
   
    return query, is_near_search_path

def mean_shift_opt2(data, radius):
    labels = np.full(len(data), fill_value=-1, dtype=int)
    
    peaks = np.empty(data.shape)
    n_peaks = 0
    
    for idx, query in enumerate(data):
        if labels[idx] != -1:
            continue
            
        peak, is_near_search_path = find_peak_opt(data, query, radius)
        label = None
        
        if n_peaks > 0:
            dist = np.linalg.norm(peaks[:n_peaks] - peak, axis=1)
            label_of_nearest_peak = np.argmin(dist)
            
            if dist[label_of_nearest_peak] <= radius / 2:
                label = label_of_nearest_peak
        
        if label is None:
            label = n_peaks
            peaks[label] = peak
            n_peaks += 1
            
            dist = np.linalg.norm(data - peak, axis=1)
            labels[dist <= radius] = label
            
        labels[is_near_search_path] = label
    
    peaks = peaks[:n_peaks]
    
    return peaks, labels

def mean_shift_segment_luv_pos(im, radius, pos_weight=1):
    im_luv = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)
    y, x = np.mgrid[:im.shape[0], :im.shape[1]]
    xy = np.stack([x, y], axis=-1)
    feature_im = np.concatenate([im_luv, pos_weight*xy], axis=-1)
    data = feature_im.reshape(-1, 5).astype(float)
    
    peaks, labels = mean_shift_opt2(data, radius)
    
    segmented_im = peaks[labels][..., :3].reshape(im.shape).astype(np.uint8)
    segmented_im = cv2.cvtColor(segmented_im, cv2.COLOR_LUV2RGB)
    return data, peaks, labels, segmented_im

def mean_shift_segment_luv(im, radius):
    im_luv = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)
    data = im_luv.reshape(-1, 3).astype(float)
    
    peaks, labels = mean_shift_opt2(data, radius)
    segmented_im = peaks[labels].reshape(im.shape).astype(np.uint8)
    segmented_im = cv2.cvtColor(segmented_im, cv2.COLOR_LUV2RGB)
    return data, peaks, labels, segmented_im


def mean_shift_segment(im, radius):
    data = im.reshape(-1, 3).astype(float)
    peaks, labels = mean_shift_opt2(data, radius)
    segmented_im = peaks[labels].reshape(im.shape).astype(np.uint8)
    return data, peaks, labels, segmented_im

def main():
    for sample in range(6,11):
        rgbimg = cv2.imread(datadir+f"{sample}/color.png", cv2.IMREAD_COLOR) [240:350,250:630]

        data, peaks, labels, segmented_im = mean_shift_segment_luv(rgbimg, radius=5)
        cv2.imshow("segmented_im",segmented_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()