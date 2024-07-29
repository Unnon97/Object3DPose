'''
This script was an implementation approach to utilise SAM(Segment Anything Model from Facebook) based segmentation for 
object detection and further the 3D pose estimation system.

Author: Dheeraj Singh
Email: dheeraj.singh@rwth-aachen.de, singh97.dheeraj@gmail.com
'''


import sys
import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


try:
    config_path = '/home/dheeraj/unnon97/Object3DPose_project/config/pose.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("File not found. Check the path variable and filename")
    exit()

device = "cpu"



datadir = config["datadirectory"]
samples = config["numsamples"]

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def main():
    for sample in range(11):
        print("SAMPLE: ", sample)
        rgb_img = cv2.imread(datadir+f"{sample}/color.png", cv2.IMREAD_COLOR) [240:360,250:630]

        sam = sam_model_registry["vit_h"](checkpoint="/home/dheeraj/unnon97/monumentalProj_3DPose_dheeraj/checkpoints/sam_vit_h_4b8939.pth")
        sam.to(device=device)

        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(rgb_img)

        print(masks[0].keys())


        plt.imshow(rgb_img)
        show_anns(masks)
        plt.axis('off')
        output_file = f'SAM_output_{sample}.png'  # Specify your desired file name and format
        plt.savefig("/home/dheeraj/unnon97/monumentalProj_3DPose_dheeraj/samoutput/"+output_file, bbox_inches='tight', pad_inches=0, dpi=300)

        # cv2.imshow("rgb_img",rgb_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


