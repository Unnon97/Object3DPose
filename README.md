# Object Pose Estimation Project

This project contains scripts that utilise RGB-D input obtained from an Intel Realsense D405 camera on the robot end effector and provides the object (here object layed just before taking the photo) pose: position and orientation.



## Repository description
The repository contents can be described as follows:
1. "checkpoints" folder stores the Segment Anything Model trained weights 
2. "config" folder contains the important yaml files used to set user defined inputs
3. "data" folder contains the input datasets used for the project
4. "output_images" folder contains different categories of images obtained during the solution implementation
5. "samoutput" folder contains images obtained after applying segmentation from SAM to experiment a different approach.
6. "scripts" folder contains all the main scripts used to run the project
7. "src" folder contains the python scripts that contain relevant classes, methods and models(in case of a Neural Network model) utilized for the project
8. "run.sh" Shell script that runs the HTTP server and requests to the server the pose estimate of provided data


## Procedure To Replicate

1. Build a docker container or python virtual-env with recommended python packages. To install the packages, use the requirements.txt file as
'pip install -r requirements.txt'
2. Once inside the virtual environment or docker container, we can begin the objective of 3D object pose extraction
3. To run the complete process as a HTTP server app and request, run './run.sh' shell script once you modify the pose.yaml file in config folder to specify the image to be used.
4. To run only the pose estimation part for each image, change the directories in yaml file and also the directory names to load the yaml files in script "object_pose_estimation.py" and "utils_cornerextraction.py". Next, run '''python src/object_pose_estimation.py''' to get object position and orientation. To visualise the 3D axes in frame, uncomment lines 142-144 in "object_pose_estimation.py".

<!-- ## Input images
<p style="display: flex; justify-content: space-between;">
  <img src="data/place_quality_inputs/0/color.png" style="width: 48%;" />
  <img src="data/place_quality_inputs/1/color.png" style="width: 48%;" /> 
</p>
<p style="display: flex; justify-content: space-between;">
  <img src="data/place_quality_inputs/2/color.png" style="width: 48%;" /> 
  <img src="data/place_quality_inputs/3/color.png" style="width: 48%;" /> 
</p>

<p style="display: flex; justify-content: space-between;">
  <img src="data/place_quality_inputs/0/depth.png" style="width: 48%;" />
  <img src="data/place_quality_inputs/1/depth.png" style="width: 48%;" /> 
</p>
<p style="display: flex; justify-content: space-between;">
  <img src="data/place_quality_inputs/2/depth.png" style="width: 48%;" /> 
  <img src="data/place_quality_inputs/3/depth.png" style="width: 48%;" /> 
</p> -->


## Output
Based on the input image samples shown above, we run the PnP algorithm to extract the object position and orientation using classical computer vision methods. More output images are in the folder "output_images".

The output value of position and orientation of the object in above shown image is: 

<!-- ### object Pose in image
<p style="display: flex; justify-content: space-between;">
  <img src="output_images/object_poses//0_object_pose.jpg" style="width: 48%;" />
  <img src="output_images/object_poses//1_object_pose.jpg" style="width: 48%;" />
</p>
<p style="display: flex; justify-content: space-between;">
  <img src="output_images/object_poses//2_object_pose.jpg" style="width: 48%;" /> 
  <img src="output_images/object_poses//3_object_pose.jpg" style="width: 48%;" /> 
</p> -->


The calculated position and orientation of the object for each image is
1. Image 0:
    * Position: [0.00805696 0.03652409 0.36702799]
    * Orientation: [91.16790437 -0.26187287 -0.13841589]

2. Image 1:
    * Position: [0.0082818  0.03977524 0.36663242]
    * Orientation: [ 8.90161177e+01 -3.61514251e-02  5.50943923e-03]

3. Image 2:
    * Position:  [0.00819917 0.03399753 0.36674328]
    * Orientation: [ 9.29081367e+01 -2.54369589e-01 -7.10254744e-02]

4. Image 3:
    * Position: [0.00792732 0.03702736 0.3672435 ]
    * Orientation: [89.91971714 -1.00304544  0.34618421]
