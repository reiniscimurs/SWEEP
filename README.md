# SWEEP - Structure and Wall Extraction from Evacuation Plan

It is very difficult for robots to navigate in unknown environments, especially if their motion optimality is of concern. A SLAM approach can be used to obtain a map to perform optimal planning for robot motion. However, this generally requires time, effort and resources to navigate through the environment and record the surroundings with sensor data. An alternative is to create a prior map based on the blueprints of the buildings the robot would be deployed in. But this information is rarely available and accesible. On the other hand, evacuation plans and similar graphical representations of the building layouts are freely displayed for everyone. Taking an image of such a layout and transforming it into a representation of obstacles can allow robots to create initial navigation plans to a given goal with prior information. This program extracts the structured obstacle and wall data from such images and creates a ROS map, based on human operator inputs.
 
Dependencies:
* Python 3.6
* OpenCV
* SciPy

**Results**
Original image on the left and the resulting output on the right. Identifying features have been removed for some images in this repository, but they are present when performing SWEEP.

Result 1:
<p float="left">
  <img src="https://github.com/reiniscimurs/SWEEP/blob/main/Results/f1.jpg" width="300" />
  <img src="https://github.com/reiniscimurs/SWEEP/blob/main/Results/res1.jpg" width="300" /> 
</p>

Result 3:
<p float="left">
  <img src="https://github.com/reiniscimurs/SWEEP/blob/main/Results/f3.jpg" width="300" />
  <img src="https://github.com/reiniscimurs/SWEEP/blob/main/Results/res3.jpg" width="300" /> 
</p>

Result 8:
<p float="left">
  <img src="https://github.com/reiniscimurs/SWEEP/blob/main/Results/f8.jpg" width="300" />
  <img src="https://github.com/reiniscimurs/SWEEP/blob/main/Results/res8.jpg" width="300" /> 
</p>

Result 11:
<p float="left">
  <img src="https://github.com/reiniscimurs/SWEEP/blob/main/Results/f11.jpg" width="300" />
  <img src="https://github.com/reiniscimurs/SWEEP/blob/main/Results/res11.jpg" width="300" /> 
</p>
