# The View of Delft dataset
This repository shares the documentation and development kit of the View of Delft automotive dataset.

<div align="center">
<figure>
<img src="docs/figures/example_frame_2.png" alt="Prius sensor setup" width="700"/>
</figure>
<br clear="left"/>
<b>Example frame from our dataset with camera, LiDAR, 3+1D radar, and annotations overlaid.</b>
</div>
<br />
<br />

## Overview
- [Introduction](#introduction)
- [Sensors and Data](docs/SENSORS.md)
- [Annotation](docs/ANNOTATIONS.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Examples and Demo](docs/EXAMPLES.md)
- [Citation](#citation)


## Introduction

We present the novel View-of-Delft (VoD) automotive dataset. It contains 8600 frames of synchronized and calibrated 64-layer LiDAR-, (stereo) camera-, and 3+1D radar-data acquired in complex, urban traffic. It consists of more than 76000 3D bounding box annotations, including more than 15000 pedestrian, 7100 cyclist and 17000 car labels.

<div align="center">
<p float="center">
<img src="docs/figures/sensors.gif" alt="Prius sensor setup" width="400"/>
<img src="docs/figures/labels.gif" alt="Prius sensor setup" width="400"/>
</p>
</div>

## Sensors and data
The LiDAR sensor is a Velodyne 64 operating at 10 Hz. The camera provides colored images of 1936 × 1216 pixels at around 12 Hz. The horizontal field of view is ~64° (± 32°), vertical field of view is ~ 44° (± 22°). Odometry is a filtered combination of several inputs: RTK GPS, IMU, and wheel odometry with a frame rate around 100 Hz. Odometry is given relative to the starting position at the beginning of the current sequence, not in absolute coordinates. Note that while camera and odometry operate at a higher frequency than the LiDAR sensor, timestamps of the LiDAR sensor were chosen as “lead”, and we provide the closest camera frame and odometry information available. 

The provided LiDAR point-clouds are already ego-motion compensated. More specifically, misalignment of LiDAR and camera data originating from ego-motion during the scan (i.e. one full rotation of the LiDAR sensor) and ego-motion between the capture of LiDAR and camera data has been compensated. See an example frame for our data with calibrated and ego-motion compensated LiDAR point-cloud overlaid on the camera image. 

We also provide intrinsic calibration for the camera and extrinsic calibration of all three sensors in a format specified by the annotating party.

<div align="center">
<img src="docs/figures/Prius_sensor_setup_5.png" alt="Prius sensor setup" width="600"/>
</div>


## Annotation

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

## Examples and Demos

Please refer to [EXAMPLES.md](docs/GETTING_STARTED.md) to learn more usage about this project.

## License


## Acknowledgement


## Citation 
If you find the dataset useful in your research, please consider citing it as:

```
@ARTICLE{apalffy2022,
  author={Palffy, Andras and Pool, Ewoud and Baratam, Srimannarayana and Kooij, Julian F. P. and Gavrila, Dariu M.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Multi-Class Road User Detection With 3+1D Radar in the View-of-Delft Dataset}, 
  year={2022},
  volume={7},
  number={2},
  pages={4961-4968},
  doi={10.1109/LRA.2022.3147324}}
```



