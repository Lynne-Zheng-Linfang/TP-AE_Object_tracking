# Warning: Not finished. Please wait until the reporsitory is finished.

## TP-AE  

### TP-AE: Temporally Primed 6D Object Pose Tracking with Auto-Encoders   
MLinfang Zheng, Ales Leonardis, Tze Ho Elden Tse, Nora Horanyi, Hua Chen, Wei Zhang, Hyung Jin Chang, ICRA2022    

[paper](https://lynne-zheng-linfang.github.io/TP-AE.github.io/resources/TP-AE.pdf)



## Overview

Fast and accurate tracking of an object's motion is one of the key functionalities of a robotic system for achieving reliable interaction with the environment. This paper focuses on the instance-level six-dimensional (6D) pose tracking problem with a symmetric and textureless object under occlusion. We propose a Temporally Primed 6D pose tracking framework with Auto-Encoders (TP-AE) to tackle the pose tracking problem. The framework consists of a prediction step and a temporally primed pose estimation step. The prediction step aims to quickly and efficiently generate a guess on the object's real-time pose based on historical information about the target object's motion. Once the prior prediction is obtained, the temporally primed pose estimation step embeds the prior pose into the RGB-D input, and leverages auto-encoders to reconstruct the target object with higher quality under occlusion, thus improving the framework's performance. Extensive experiments show that the proposed 6D pose tracking method can accurately estimate the 6D pose of a symmetric and textureless object under occlusion, and significantly outperforms the state-of-the-art on T-LESS dataset while running in real-time at 26 FPS. 

<!-- <p align="center">
<img src='docs/pipeline_with_scene_vertical_ext.jpeg' width='600'>
<p> -->
<!-- 
1.) Train the Augmented Autoencoder(s) using only a 3D model to predict 3D Object Orientations from RGB image crops \
2.) For full RGB-based 6D pose estimation, also train a 2D Object Detector (e.g. https://github.com/fizyr/keras-retinanet) \
3.) Optionally, use our standard depth-based ICP to refine the 6D Pose -->

## Requirements: Hardware
### For Training
Nvidia GPU with >4GB memory (or adjust the batch size)  
RAM >8GB  
Duration depending on Configuration and Hardware: ~18h per Object

## Requirements: Software

Linux Python 3 

GLFW for OpenGL: 
```bash
sudo apt-get install libglfw3-dev libglfw3  
```
Assimp: 
```bash
sudo apt-get install libassimp-dev  
```

Tensorflow >= 1.6  
OpenCV >= 3.1

```bash
pip install --user --pre --upgrade PyOpenGL PyOpenGL_accelerate
pip install --user cython
pip install --user cyglfw3
pip install --user pyassimp==3.3
pip install --user imgaug
pip install --user progressbar
```

### Headless Rendering
Please note that we use the GLFW context as default which does not support headless rendering. To allow for both, onscreen rendering & headless rendering on a remote server, set the context to EGL: 
```
export PYOPENGL_PLATFORM='egl'
```
In order to make the EGL context work, you might need to change PyOpenGL like [here](https://github.com/mcfletch/pyopengl/issues/27)

<!-- ## Preparatory Steps

*1. Pip installation*
```bash
pip install --user .
``` -->


## Citation
If you find Augmented Autoencoders useful for your research, please consider citing:  
```
@INPROCEEDINGS{9811890,
  author={Zheng, Linfang and Leonardis, Aleš and Tse, Tze Ho Elden and Horanyi, Nora and Chen, Hua and Zhang, Wei and Chang, Hyung Jin}, 
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={TP-AE: Temporally Primed 6D Object Pose Tracking with Auto-Encoders}, 
  year={2022}, 
  volume={}, 
  number={}, 
  pages={10616-10623}, 
  doi={10.1109/ICRA46639.2022.9811890}}

```

## Acknowledgement
Our implementation leverages the code from [AAE](https://github.com/DLR-RM/AugmentedAutoencoder) 