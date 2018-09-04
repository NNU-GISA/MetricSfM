# MetricSfM
A SfM framework for photogrammetry applications. The fold SfM contains source codes for sparse 3D reconstruction for both web-images and aerial-images. The fold MSP contains exe for dense reconstruction given the sparse result from the SfM.

Framework:
---
![image](https://github.com/xiaohulugo/images/blob/master/sfm_framework.png)

Prerequisites:
---
1. OpenCV > 2.4.x (tested on OpenCV2.4.16)
2. Boost
3. Ceres (added the libs in to thirdparty/ceres-1.13/lib)
4. Cuda
5. Other libs are included in thirdparty

Usage of SfM:
---
1. build the project in fold SfM with Cmake
2. put all the input images into a fold
3. run test_sfm 

Usage of MSP:
---
