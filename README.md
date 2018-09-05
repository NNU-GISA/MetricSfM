# MetricSfM
A SfM + MSP framework for photogrammetry applications. The fold SfM contains source codes for sparse 3D reconstruction for both web-images and aerial-images. The fold MSP contains exe for dense reconstruction given the sparse result from the SfM.

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
1. build the project in fold SfM with Cmake (make sure no lib linking problems)
2. put all the testing images into a fold
3. run test_sfm.cc

Usage of MSP:
---
1. Open MSP.exe in CMD at the MSP.exe folder. It will create a xml file named ms_proc_control_file.xml in the same fold
2. Open ms_proc_control_file.xml, change the absolute file path of Ori_QIN.qin in the folder named undistorted_Img and set an output folder for the results. 
3. In the CMD, run the command: MSP c:\...\...\...\ ms_proc_control_file.xml (absolute path).
4. The results will be in the Output_Working_Directory.

SfM Results:
---
1. Website image
![image](https://github.com/xiaohulugo/images/blob/master/web_result.jpg)

2. Aerial image
![image](https://github.com/xiaohulugo/images/blob/master/aerial_result.jpg)

3. Street image
![image](https://github.com/xiaohulugo/images/blob/master/street_result.jpg)

MSP Results:
---
1. Aerial image
![image](https://github.com/xiaohulugo/images/blob/master/msp1.bmp)
![image](https://github.com/xiaohulugo/images/blob/master/msp2.bmp)
