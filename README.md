# Dummy机械臂物体抓取项目

## 项目概述
本项目基于YOLOv8目标检测算法和深度视觉信息，实现Dummy机械臂的物体抓取功能。系统通过Intel RealSense D435深度相机获取目标物体的三维坐标信息，完成定位与抓取操作。

## 硬件需求
- Dummy机械臂（型号请根据实际情况修改）
- Intel RealSense D435深度相机
- 支持USB 3.0的计算机

## 软件依赖
```python
Python >= 3.8
PyTorch >= 1.8
ultralytics (YOLOv8)
pyrealsense2
opencv-python
numpy

