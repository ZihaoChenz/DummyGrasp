import argparse
import ast
import numpy as np


def get_config():
    """获取全局配置参数"""
    parser = argparse.ArgumentParser(description="Dummy Parameters")

    # 相机参数组
    cam_group = parser.add_argument_group("Camera")
    cam_group.add_argument("--cam_width", type=int, default=640,
                           help="相机分辨率宽度")
    cam_group.add_argument("--cam_height", type=int, default=480,
                           help="相机分辨率高度")
    cam_group.add_argument("--cam_fps", type=int, default=30,
                           help="相机帧率")

    # 模型参数组
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model_path", type=str,
                             default="model/yolov8n.pt",
                             help="YOLO模型文件路径")
    model_group.add_argument("--conf_thres", type=float, default=0.5,
                             help="检测置信度阈值")
    model_group.add_argument("--iou_thres", type=float, default=0.4,
                             help="检测IOU阈值")

    # 机械臂参数组
    robot_group = parser.add_argument_group("Robot")
    robot_group.add_argument("--home_pose", type=str,
                             default="&0,-75,180,0,0,0",
                             help="机械臂初始位姿")

    # 串口参数组
    serial_group = parser.add_argument_group("Serial")
    serial_group.add_argument("--serial_port", type=str, default="COM6",
                              help="串口号")
    serial_group.add_argument("--baudrate", type=int, default=115200,
                              help="波特率")

    args = parser.parse_args()
    return args
# 全局配置实例
CFG = get_config()