import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import threading
from config.params import CFG


class YOLO2Base:
    def __init__(self, model_path, hand_eye_matrix=None):
        """
        初始化类参数和硬件连接
        :param model_path: YOLO模型文件路径
        :param hand_eye_matrix: 手眼标定矩阵（4x4齐次矩阵）
        """
        # 手眼矩阵（机械臂坐标系转换矩阵）
        self.D = hand_eye_matrix if hand_eye_matrix is not None else np.array([
            [0.06029276,   0.80633451,  -0.58837866, 419.90258661],
            [0.97866863,  -0.16373115,  -0.12409601,  32.08943583],
            [-0.19639882,  -0.56834565,  -0.79900609, 298.82754985],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        # 相机参数
        self.cam_width = CFG.cam_width
        self.cam_height = CFG.cam_height
        self.cam_fps = CFG.cam_fps


        # RealSense 初始化
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.cam_width, self.cam_height, rs.format.z16, self.cam_fps)
        config.enable_stream(rs.stream.color, self.cam_width, self.cam_height, rs.format.bgr8, self.cam_fps)

        try:
            self.pipeline.start(config)
        except Exception as e:
            raise RuntimeError(f"Camera initialization failed: {e}")

        # 对齐工具和YOLO模型
        self.align = rs.align(rs.stream.color)
        self.model = YOLO(model_path)

        # 多线程相关
        self.lock = threading.Lock()
        self.current_coordinates = []
        self.running = False
        self.detection_thread = None

        # 键盘事件处理
        self.key_pressed = -1

    def __del__(self):
        """析构函数确保释放资源"""
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        cv2.destroyAllWindows()

    def get_aligned_frames(self):
        """
        获取对齐的深度和彩色帧
        :return: (depth_intrinsics, depth_frame, color_image)
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())
        return depth_intri, depth_frame, color_image

    def detect_objects(self, color_image):
        """
        执行YOLO目标检测
        :param color_image: 输入BGR图像
        :return: YOLO检测结果
        """
        confidence = CFG.conf_thres
        iou = CFG.iou_thres
        return self.model.predict(color_image, conf=confidence, iou=iou, verbose=False)

    def convert_coordinates(self, detection_results, depth_intri, depth_frame):
        """
        坐标转换处理
        :return: 包含类别的坐标列表 (class_name, ux, uy, arm_xyz_mm)
        """
        coord_list = []
        for result in detection_results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                class_name = self.model.names[cls_id]
                box = boxes.xywh[i].cpu().numpy()
                ux, uy = int(box[0]), int(box[1])

                depth = depth_frame.get_distance(ux, uy)
                if depth <= 0:
                    coord_list.append((class_name, ux, uy, None))
                    continue

                # 坐标转换流程
                camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intri, [ux, uy], depth)
                camera_xyz = np.append(camera_xyz, 1)  # 转换为齐次坐标

                # 应用手眼矩阵（调整单位到米）
                D_corrected = self.D.copy()
                D_corrected[:3, 3] /= 1000.0
                arm_xyz = D_corrected @ camera_xyz
                arm_xyz_mm = np.round(arm_xyz[:3] * 1000, 2)

                coord_list.append((class_name, ux, uy, arm_xyz_mm))
        return coord_list

    def visualize_results(self, image, coordinates):
        """
        可视化检测结果
        :return: 可视化后的图像
        """
        display_image = image.copy()
        for class_name, ux, uy, arm_xyz in coordinates:
            if arm_xyz is not None:
                # 绘制中心点
                cv2.circle(display_image, (ux, uy), 5, (0, 0, 255), -1)
                # 显示坐标文本
                text = f"{class_name}: X:{arm_xyz[0]:.1f}, Y:{arm_xyz[1]:.1f}, Z:{arm_xyz[2]:.1f}"
                cv2.putText(display_image, text, (ux + 10, uy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return display_image

    def run_detection(self):
        """检测线程主循环"""
        self.running = True
        try:
            while self.running:
                depth_intri, depth_frame, color_image = self.get_aligned_frames()
                results = self.detect_objects(color_image)
                coordinates = self.convert_coordinates(results, depth_intri, depth_frame)

                # 更新共享数据
                with self.lock:
                    self.current_coordinates = coordinates

                # 可视化处理
                visualized_image = self.visualize_results(results[0].plot(), coordinates)

                # 显示操作提示
                cv2.putText(visualized_image, "Press SPACE to grab target | Press Q to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("RealSense Detection", visualized_image)
                self.key_pressed = cv2.waitKey(1)

                if self.key_pressed == ord('q'):
                    self.running = False
        finally:
            self.__del__()

    def get_key_pressed(self):
        """获取当前按下的键"""
        key = self.key_pressed
        self.key_pressed = -1  # 重置按键状态
        return key

    def start(self):
        """启动检测线程"""
        self.detection_thread = threading.Thread(target=self.run_detection)
        self.detection_thread.daemon = True  # 设置为守护线程，主线程结束时自动结束
        self.detection_thread.start()

    def stop(self):
        """停止检测线程"""
        self.running = False
        if self.detection_thread is not None:
            self.detection_thread.join()