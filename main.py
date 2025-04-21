import cv2
import serial
import threading
import time
from utils.yolo2base_realsense import YOLO2Base
from config.params import CFG

def string_to_hex(input_str: str, sep: str = "", uppercase: bool = True, encoding: str = 'utf-8') -> str:
    """字符串转十六进制"""
    hex_bytes = []
    for byte in input_str.encode(encoding):
        fmt = "{:02X}" if uppercase else "{:02x}"
        hex_bytes.append(fmt.format(byte))
    return sep.join(hex_bytes)


class SerialManager:
    """串口管理类"""

    def __init__(self):
        try:
            self.ser = serial.Serial(
                port=CFG.serial_port,
                baudrate=CFG.baudrate,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=1
            )
            print(f"Serial port {CFG.serial_port} opened successfully")
        except Exception as e:
            print(f"Error opening serial port: {e}")
            self.ser = None

    def send_command(self, command):
        """发送命令"""
        if not self.ser or not self.ser.is_open:
            print("Serial port not available")
            return False

        try:
            # 关键修改：发送前清空缓冲区
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            hex_cmd = string_to_hex(command, sep=" ")
            self.ser.write(bytes.fromhex(hex_cmd.replace(" ", "")))
            self.ser.flush()
            print(f"Sent: {command.strip()} -> {hex_cmd}")
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False

    def read_latest_response(self, timeout=1.0, initial_delay=0.2):
        """响应处理（适配新格式）"""
        if not self.ser or not self.ser.is_open:
            return None

        try:
            time.sleep(initial_delay)
            original_timeout = self.ser.timeout
            self.ser.timeout = timeout - initial_delay

            response = b''
            start_time = time.time()
            while time.time() - start_time < (timeout - initial_delay):
                data = self.ser.read(self.ser.in_waiting or 1)
                if data:
                    response += data
                else:
                    time.sleep(0.01)

            if response:
                raw_str = response.decode('utf-8', errors='ignore').strip()
                print(f"原始响应: {raw_str}")

                # 新格式解析逻辑
                if raw_str.startswith("ok "):
                    parts = raw_str.split()
                    if len(parts) == 7:  # ok + 6个数值
                        try:
                            # 提取前5个坐标值
                            coordinates = [float(x) for x in parts[1:6]]

                            # 构造新指令
                            new_cmd = (
                                f"&{coordinates[0]:.2f},"
                                f"{coordinates[1]:.2f},"
                                f"{coordinates[2]:.2f},"
                                f"{coordinates[3]:.2f},"
                                f"{coordinates[4]:.2f},"
                                "-70\r\n"
                            )

                            # 发送加工后的指令
                            if self.send_command(new_cmd):
                                print(f"已发送坐标指令: {new_cmd.strip()}")
                            else:
                                print("指令发送失败")
                        except ValueError:
                            print("坐标值转换失败")
                    else:
                        print(f"无效数据长度，预期7个元素，实际收到{len(parts)}个")
                else:
                    print("非ok开头的响应")

                hex_response = ' '.join(f"{b:02X}" for b in response)
                return response
            return None
        except Exception as e:
            print(f"Read error: {e}")
            return None
        finally:
            self.ser.timeout = original_timeout

    def close(self):
        """关闭串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed")


def delayed_status_query(serial_mgr):
    """优化后的状态查询流程"""
    time.sleep(5)  # 主延迟

    # 发送指令前已自动清空缓冲区
    if serial_mgr.send_command("#GETJPOS\r\n"):
        print("已发送状态查询指令，等待响应...")
        # 读取最新数据（无需手动清空）
        response = serial_mgr.read_latest_response(
            timeout=1.0,
            initial_delay=0.2
        )
        if response:
            print(f"机械臂最新状态: {response.decode('utf-8', errors='ignore')}")
        else:
            print("状态查询超时，未收到有效响应")


def main():
    # 用户输入目标类别
    target_class = input("请输入需要抓取的物体类别（英文名称，如 'bottle'）: ").strip().lower()
    print(f"目标设置为: {target_class}")
    print("按空格键抓取目标物体，按Q键退出程序")

    # 初始化硬件连接
    serial_mgr = SerialManager()
    model_path = CFG.model_path
    detector = YOLO2Base(model_path)

    # 机械臂使能
    if serial_mgr.send_command("!START\r\n"):
        print("机械臂已使能")
    else:
        print("机械臂使能失败")

    try:
        # 启动检测线程
        detector.start()

        last_command_time = 0  # 控制命令发送频率

        # 主循环
        while detector.running:
            # 获取按键事件
            key = detector.get_key_pressed()

            # 退出程序
            if key == ord('q'):
                print("正在退出程序...")
                serial_mgr.send_command("!DISABLE\r\n")
                detector.stop()
                break

            elif key == ord('b'):
                current_time = time.time()
                # 防止命令发送过于频繁 (至少间隔1秒)
                if current_time - last_command_time < 1:
                    continue
                last_command_time = current_time
                print("正在返回起始点...")
                serial_mgr.send_command(f"{CFG.home_pose}\r\n")

            # 放下物品
            elif key == ord('p'):
                current_time = time.time()
                # 防止命令发送过于频繁 (至少间隔1秒)
                if current_time - last_command_time < 1:
                    continue
                last_command_time = current_time
                serial_mgr.send_command("&-96,-1,150,0,40,-70\r\n")
            elif key == ord('l'):
                current_time = time.time()
                # 防止命令发送过于频繁 (至少间隔1秒)
                if current_time - last_command_time < 1:
                    continue
                last_command_time = current_time
                serial_mgr.send_command("&-96,-1,150,0,40,0\r\n")


            # 处理空格键抓取事件
            elif key == ord(' '):
                current_time = time.time()
                # 防止命令发送过于频繁 (至少间隔1秒)
                if current_time - last_command_time < 1:
                    continue

                last_command_time = current_time
                print("尝试抓取目标...")


                # 获取当前坐标（线程安全）
                with detector.lock:
                    current_coords = detector.current_coordinates.copy()

                # 筛选目标类别坐标
                targets = [coord for coord in current_coords
                           if coord[0].lower() == target_class and coord[3] is not None]

                if targets:
                    # 取第一个检测到的目标
                    _, _, _, arm_xyz = targets[0]
                    command_str = f"@{arm_xyz[0]:.1f},{arm_xyz[1]:.1f},{arm_xyz[2]:.1f},180,0,180\r\n"
                    # command_str = f"@{196},{7},{122},180,32,-178\r\n"
                    if serial_mgr.send_command(command_str):
                        print(f"已发送抓取命令: 坐标 X:{arm_xyz[0]:.1f}, Y:{arm_xyz[1]:.1f}, Z:{arm_xyz[2]:.1f}")
                        # 启动延迟查询线程（新增功能）
                        threading.Thread(
                            target=delayed_status_query,
                            args=(serial_mgr,)
                        ).start()
                else:
                    print(f"未检测到 {target_class} 类别的有效目标")

            # 短暂休眠，减少CPU使用
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理资源
        print("正在清理资源...")
        detector.stop()
        serial_mgr.close()
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == '__main__':
    main()