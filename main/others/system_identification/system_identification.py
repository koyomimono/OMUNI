import numpy as np
import cv2
from roboclaw_3 import Roboclaw
import csv
import time
import threading
import os
import datetime
from evdev import InputDevice, categorize, ecodes
import math

# RoboClawの初期化
roboclaw1 = Roboclaw("/dev/ttyACM0", 115200)
roboclaw2 = Roboclaw("/dev/ttyACM1", 115200)
roboclaw1.Open()
roboclaw2.Open()
address = 0x80  # RoboClawのアドレス

# グローバル変数
SPHERE_RADIUS = 150
MOUSE_SENSOR_X1DISTANCE = 0.017692852
MOUSE_SENSOR_X2DISTANCE = 0.015048343
MOUSE_SENSOR_Y1DISTANCE = 0.016964967
MOUSE_SENSOR_Y2DISTANCE = 0.018324415
motor_control_active = True
data_lock = threading.Lock()  # データ共有用のロック
current_speeds = [0, 0, 0, 0]  # [speed_x1, speed_x2, speed_y1, speed_y2]

# モータ速度を送信する関数
def send_speed_to_motors(speed_x1, speed_x2, speed_y1, speed_y2):
    global current_speeds
    with data_lock:
        current_speeds = [speed_x1, speed_x2, speed_y1, speed_y2]

    if speed_x1 >= 0:
        roboclaw1.ForwardM1(address, speed_x1)
    else:
        roboclaw1.BackwardM1(address, abs(speed_x1))
    
    if speed_x2 >= 0:
        roboclaw1.ForwardM2(address, speed_x2)
    else:
        roboclaw1.BackwardM2(address, abs(speed_x2))
    
    if speed_y1 >= 0:
        roboclaw2.ForwardM1(address, speed_y1)
    else:
        roboclaw2.BackwardM1(address, abs(speed_y1))
    
    if speed_y2 >= 0:
        roboclaw2.ForwardM2(address, speed_y2)
    else:
        roboclaw2.BackwardM2(address, abs(speed_y2))

# モータを停止する関数
def stop_motors():
    send_speed_to_motors(0, 0, 0, 0)

# モータ制御用スレッド
def motor_control_thread():
    global motor_control_active
    try:
        with open("random_data.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # ヘッダー行をスキップ
            
            for row in reader:
                speed_x1, speed_x2, speed_y1, speed_y2 = [int(value) * 35 for value in row]
                send_speed_to_motors(speed_x1, speed_x2, speed_y1, speed_y2)
                time.sleep(1)  # 30Hz

                if not motor_control_active:
                    break
    finally:
        motor_control_active = False
        stop_motors()
        print("Motor control finished.")

# 角速度と角加速度の記録用クラス
class MouseTracker:
    def __init__(self, device_paths, k_angle):
        self.device_paths = device_paths
        self.mouse_positions = {i: (0, 0, 0) for i in range(1, len(device_paths) + 1)}
        self.k_angle = k_angle  
        self.angle_offset = [math.radians(0), math.radians(90), math.radians(180), math.radians(270)]
        self.prev_radians = (0, 0, 0)
        self.prev_angular_velocity = (0, 0, 0)
        self.prev_time = time.time()

        directory = os.path.expanduser("~/Documents/python_ANTAM")
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(directory, f"BT.csv")
        self.data_file = open(filename, 'w')
        #self.data_file.write("col1,col2,col3,col4,col5,speed\n")
        self.file_lock = threading.Lock()
        print(f"Data file created: {filename}")

    def close_data_file(self):
        if self.data_file:
            self.data_file.close()
            print("Data file closed.")

    def read_mouse(self, device_path, mouse_index):
        dev = InputDevice(device_path)
        x, y = 0, 0
        angle = self.angle_offset[mouse_index - 1]

        if "event8" in device_path:
            distance_factor = MOUSE_SENSOR_Y1DISTANCE
        elif "event12" in device_path:
            distance_factor = MOUSE_SENSOR_X1DISTANCE
        elif "event16" in device_path:
            distance_factor = MOUSE_SENSOR_Y2DISTANCE
        elif "event20" in device_path:
            distance_factor = MOUSE_SENSOR_X2DISTANCE
        else:
            distance_factor = 1

        for event in dev.read_loop():
            if not motor_control_active:
                break
            if event.type == ecodes.EV_REL:
                if event.code == ecodes.REL_X:
                    y += event.value
                elif event.code == ecodes.REL_Y:
                    x -= event.value

                get_point_x = y * math.sin(angle) * distance_factor
                get_point_y = y * math.cos(angle) * distance_factor
                get_point_z = x * 1 / math.cos(self.k_angle) * distance_factor

                self.mouse_positions[mouse_index] = (get_point_x, get_point_y, get_point_z)

    def calculate_average(self):
        global motor_control_active
        while motor_control_active:
            average_x = sum(pos[0] for pos in self.mouse_positions.values()) /2
            average_y = sum(pos[1] for pos in self.mouse_positions.values()) /2
            average_z = sum(pos[2] for pos in self.mouse_positions.values()) / len(self.mouse_positions)

            radian_x = average_x / SPHERE_RADIUS
            radian_y = average_y / SPHERE_RADIUS
            radian_z = average_z / SPHERE_RADIUS

            current_time = time.time()
            dt = current_time - self.prev_time
            angular_velocity_x = (radian_x - self.prev_radians[0]) / dt
            angular_velocity_y = (radian_y - self.prev_radians[1]) / dt

            angular_accel_x = (angular_velocity_x - self.prev_angular_velocity[0]) / dt
            angular_accel_y = (angular_velocity_y - self.prev_angular_velocity[1]) / dt
            angular_accel_z = (radian_z - self.prev_radians[2]) / dt

            self.prev_radians = (radian_x, radian_y, radian_z)
            self.prev_angular_velocity = (angular_velocity_x, angular_velocity_y, 0)
            self.prev_time = current_time

            # Calculate speed_x, speed_y, and speed_z
            with data_lock:
                speed_x = current_speeds[0] - current_speeds[1]
                speed_y = current_speeds[2] - current_speeds[3]
                speed_z = sum(current_speeds)

            with self.file_lock:
                if self.data_file:
                    # データを指定形式で保存
                    self.data_file.write(f"{angular_accel_x},0,0,{angular_velocity_x},0,{speed_x}\n")
                    self.data_file.write(f"0,{angular_accel_y},0,0,{angular_velocity_y},{speed_y}\n")
                    self.data_file.write(f"0,0,{angular_accel_z},0,0,{speed_z}\n")
                    self.data_file.flush()

            time.sleep(1 )
        self.close_data_file()

    def start(self):
        threads = [threading.Thread(target=self.read_mouse, args=(path, idx+1)) for idx, path in enumerate(self.device_paths)]
        threads.append(threading.Thread(target=self.calculate_average))
        for t in threads:
            t.start()
        return threads

if __name__ == "__main__":
    motor_thread = threading.Thread(target=motor_control_thread)
    motor_thread.start()

    device_paths = ["/dev/input/event8", "/dev/input/event12", "/dev/input/event16", "/dev/input/event20"]
    tracker = MouseTracker(device_paths, k_angle=math.radians(27))
    mouse_threads = tracker.start()

    motor_thread.join()
    for t in mouse_threads:
        t.join()

    print("Data recording and motor control completed.")
