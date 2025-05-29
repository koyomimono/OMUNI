import datetime
import math
import os
import threading
import time

from evdev import InputDevice, categorize, ecodes

SPHERE_RADIUS = 150

MOUSE_SENSOR_X1DISTANCE = 0.017692852
MOUSE_SENSOR_X2DISTANCE = 0.015048343

MOUSE_SENSOR_Y1DISTANCE = 0.016964967
# MOUSE_SENSOR_Y1DISTANCE =0.020932983
MOUSE_SENSOR_Y2DISTANCE = 0.018324415


class MouseTracker:
    def __init__(self, device_paths, k_angle):
        self.device_paths = device_paths
        self.mouse_positions = {i: (0, 0, 0) for i in range(1, len(device_paths) + 1)}
        self.track_movement = threading.Event()
        self.keep_running = True
        self.k_angle = k_angle
        self.angle_offset = [
            math.radians(0),
            math.radians(90),
            math.radians(180),
            math.radians(270),
        ]
        self.data_file = None
        self.file_lock = threading.Lock()

        # 角速度と角加速度計算用の変数
        self.prev_radians = (0, 0, 0)
        self.prev_angular_velocity = (0, 0, 0)
        self.prev_time = time.time()

    def toggle_tracking(self):
        if self.track_movement.is_set():
            self.track_movement.clear()
            print("Tracking stopped.")
            self.close_data_file()
        else:
            self.track_movement.set()
            print("Tracking started.")
            self.setup_data_file()

    def setup_data_file(self):
        directory = "/home/swarm/Documents/python_ANTAM"
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(directory, "Motor_30Hz.csv")
        self.data_file = open(filename, "w")
        self.data_file.write(
            "Timestamp,X,Y,Z,AngularVelocityX,AngularVelocityY,AngularVelocityZ,AngularAccelX,AngularAccelY,AngularAccelZ\n"
        )

    def close_data_file(self):
        if self.data_file:
            self.data_file.close()
            self.data_file = None

    def read_mouse(self, device_path, mouse_index):
        dev = InputDevice(device_path)
        x, y = 0, 0
        angle = self.angle_offset[mouse_index - 1]

        # デバイスに応じて距離定数を設定
        if "event12" in device_path:
            distance_factor = MOUSE_SENSOR_Y1DISTANCE
        elif "event16" in device_path:
            distance_factor = MOUSE_SENSOR_X1DISTANCE
        elif "event20" in device_path:
            distance_factor = MOUSE_SENSOR_Y2DISTANCE
        elif "event24" in device_path:
            distance_factor = MOUSE_SENSOR_X2DISTANCE
        else:
            distance_factor = 1  # 万一の場合のデフォルト値

        for event in dev.read_loop():
            if not self.keep_running:
                break
            if not self.track_movement.is_set():
                continue

            if event.type == ecodes.EV_REL:
                if event.code == ecodes.REL_X:
                    y += event.value
                elif event.code == ecodes.REL_Y:
                    x -= event.value

                get_point_x = y * math.sin(angle) * distance_factor
                get_point_y = y * math.cos(angle) * distance_factor
                get_point_z = x * 1 / math.cos(self.k_angle) * distance_factor

                self.mouse_positions[mouse_index] = (
                    get_point_x,
                    get_point_y,
                    get_point_z,
                )

    def calculate_average(self):
        while self.keep_running:
            if self.track_movement.is_set():
                average_x = sum(pos[0] for pos in self.mouse_positions.values()) / 2
                average_y = sum(pos[1] for pos in self.mouse_positions.values()) / 2
                average_z = sum(pos[2] for pos in self.mouse_positions.values()) / len(
                    self.mouse_positions
                )
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                radian_x = average_x / SPHERE_RADIUS
                radian_y = average_y / SPHERE_RADIUS
                radian_z = average_z / SPHERE_RADIUS

                # 時間差分と角速度の計算
                current_time = time.time()
                dt = current_time - self.prev_time
                angular_velocity_x = (radian_x - self.prev_radians[0]) / dt
                angular_velocity_y = (radian_y - self.prev_radians[1]) / dt
                angular_velocity_z = (radian_z - self.prev_radians[2]) / dt

                # 角加速度の計算
                angular_accel_x = (
                    angular_velocity_x - self.prev_angular_velocity[0]
                ) / dt
                angular_accel_y = (
                    angular_velocity_y - self.prev_angular_velocity[1]
                ) / dt
                angular_accel_z = (
                    angular_velocity_z - self.prev_angular_velocity[2]
                ) / dt

                # 状態の更新
                self.prev_radians = (radian_x, radian_y, radian_z)
                self.prev_angular_velocity = (
                    angular_velocity_x,
                    angular_velocity_y,
                    angular_velocity_z,
                )
                self.prev_time = current_time

                # データをファイルに記録
                with self.file_lock:
                    if self.data_file:
                        self.data_file.write(
                            f"{timestamp},{average_x},{average_y},{average_z},{angular_velocity_x},{angular_velocity_y},{angular_velocity_z},{angular_accel_x},{angular_accel_y},{angular_accel_z}\n"
                        )
                    print(
                        f"Average Position -> X: {average_x}, Y: {average_y}, Z: {average_z}"
                    )

            time.sleep(0.1)

    def start(self):
        threads = [
            threading.Thread(target=self.read_mouse, args=(path, idx + 1))
            for idx, path in enumerate(self.device_paths)
        ]
        threads.append(threading.Thread(target=self.calculate_average))
        for t in threads:
            t.start()
        return threads

    def stop(self):
        self.keep_running = False
        self.track_movement.clear()
        self.close_data_file()


if __name__ == "__main__":
    device_paths = [
        "/dev/input/event12",
        "/dev/input/event16",
        "/dev/input/event20",
        "/dev/input/event24",
    ]
    tracker = MouseTracker(device_paths, k_angle=math.radians(27))

    threads = tracker.start()
    try:
        while True:
            command = input("Enter 's' to start/stop tracking, 'q' to quit: ")
            if command.lower() == "s":
                tracker.toggle_tracking()
            elif command.lower() == "q":
                tracker.stop()
                break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
