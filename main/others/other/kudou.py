import cv2
import numpy as np
import time
import collections
from threading import Event
from roboclaw_3 import Roboclaw
import matplotlib.pyplot as plt

# RoboClawの初期化
roboclaw1 = Roboclaw("/dev/ttyACM0", 115200)
roboclaw2 = Roboclaw("/dev/ttyACM1", 115200)
roboclaw1.Open()
roboclaw2.Open()
address = 0x80  # RoboClawのアドレス

# 定数の設定
SPHERE_RADIUS = 150
SPHERE_DIAMETER = 300
CAMERA_INDEX = 0
PIXEL = 61.0 / 480.0  # mm/pixel
MAX_AREA = 5000  # pixel
MIN_AREA = 1500  # pixel
WEIGHT = 640
HEIGHT = 480
FPS = 120
LENGTH = 50
MAX_SPEED = 30
MIN_SPEED = -30
POSITION_VARIATION_THRESHOLD = 20
AREA_VARIATION_THRESHOLD = 1000
STABILITY_HISTORY_LENGTH = 10

INERTIAL_FORCE_X = -0.041552914  # acceletion
INERTIAL_FORCE_Y = -0.030096533  # acceletion
FRICTION_COEFFICIENT_X = 0.032979313  # velocity
FRICTION_COEFFICIENT_Y = 0.041094364  # velocity
#x,y axis
PARAMETER_P = 2.5
PARAMETER_I =0.95
PARAMETER_D = 0.12
PARAMETER_K = 3
#z axis


PARAMETER_A = 30

class AngleVelocityAcceleration:
    def __init__(self):
        self.prev_time = time.time()

class StableObjectDetector:
    def __init__(self):
        self.position_threshold = POSITION_VARIATION_THRESHOLD
        self.area_threshold = AREA_VARIATION_THRESHOLD
        self.history_length = STABILITY_HISTORY_LENGTH
        self.positions = collections.deque(maxlen=STABILITY_HISTORY_LENGTH)
        self.areas = collections.deque(maxlen=STABILITY_HISTORY_LENGTH)

    def add_detection(self, centroid, area):
        self.positions.append(centroid)
        self.areas.append(area)

    def is_stable(self):
        if len(self.positions) < self.history_length:
            return False
        position_variation = np.max(np.abs(np.diff(self.positions, axis=0)), axis=0)
        if np.any(position_variation > self.position_threshold):
            return False
        area_variation = max(self.areas) - min(self.areas)
        if area_variation > self.area_threshold:
            return False
        return True

def calculate_angle_from_center(error_x, error_y):
    angle_x = np.arctan2(error_x, SPHERE_RADIUS)
    angle_y = np.arctan2(error_y, SPHERE_RADIUS)
    return angle_x, angle_y

def calculate_control_period(last_control_time):
    current_time = time.time()
    time_diff = current_time - last_control_time
    last_control_time = current_time
    return time_diff, last_control_time

class ErrorAdjustment:
    def __init__(self, kp, kd):
        self.Kp = kp
        self.Kd = kd
        self.prev_error = np.array([0.0, 0.0])
        self.prev_velocity = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.integral = np.array([0.0,0.0])
        self.speeds = np.array([0, 0, 0, 0])

    def set_error_and_radian(self, error_x, error_y, theta, last_control_time,fps):
        
        self.error = np.array([error_x, error_y])
        self.velocity = self.error - self.prev_error
        self.acceleration = self.velocity - self.prev_velocity
        self.integral += (self.error + self.prev_error) / 2 * (1/fps)

        p_control_x = PARAMETER_P * self.error[0]
        p_control_y = PARAMETER_P * self.error[1]

        d_control_x = PARAMETER_D * self.velocity[0] / (1/fps)
        d_control_y = PARAMETER_D * self.velocity[1] / (1/fps)

        i_control_x = PARAMETER_I * self.integral[0]
        i_control_y = PARAMETER_I * self.integral[1]

        k_control_x = PARAMETER_K * self.acceleration[0]
        k_control_y = PARAMETER_K * self.acceleration[1]

     
    
        self.speeds[0] = (p_control_x + d_control_x + i_control_x- k_control_x ) 
        self.speeds[1] = - (p_control_x + d_control_x + i_control_x- k_control_x ) 
        self.speeds[2] = (p_control_y + d_control_y + i_control_y- k_control_y ) 
        self.speeds[3] = - (p_control_y + d_control_y + i_control_y- k_control_y ) 

        self.prev_error = self.error
        self.prev_velocity = self.velocity

        print(f"Speeds: {self.speeds}")
        print(f"theta : {theta}")

        return last_control_time

    def get_speeds(self):
        return self.speeds

def send_speed_to_motors(speeds):
    speeds = np.clip(speeds, MIN_SPEED, MAX_SPEED)
    roboclaw1.ForwardM1(address, abs(int(speeds[0]))) if speeds[0] >= 0 else roboclaw1.BackwardM1(address, abs(int(speeds[0])))
    roboclaw1.ForwardM2(address, abs(int(speeds[1]))) if speeds[1] >= 0 else roboclaw1.BackwardM2(address, abs(int(speeds[1])))
    roboclaw2.ForwardM1(address, abs(int(speeds[2]))) if speeds[2] >= 0 else roboclaw2.BackwardM1(address, abs(int(speeds[2])))
    roboclaw2.ForwardM2(address, abs(int(speeds[3]))) if speeds[3] >= 0 else roboclaw2.BackwardM2(address, abs(int(speeds[3])))

def stop_motors():
    send_speed_to_motors([0, 0, 0, 0])

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Camera could not be opened!")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap if cap.read()[0] else None

def calculate_centroid(binary_image):
    m = cv2.moments(binary_image, True)
    if m['m00'] != 0:
        x_center = m['m10'] / m['m00']
        y_center = m['m01'] / m['m00']
        return np.array([int(x_center), int(y_center)])
    return np.array([-1, -1])

def calculate_angle(m):
    mu20 = m['mu20'] / m['m00']
    mu02 = m['mu02'] / m['m00']
    mu11 = m['mu11'] / m['m00']
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    if theta < 0:
        theta += np.pi
    return theta

def calculate_distance_from_center(image_center, centroid):
    distance = centroid - image_center
    distance[1] = -distance[1]
    error = np.array([int(distance[0] * PIXEL), int(distance[1] * PIXEL)])
    return error[0], error[1]


def draw_picture(frame, centroid, radian, fps, elapsed_time, error_x, error_y):
    distance_text = f"error_(x, y): ({error_x}, {error_y})"
    #angle_deg = radian * 180 / np.pi
    #elapsed_time_text = f"Elapsed Time: {elapsed_time:.2f} s"
    fps_text = f"FPS: {fps:.0f}"
    image_center = (int(WEIGHT // 2), int(HEIGHT // 2))
    
    cv2.drawMarker(frame, image_center, (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=700, thickness=1, line_type=cv2.LINE_AA)
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, distance_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #cv2.putText(frame, f"Angle: {angle_deg:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #cv2.putText(frame, elapsed_time_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def calculate_fps(last_time):
    current_time = time.time()
    time_diff = current_time - last_time
    last_time = current_time
    return 1.0 / time_diff if time_diff > 0 else 0, last_time

def filter_color_and_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    return mask

def process_camera_feed():
    cap = initialize_camera()
    if cap is None:
        return

    image_center = np.array([WEIGHT // 2, HEIGHT // 2])
    last_time = time.time()
    start_time = time.time()
    last_control_time = time.time()

    error_speed_controller = ErrorAdjustment(0.0, 0.0)
    stable_detector = StableObjectDetector()

    while True :
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        mask = filter_color_and_brightness(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            m = cv2.moments(mask, True)
            area = m['m00']
            elapsed_time = time.time() - start_time

            if MIN_AREA < area < MAX_AREA:
                centroid = calculate_centroid(mask)
                if centroid[0] != -1 and centroid[1] != -1:
                    stable_detector.add_detection(centroid, area)
                    
                    if stable_detector.is_stable():
                        radian = calculate_angle(m)
                        error_x, error_y = calculate_distance_from_center(image_center, centroid)
                        fps, last_time = calculate_fps(last_time)
                        last_control_time = error_speed_controller.set_error_and_radian(error_x, error_y, radian, last_control_time,fps)
                        speeds = error_speed_controller.get_speeds()
                        send_speed_to_motors(speeds)
                        draw_picture(frame, centroid, radian, fps, elapsed_time, error_x, error_y)
            else:
                fps, last_time = calculate_fps(last_time)
                stop_motors()
                draw_picture(frame, np.array([-1, -1]), 0, fps, elapsed_time, 0, 0)

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            stop_motors()
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_camera_feed()