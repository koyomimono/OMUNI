# roboclaw_motor_library.py
from scr.roboclaw_3 import Roboclaw

# 定数
ADDRESS = 0x80
MAX_SPEED = 30
MIN_SPEED = -30

# RoboClaw初期化（グローバルに保持）
roboclaw1 = Roboclaw("/dev/ttyACM0", 115200)
roboclaw2 = Roboclaw("/dev/ttyACM1", 115200)
roboclaw1.Open()
roboclaw2.Open()

def clamp_speed(v):
    """速度をMIN/MAX範囲に制限"""
    return max(MIN_SPEED, min(MAX_SPEED, v))

def set_z(v):
    """Z軸方向移動"""
    v = clamp_speed(v)
    if v >= 0:
        roboclaw1.ForwardM1(ADDRESS, v)
        roboclaw1.ForwardM2(ADDRESS, v)
        roboclaw2.ForwardM1(ADDRESS, v)
        roboclaw2.ForwardM2(ADDRESS, v)
    else:
        v = abs(v)
        roboclaw1.BackwardM1(ADDRESS, v)
        roboclaw1.BackwardM2(ADDRESS, v)
        roboclaw2.BackwardM1(ADDRESS, v)
        roboclaw2.BackwardM2(ADDRESS, v)

def set_x(v):
    """X軸方向移動"""
    v = clamp_speed(v)
    if v >= 0:
        roboclaw1.ForwardM1(ADDRESS, v)
        roboclaw1.ForwardM2(ADDRESS, v)
        roboclaw2.BackwardM1(ADDRESS, v)
        roboclaw2.BackwardM2(ADDRESS, v)
    else:
        v = abs(v)
        roboclaw1.BackwardM1(ADDRESS, v)
        roboclaw1.BackwardM2(ADDRESS, v)
        roboclaw2.ForwardM1(ADDRESS, v)
        roboclaw2.ForwardM2(ADDRESS, v)

def set_y(v):
    """Y軸方向移動"""
    v = clamp_speed(v)
    if v >= 0:
        roboclaw1.BackwardM1(ADDRESS, v)
        roboclaw1.ForwardM2(ADDRESS, v)
        roboclaw2.BackwardM1(ADDRESS, v)
        roboclaw2.ForwardM2(ADDRESS, v)
    else:
        v = abs(v)
        roboclaw1.ForwardM1(ADDRESS, v)
        roboclaw1.BackwardM2(ADDRESS, v)
        roboclaw2.ForwardM1(ADDRESS, v)
        roboclaw2.BackwardM2(ADDRESS, v)

def stop_all():
    """すべてのモーターを停止"""
    roboclaw1.ForwardM1(ADDRESS, 0)
    roboclaw1.ForwardM2(ADDRESS, 0)
    roboclaw2.ForwardM1(ADDRESS, 0)
    roboclaw2.ForwardM2(ADDRESS, 0)

def motor_m1(speed):
    """モーター1を制御"""
    speed = clamp_speed(speed)
    if speed >= 0:
        roboclaw1.ForwardM2(ADDRESS, speed)
    else:
        roboclaw1.BackwardM2(ADDRESS, abs(speed))

def motor_m2(speed):
    """モーター2を制御"""
    speed = clamp_speed(speed)
    if speed >= 0:
        roboclaw2.BackwardM2(ADDRESS, speed)
    else:
        roboclaw2.ForwardM2(ADDRESS, abs(speed))

def motor_m3(speed):
    """モーター3を制御"""
    speed = clamp_speed(speed)
    if speed >= 0:
        roboclaw2.ForwardM1(ADDRESS, speed)
    else:
        roboclaw2.BackwardM1(ADDRESS, abs(speed))

def motor_m4(speed):
    """モーター4を制御"""
    speed = clamp_speed(speed)
    if speed >= 0:
        roboclaw1.BackwardM1(ADDRESS, speed)
    else:
        roboclaw1.ForwardM1(ADDRESS, abs(speed))
