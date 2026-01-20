# step_response_axis_step.py
# - X / Y / Z 축을 모드에 따라 하나씩 스텝입력
# - + / - 방향 선택 가능
# - t < STEP_TIME: 0, t >= STEP_TIME: 선택 축만 일정 속도 스텝
# - main.py 와 동일한 카메라 설정 (V4L2 + YUYV + 중앙 정사각 크롭)
# - MouseTracker:
#       /dev/input/event9 → x, y (이동량/위치)
#       /dev/input/event14 → z (회전 센서, 여기서는 y값만 사용)
# - CSV 파일 이름: 오무니스텝반응_YYYYMMDD_HHMMSS.csv

import time
import csv
import math
from datetime import datetime

import cv2
import numpy as np

from mouse_tracking import MouseTracker
from scr.roboclaw_motor_library import motor_m1, motor_m2, motor_m3, stop_all

# ===============================
# 실험 파라미터
# ===============================
RUN_DURATION = 5.0    # 전체 실험 시간 [s]
STEP_TIME    = 1.0    # 스텝이 걸리는 시각 [s]

# 저장 파일 이름 = 오무니스텝반응_실험시각.csv
LOG_FILENAME = f"omunistep_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# --- 축별 스텝 크기 (필요하면 여기 값만 조정해서 사용) ---
# X/Y : mm/s,  Z : rad/s
X_STEP_VEL   = 200.0                    # X축 스텝 속도 [mm/s]
Y_STEP_VEL   = 200.0                    # Y축 스텝 속도 [mm/s]
Z_STEP_OMEGA = math.radians(200.0)      # Z축 스텝 속도 [rad/s]

# ===============================
# 카메라 설정 (main.py 와 같은 계열)
# ===============================
CAMERA_INDEX = 0
WIDTH  = 640
HEIGHT = 480
FPS_TARGET = 60
WAIT = 1

# 중앙 정사각형 크롭
CROP_LEFT  = (WIDTH - HEIGHT) // 2
CROP_RIGHT = CROP_LEFT + HEIGHT

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    cv2.namedWindow("StepTest", flags=cv2.WINDOW_GUI_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS_TARGET)
    # main.py 와 동일하게 YUYV 사용
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None

    return cap

# ===============================
# 휠 할당 (main.py 기반)
# ===============================
THETA    = np.radians([90.0, 120.0, 240.0])
DIR_SGN  = np.array([+1, +1, +1], dtype=float)
R_SPIN   = 10.0
CMD_MAX  = 80
SPEED_TO_CMD = 0.5

def wheels_command_from_v(vx, vy, omega):
    """
    vx, vy [mm/s], omega [rad/s] -> 3개 휠 속도 -> 모터 명령
    """
    v_wheels = (-vx * np.sin(THETA) + vy * np.cos(THETA)) * DIR_SGN
    cmds = np.clip(v_wheels * SPEED_TO_CMD, -CMD_MAX, CMD_MAX).astype(int)
    return cmds, v_wheels

def move_motors_cmds(cmds, omega):
    """
    main.py 와 동일한 조합:
    M1 <- wheel3 + R_SPIN*omega
    M2 <- wheel2 - R_SPIN*omega
    M3 <- wheel1 + R_SPIN*omega
    """
    m1 = int(cmds[2] + R_SPIN * omega)
    m2 = int(cmds[1] - R_SPIN * omega)
    m3 = int(cmds[0] + R_SPIN * omega)

    motor_m1(m1)
    motor_m2(m2)
    motor_m3(m3)

# ===============================
# MouseTracker (event9 → x,y, event14 → z)
# ===============================
MOUSE_DEV_XY = "/dev/input/event6"
MOUSE_DEV_Z  = "/dev/input/event10"
SCALING_XY   = 0.0172
   # mm 단위 (main.py 와 동일 가정)
SCALING_Z    = 0.0224   # 회전 센서 스케일(원하면 나중에 조정)

mouse_x = 0.0
mouse_y = 0.0
mouse_z = 0.0

def mouse_xy_callback(x, y):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

def mouse_z_callback(x, y):
    # event14의 x축 움직임을 z축으로 사용
    global mouse_z
    mouse_z = x


# ===============================
# 유틸
# ===============================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ===============================
# 메인
# ===============================
def main():
    global mouse_x, mouse_y, mouse_z

    cap = initialize_camera()
    if not cap:
        print("❌ カメラが開けない．")
        return

    # ---- MouseTracker 시작 ----
    mouse_tracker_xy = None
    mouse_tracker_z  = None

    try:
        mouse_tracker_xy = MouseTracker(MOUSE_DEV_XY, SCALING_XY)
        mouse_tracker_xy.start(callback=mouse_xy_callback)
        print(f"✅ MouseTracker XY start: {MOUSE_DEV_XY}")
    except Exception as e:
        print(f"⚠ XY 마우스 센서 시작 실패: {e}")

    try:
        mouse_tracker_z = MouseTracker(MOUSE_DEV_Z, SCALING_Z)
        mouse_tracker_z.start(callback=mouse_z_callback)
        print(f"✅ MouseTracker Z start: {MOUSE_DEV_Z}")
    except Exception as e:
        print(f"⚠ Z 마우스 센서 시작 실패: {e}")

    print("📌 축별 스텝 응답 테스트 시작")
    print("📁 로그 파일:", LOG_FILENAME)
    print("--------- モード選択 ---------")
    print("  1 : X+  ( +X 방향 스텝 )")
    print("  2 : X-  ( -X 방향 스텝 )")
    print("  3 : Y+  ( +Y 방향 스텝 )")
    print("  4 : Y-  ( -Y 방향 스텝 )")
    print("  5 : Z+  ( +Z 방향 스텝 )")
    print("  6 : Z-  ( -Z 방향 스텝 )")
    print("  q : 종료")
    print("------------------------------")

    # 모드 선택 단계
    mode = None
    axis = None
    direction = 0  # +1 or -1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패 (모드 선택 단계)")
            cap.release()
            return

        frame_cropped = frame[:, CROP_LEFT:CROP_RIGHT]
        cv2.putText(
            frame_cropped,
            "Select: 1:X+ 2:X- 3:Y+ 4:Y- 5:Z+ 6:Z- (q: quit)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.imshow("StepTest", frame_cropped)

        key = cv2.waitKey(WAIT) & 0xFF
        if key == ord("1"):
            mode = "X+"
            axis = "x"
            direction = +1
            break
        elif key == ord("2"):
            mode = "X-"
            axis = "x"
            direction = -1
            break
        elif key == ord("3"):
            mode = "Y+"
            axis = "y"
            direction = +1
            break
        elif key == ord("4"):
            mode = "Y-"
            axis = "y"
            direction = -1
            break
        elif key == ord("5"):
            mode = "Z+"
            axis = "z"
            direction = +1
            break
        elif key == ord("6"):
            mode = "Z-"
            axis = "z"
            direction = -1
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("사용자 종료 (모드 선택 단계)")
            return

    print(f"✅ 선택된 모드: {mode} (axis={axis}, dir={direction:+d})")
    print(f"⏱ {RUN_DURATION:.1f} s 동안 STEP_TIME={STEP_TIME:.1f} s 에서 스텝 적용")
    print(f"📁 CSV: {LOG_FILENAME}")

    # CSV 헤더
    with open(LOG_FILENAME, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t[s]",
            "axis", "direction",          # X/Y/Z, +1/-1
            "vx_ref", "vy_ref", "omega_ref",
            "mouse_x", "mouse_y", "mouse_z",
            "cmd1", "cmd2", "cmd3",
        ])

    start_time = time.time()
    prev_time = start_time

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed >= RUN_DURATION:
                print("⏱ 실험 종료")
                break

            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임 읽기 실패 (실험 단계)")
                break

            dt = clamp(now - prev_time, 1e-3, 0.1)
            prev_time = now
            fps = 1.0 / dt

            # =======================
            # 1) 참조 속도 생성 (STEP 입력)
            # =======================
            if elapsed < STEP_TIME:
                vx_ref = 0.0
                vy_ref = 0.0
                omega_ref = 0.0
            else:
                if axis == "x":
                    vx_ref = direction * X_STEP_VEL
                    vy_ref = 0.0
                    omega_ref = 0.0
                elif axis == "y":
                    vx_ref = 0.0
                    vy_ref = direction * Y_STEP_VEL
                    omega_ref = 0.0
                elif axis == "z":
                    vx_ref = 0.0
                    vy_ref = 0.0
                    omega_ref = direction * Z_STEP_OMEGA
                else:
                    vx_ref = vy_ref = omega_ref = 0.0  # safety

            # =======================
            # 2) 휠 명령 생성 및 모터 구동
            # =======================
            cmds, vwh = wheels_command_from_v(vx_ref, vy_ref, omega_ref)
            move_motors_cmds(cmds, omega_ref)

            # =======================
            # 3) 화면 표시
            # =======================
            frame_cropped = frame[:, CROP_LEFT:CROP_RIGHT]

            text1 = f"Mode: {mode}   t={elapsed:5.2f}s  FPS={fps:4.1f}"
            text2 = f"vx_ref={vx_ref:+.1f} mm/s,  vy_ref={vy_ref:+.1f} mm/s"
            text3 = f"wz_ref={math.degrees(omega_ref):+.1f} deg/s"
            text4 = f"cmds = [{int(cmds[0]):+d}, {int(cmds[1]):+d}, {int(cmds[2]):+d}]"
            text5 = f"mouse_xy=({mouse_x:+.2f}, {mouse_y:+.2f}),  mouse_z={mouse_z:+.2f}"

            cv2.putText(frame_cropped, text1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_cropped, text2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame_cropped, text3, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(frame_cropped, text4, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_cropped, text5, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)

            if elapsed < STEP_TIME:
                cv2.putText(frame_cropped, "PRE-STEP", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame_cropped, "STEP APPLIED", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

            cv2.imshow("StepTest", frame_cropped)

            # =======================
            # 4) CSV 로깅
            # =======================
            with open(LOG_FILENAME, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{elapsed:.6f}",
                    axis, direction,
                    f"{vx_ref:.6f}", f"{vy_ref:.6f}", f"{omega_ref:.6f}",
                    f"{mouse_x:.6f}", f"{mouse_y:.6f}", f"{mouse_z:.6f}",
                    int(cmds[0]), int(cmds[1]), int(cmds[2]),
                ])

            key = cv2.waitKey(WAIT) & 0xFF
            if key == ord("q"):
                print("사용자 종료(q)")
                break

    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        if mouse_tracker_xy is not None:
            mouse_tracker_xy.stop()
        if mouse_tracker_z is not None:
            mouse_tracker_z.stop()
        print("🔚 모터 정지, 카메라 및 센서 해제 완료")


if __name__ == "__main__":
    main()
