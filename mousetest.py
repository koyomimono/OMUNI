import time
from mouse_tracking import MouseTracker

# ===== 디바이스 설정 =====
DEV_XY = "/dev/input/event9"
DEV_Z  = "/dev/input/event14"

SCALING_XY = 0.0172   # 기존 main.py 값
SCALING_Z  = 1.0

mouse_x = 0.0
mouse_y = 0.0
mouse_z = 0.0

# ===== 콜백 =====
def xy_callback(x, y):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

def z_callback(x, y):
    global mouse_z
    mouse_z = x   # ★ Z축을 x방향 움직임으로 사용

# ===== 메인 =====
def main():
    print("🖱 Mouse XYZ Monitor")
    print(" event9  -> X,Y")
    print(" event14 -> Z (REL_X)")
    print(" Ctrl+C 로 종료\n")

    mt_xy = MouseTracker(DEV_XY, SCALING_XY)
    mt_z  = MouseTracker(DEV_Z,  SCALING_Z)

    mt_xy.start(callback=xy_callback)
    mt_z.start(callback=z_callback)

    try:
        while True:
            print(
                f"X={mouse_x:+8.3f} mm | "
                f"Y={mouse_y:+8.3f} mm | "
                f"Z={mouse_z:+8.3f}"
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료")

    finally:
        mt_xy.stop()
        mt_z.stop()

if __name__ == "__main__":
    main()
