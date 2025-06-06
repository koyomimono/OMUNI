from roboclaw_3 import Roboclaw
import time

PORT = "/dev/ttyACM0"  # Change this to your actual port
BAUD = 115200
ADDRESS = 0x80

roboclaw = Roboclaw(PORT, BAUD)

try:
    print("[INFO] Opening port...")
    roboclaw.Open()
    print("[INFO] Port opened.")
except Exception as e:
    print("[ERROR] roboclaw.Open() failed:", e)

# Check if _port exists
if hasattr(roboclaw, "_port"):
    print("[INFO] _port exists.")
else:
    print("[ERROR] _port not found after Open(). Check roboclaw_3.py.")
    exit()

try:
    roboclaw.ForwardM1(ADDRESS, 50)
    time.sleep(2)
    roboclaw.ForwardM1(ADDRESS, 0)
    print("[INFO] Motor command sent.")
except Exception as e:
    print("[ERROR] Motor control failed:", e)
