from scr.roboclaw_3 import Roboclaw
from time import sleep

# ★ RoboClaw 보드 주소 (기본 0x80 = 128)
ADDRESS  = 0x80
BAUDRATE = 38400          # Motion Studio에서 바꿔놓지 않았다면 보통 38400

PORT = "/dev/ttyACM2"     # 위에서 ls /dev/ttyACM* 로 확인한 값으로 바꿔

rc = Roboclaw(PORT, BAUDRATE)
rc.Open()

# 1) 펌웨어 버전 읽기 시도
ok, ver = rc.ReadVersion(ADDRESS)
print("ReadVersion:", ok, ver)

# 2) 모터 간단 구동
print("M1 forward 20")
rc.ForwardM1(ADDRESS, 20)
sleep(2)

print("stop")
rc.ForwardM1(ADDRESS, 0)
