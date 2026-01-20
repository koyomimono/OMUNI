import serial, time

# ----- 환경에 맞게 포트 확인 -----
PORT = '/dev/ttyACM0'   # arduino-cli board list 로 확인
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # 아두이노 리셋 대기

ser.write(b'1')  # 개행 없이 '1'만 전송
ser.flush()

print("Sent: 1 -> Arduino")
ser.close()
