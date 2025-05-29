import numpy as np
import cv2
from roboclaw_3 import Roboclaw
import csv
import time

# RoboClawの初期化
roboclaw1 = Roboclaw("/dev/ttyACM0", 115200)
roboclaw2 = Roboclaw("/dev/ttyACM1", 115200)
roboclaw1.Open()
roboclaw2.Open()
address = 0x80  # RoboClawのアドレス

def send_speed_to_motors(speed_x1, speed_x2, speed_y1, speed_y2):
    # 指定された速度をモータに送信
    roboclaw1.ForwardM1(address, speed_x1)
    roboclaw1.ForwardM2(address, speed_x2)
    roboclaw2.ForwardM1(address, speed_y1)
    roboclaw2.ForwardM2(address, speed_y2)

def stop_motors():
    # モータを停止
    send_speed_to_motors(0, 0, 0, 0)

# CSVファイルを読み込み、データに基づいてモータを制御
try:
    with open("random_data.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # ヘッダー行をスキップ
        
        for row in reader:
            # CSVから速度データを取得し、30倍
            speed_x1, speed_x2, speed_y1, speed_y2 = [int(value) * 30 for value in row]

            # モータを駆動
            send_speed_to_motors(speed_x1, speed_x2, speed_y1, speed_y2)

            # 30Hz（約0.033秒）間隔で制御
            time.sleep(0.033)

            # 'q'キーが押されたらループを終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_motors()
                print("Motors stopped.")
                break

finally:
    stop_motors()
    cv2.destroyAllWindows()
