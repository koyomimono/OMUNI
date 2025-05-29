import queue

from roboclaw_3 import Roboclaw

# Queue to store frames to be displayed
frame_queue = queue.Queue(maxsize=1)

# RoboClawの初期化
roboclaw1 = Roboclaw("/dev/ttyACM0", 115200)
roboclaw2 = Roboclaw("/dev/ttyACM1", 115200)
roboclaw1.Open()
roboclaw2.Open()
address = 0x80  # RoboClawのアドレス
x1 = 30
x2 = -20
y1 = 30
y2 = -20

try:
    while True:
        # `q`キーが押されたかをinput()で確認
        user_input = input("終了するには'q'を押してください: ")
        if user_input == "q":
            print("プログラムを終了します...")
            break

        # モーター制御
        if x1 >= 0:
            roboclaw1.ForwardM1(address, abs(int(x1)))
        else:
            roboclaw1.BackwardM1(address, abs(int(x1)))

        if x2 >= 0:
            roboclaw1.ForwardM2(address, abs(int(x2)))
        else:
            roboclaw1.BackwardM2(address, abs(int(x2)))

        if y1 >= 0:
            roboclaw2.ForwardM1(address, abs(int(y1)))
        else:
            roboclaw2.BackwardM1(address, abs(int(y1)))

        if y2 >= 0:
            roboclaw2.ForwardM2(address, abs(int(y2)))
        else:
            roboclaw2.BackwardM2(address, abs(int(y2)))

except KeyboardInterrupt:
    # Ctrl+Cで安全に終了
    print("プログラムを中断しました。")

finally:
    # 終了時の処理（Closeの呼び出しなし）
    print("終了処理完了。")
