from evdev import InputDevice, ecodes
import os

SCALING =0.01
# デバイスのパス
device_path = '/dev/input/event9'

# デバイスを開く
dev = InputDevice(device_path)

print(f"デバイス: {dev.name}")
print("マウスの移動を監視中... Ctrl+C で終了")

x, y = 0, 0  # 仮想座標

try:
    for event in dev.read_loop():
        if event.type == ecodes.EV_REL:
            if event.code == ecodes.REL_X:
                y -= event.value * SCALING
            elif event.code == ecodes.REL_Y:
                x += event.value * SCALING
            print(f"現在の座標: ({x:.2f}, {y:.2f})")
except KeyboardInterrupt:
    print("終了しました。")