# scr/mouse_traking.py

import threading
from evdev import InputDevice, ecodes

class MouseTracker(threading.Thread):
    def __init__(self, device_path='/dev/input/event9', scaling=0.01):
        super().__init__(daemon=True)
        self.device_path = device_path
        self.scaling = scaling
        self.running = False
        self.callback = None

    def run(self):
        try:
            dev = InputDevice(self.device_path)
            print(f"[MouseTracker] デバイス: {dev.name}")
        except Exception as e:
            print(f"[MouseTracker] デバイスオープンエラー: {e}")
            return

        x, y = 0.0, 0.0
        self.running = True

        for event in dev.read_loop():
            if not self.running:
                break
            if event.type == ecodes.EV_REL:
                if event.code == ecodes.REL_X:
                    y -= event.value * self.scaling
                elif event.code == ecodes.REL_Y:
                    x += event.value * self.scaling
                if self.callback:
                    self.callback(x, y)

    def start(self, callback):
        self.callback = callback
        super().start()

    def stop(self):
        self.running = False
