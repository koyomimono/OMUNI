#-- mouseraking.py --#
from evdev import InputDevice, ecodes
import threading

class MouseTracker:
    def __init__(self, device_path='/dev/input/event5', scaling=0.01):
        self.device_path = device_path
        self.scaling = scaling
        self.device = InputDevice(device_path)
        self.x = 0.0
        self.y = 0.0
        self._running = False
        self._thread = None

    def start(self, callback=None):
        """
        開始し、オプションで位置更新時にコールバック関数を呼び出す。
        """
        self._running = True
        self._thread = threading.Thread(target=self._track_loop, args=(callback,))
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """ トラッキングを停止 """
        self._running = False
        if self._thread:
            self._thread.join()

    def get_position(self):
        """ 現在の座標を返す """
        return self.x, self.y

    def _track_loop(self, callback):
        for event in self.device.read_loop():
            if not self._running:
                break
            if event.type == ecodes.EV_REL:
                if event.code == ecodes.REL_X:
                    self.y -= event.value * self.scaling
                elif event.code == ecodes.REL_Y:
                    self.x += event.value * self.scaling
                if callback:
                    callback(self.x, self.y)
