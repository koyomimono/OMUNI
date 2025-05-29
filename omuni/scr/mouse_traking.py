import threading
from evdev import InputDevice, ecodes

class MouseTracker:
    def __init__(self, device_path='/dev/input/event5', scaling=0.01):
        self.device_path = device_path
        self.scaling = scaling
        self.device = InputDevice(device_path)
        self.x = 0.0
        self.y = 0.0
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()  # 停止フラグ用

    def start(self, callback=None):
        """ スレッドを開始し、位置更新時にコールバック関数を呼び出す """
        self._running = True
        self._stop_event.clear()  # 停止フラグをリセット
        self._thread = threading.Thread(target=self._track_loop, args=(callback,))
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """ トラッキングを停止 """
        self._running = False
        self._stop_event.set()  # 停止信号を設定
        if self._thread:
            self._thread.join()  # スレッドが終了するのを待つ

    def get_position(self):
        """ 現在の座標を返す """
        return self.x, self.y

    def _track_loop(self, callback):
        """ マウス移動を監視し、位置を更新する """
        while not self._stop_event.is_set():  # 停止フラグが立つまでループ
            event = self.device.read_one()  # 非ブロッキングでイベントを1つ読む
            if self._stop_event.is_set():
                return  # 停止フラグが立ったらスレッドを終了
            if event is None:
                continue  # イベントがない場合は次のループへ
            if event.type == ecodes.EV_REL:
                if event.code == ecodes.REL_X:
                    self.y -= event.value * self.scaling
                elif event.code == ecodes.REL_Y:
                    self.x += event.value * self.scaling
                if callback:
                    callback(self.x, self.y)
