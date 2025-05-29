import threading
import datetime
from evdev import InputDevice, ecodes
import os

class MouseTracker:
    def __init__(self, device_path):
        self.device_path = device_path
        self.track_movement = threading.Event()
        self.keep_running = True
        self.data_file = None
        self.file_lock = threading.Lock()

    def toggle_tracking(self):
        if self.track_movement.is_set():
            self.track_movement.clear()
            print("Tracking stopped.")
            self.close_data_file()
        else:
            self.track_movement.set()
            print("Tracking started.")
            self.setup_data_file()

    def setup_data_file(self):
        directory = "/home/swarm/Documents/python_ANTAM/test"
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(directory, f"BT_{timestamp}.csv")
        self.data_file = open(filename, 'w')
        self.data_file.write("Timestamp,X,Y\n")

    def close_data_file(self):
        if self.data_file:
            self.data_file.close()
            self.data_file = None

    def read_mouse(self):
        dev = InputDevice(self.device_path)
        x, y = 0, 0

        for event in dev.read_loop():
            if not self.keep_running:
                break
            if not self.track_movement.is_set():
                continue

            if event.type == ecodes.EV_REL:
                if event.code == ecodes.REL_X:
                    x += event.value
                elif event.code == ecodes.REL_Y:
                    y += event.value

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                # データをファイルに記録
                with self.file_lock:
                    if self.data_file:
                        self.data_file.write(f"{timestamp},{x},{y}\n")
                    print(f"X: {x}, Y: {y}")


    def start(self):
        thread = threading.Thread(target=self.read_mouse)
        thread.start()
        return thread

    def stop(self):
        self.keep_running = False
        self.track_movement.clear()
        self.close_data_file()

if __name__ == "__main__":
    tracker = MouseTracker("/dev/input/event12")

    thread = tracker.start()
    try:
        while True:
            command = input("Enter 's' to start/stop tracking, 'q' to quit: ")
            if command.lower() == 's':
                tracker.toggle_tracking()
            elif command.lower() == 'q':
                tracker.stop()
                break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
