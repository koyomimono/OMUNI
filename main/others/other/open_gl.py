# jetson_camera_send.py
import cv2
import socket

# カメラの設定
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ソケット設定
HOST = '192.168.50.9'  # Raspberry PiのIPアドレスに置き換え
PORT = 50001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケール変換と黒い物体の検出
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 最大輪郭（黒い物体）の重心を計算
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # 画像中心との距離
                x = cx - frame.shape[1] // 2
                y = cy - frame.shape[0] // 2
                # データを送信
                sock.sendall(f"{x},{y}".encode())

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    sock.close()
    cv2.destroyAllWindows()
