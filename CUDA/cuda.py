import cv2
import numpy as np

# CUDAが利用可能か確認
if not cv2.cuda.getCudaEnabledDeviceCount():
    print('CUDAサポートのないデバイスです。')
    exit()

# カメラキャプチャの初期化
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 二値化処理
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 輪郭を取得
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 最大の輪郭を取得
        c = max(contours, key=cv2.contourArea)

        # モーメントを計算
        M = cv2.moments(c)

        # 重心を計算
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # 結果を表示
    cv2.imshow('Frame', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()