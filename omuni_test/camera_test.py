# camera_test.py
import cv2
import time

CAMERA_INDEX  =1
# カメラの初期化（0は通常ノートPC内蔵カメラ）
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)#

if not cap.isOpened():
    print("カメラが見つかりません")
    exit()

# FPS計算用タイマー
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できません")
        break

    # 現在時刻
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cannyエッジ検出
    edges = cv2.Canny(gray, 100, 200)

    # FPSをフレームに描画（位置、フォント、色など調整可）
    cv2.putText(gray, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)

    # 表示
    cv2.imshow('Camera Feed (with FPS)', gray)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
