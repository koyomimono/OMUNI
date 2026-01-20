import cv2
import time
import numpy as np

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

def open_camera_with_warmup(index=0, width=640, height=480, warmup_sec=2.0):
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print("❌ 카메라 오픈 실패")
        return None, None

    # 해상도 설정 (선택)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("🎥 카메라 워밍업 중...")

    start = time.time()
    good_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ 프레임 읽기 실패, 재시도...")
            time.sleep(0.05)
            continue

        # 완전 새까만 프레임은 버림
        mean_val = frame.mean()
        # print("mean:", mean_val)  # 디버깅용

        if mean_val > 5:  # 밝기가 0 근처면 아직 진짜 영상이 아님
            good_frame = frame
            break

        if time.time() - start > warmup_sec:
            # 일정 시간 지나도 어두우면 그냥 마지막 프레임이라도 사용
            good_frame = frame
            print("⚠ 워밍업 타임아웃, 현재 프레임으로 진행")
            break

    print("✅ 카메라 워밍업 완료")
    return cap, good_frame


def main():
    cap, first_frame = open_camera_with_warmup(CAM_INDEX, FRAME_W, FRAME_H)
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패")
            break

        cv2.imshow("Camera (warmup handled)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
