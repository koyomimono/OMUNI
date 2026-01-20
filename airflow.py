# optical_flow_highlight_robust_bitmap.py
# - 1초 동안 측정된 움직임을 누적해서 비트맵(heatmap)으로 시각화
# - 's' 키: 1초 측정 시작 + /dev/ttyACM2(9600bps)로 '1' 전송
# - 측정 종료 후 누적 움직임 비트맵을 별도 창에 표시 및 PNG 저장

import cv2
import numpy as np
import time
import serial  # 시리얼 통신용

# ===== 카메라 & 해상도 설정 =====
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
TARGET_FPS = 60

# 광류 계산용 다운스케일 (작을수록 빠름, 너무 작으면 세부 손실)
SMALL_W, SMALL_H = 160, 120

# ===== 시리얼 설정 =====
SERIAL_PORT = "/dev/ttyACM2"
BAUDRATE = 9600

# ===== 노이즈/잔상 억제 파라미터 =====
BASELINE_EMA_ALPHA = 0.2  # 프레임 전체 플리커 보정용 EMA
FLOW_GAUSS_SIGMA = 1.0    # 공간 노이즈 축소용 가우시안 블러

# 시간적 평활화(EMA)
MOTION_EMA_ALPHA = 0.3

# 히스테리시스 임계값(퍼센타일 기준)
PCT_HIGH = 90
PCT_LOW = 75

# 형태학적 처리 & 최소 면적
MORPH_KERNEL_SIZE = 3
MIN_BLOB_AREA = 120  # 너무 작은 점 잡음 제거

# 오버레이
ALPHA = 0.6
COLORMAP = cv2.COLORMAP_JET


def _hysteresis_mask(mag_ema, prev_mask):
    """히스테리시스(High/Low) 임계값으로 마스크 생성"""
    valid = mag_ema > 0
    if np.count_nonzero(valid) == 0:
        high = low = 0.0
    else:
        vals = mag_ema[valid]
        high = np.percentile(vals, PCT_HIGH)
        low = np.percentile(vals, PCT_LOW)

    new_on = mag_ema >= high
    keep_on = (prev_mask > 0) & (mag_ema >= low)
    mask = (new_on | keep_on).astype(np.uint8)

    return mask


def _postprocess_mask(mask):
    """모폴로지 오프닝 + 작은 블롭 제거"""
    if MORPH_KERNEL_SIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    if MIN_BLOB_AREA > 0:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(mask)
        for c in cnts:
            area = cv2.contourArea(c)
            if area >= MIN_BLOB_AREA:
                cv2.drawContours(cleaned, [c], -1, color=1, thickness=-1)
        mask = cleaned

    return mask


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print("카메라 오픈 실패")
        return

    ok, prev_bgr = cap.read()
    if not ok:
        print("첫 프레임 읽기 실패")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    prev_small = cv2.resize(prev_gray, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)

    # 잔상 억제용 상태 변수
    mag_ema_full = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    baseline_mag = 0.0
    prev_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

    # 측정 시작 관련 상태
    started = False
    start_time = 0.0

    # 1초 동안의 움직임 누적 비트맵 (float32로 누적)
    motion_accum = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)

    # 시리얼 포트 핸들
    ser = None

    print("대기 상태입니다. 윈도우에 포커스를 두고 's' 키를 누르면 1초간 측정을 시작합니다. (q: 종료)")

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)

        # Farnebäck 광학 흐름
        flow = cv2.calcOpticalFlowFarneback(
            prev_small,
            small_gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=11,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )

        # 크기만 사용 → 원본 해상도로 보간
        mag_small, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_full = cv2.resize(
            mag_small, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR
        )

        # 공간 노이즈 축소
        if FLOW_GAUSS_SIGMA > 0:
            mag_full = cv2.GaussianBlur(mag_full, ksize=(0, 0), sigmaX=FLOW_GAUSS_SIGMA)

        # 프레임 전체 플리커(노이즈) 보정: 중앙값 베이스라인 추적 후 제거
        frame_med = float(np.median(mag_full))
        baseline_mag = (
            1 - BASELINE_EMA_ALPHA
        ) * baseline_mag + BASELINE_EMA_ALPHA * frame_med
        mag_detrend = mag_full - baseline_mag
        mag_detrend[mag_detrend < 0] = 0.0

        # 시간적 EMA (지속된 변화만 통과)
        mag_ema_full = (
            1 - MOTION_EMA_ALPHA
        ) * mag_ema_full + MOTION_EMA_ALPHA * mag_detrend

        # ======= 여기서부터: 1초 동안 움직임 누적 (비트맵 생성용) =======
        if started:
            elapsed = time.time() - start_time
            if elapsed <= 1.0:
                # 노이즈 제거된 움직임 크기를 계속 더함
                motion_accum += mag_detrend
            # 1초가 넘으면 루프 종료
            if elapsed >= 1.0:
                print("1초 측정 완료, 비트맵 생성 단계로 이동")
                break
        # ============================================================

        # 히스테리시스 임계값으로 마스크 생성
        mask = _hysteresis_mask(mag_ema_full, prev_mask)

        # 형태학적 오프닝 + 최소 면적 필터
        mask = _postprocess_mask(mask)

        # 컬러맵용 정규화
        mag_norm = cv2.normalize(mag_ema_full, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        heat = cv2.applyColorMap(mag_norm, COLORMAP)

        # 전체 프레임을 한 번 블렌딩
        overlay_all = cv2.addWeighted(bgr, 1 - ALPHA, heat, ALPHA, 0)

        # 움직임 영역만 오버레이
        mask_3c = cv2.merge([mask, mask, mask])
        result = np.where(mask_3c == 1, overlay_all, bgr)

        # 화면에 상태 텍스트 표시
        if not started:
            cv2.putText(
                result,
                "Press 's' to START (1 sec measurement)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                result,
                "MEASURING... (1 sec window)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Robust Optical Flow Highlight", result)

        # 다음 루프 준비
        prev_small = small_gray
        prev_mask = mask

        key = cv2.waitKey(1) & 0xFF

        # 'q' 종료
        if key == ord("q"):
            print("사용자에 의해 종료(q)")
            started = False  # 측정 시작 전에 끊을 수도 있으니
            break

        # 's' 눌렸고 아직 시작 안 했을 때 → 측정 시작 + 시리얼로 '1' 전송
        if key == ord("s") and not started:
            started = True
            start_time = time.time()
            print("측정 시작: 1초 동안 동작, /dev/ttyACM2 로 '1' 전송")

            # 시리얼 오픈 후 '1' 전송
            ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
            ser.write(b"1")

    # 루프 종료 후 정리
    cap.release()
    if ser is not None and ser.is_open:
        ser.close()

    # ===== 1초 동안 누적된 움직임 비트맵 시각화 =====
    if started and np.any(motion_accum > 0):
        # 0~255로 정규화해서 uint8로 변환
        accum_norm = cv2.normalize(motion_accum, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        # 컬러맵 적용 (heatmap)
        accum_color = cv2.applyColorMap(accum_norm, COLORMAP)

        cv2.imshow("Motion Bitmap (1s)", accum_color)
        # 파일로도 저장 (원하면 사용)
        cv2.imwrite("motion_bitmap_1s.png", accum_color)
        print("motion_bitmap_1s.png 로 저장 완료")

        print("아무 키나 누르면 창이 닫힙니다.")
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
