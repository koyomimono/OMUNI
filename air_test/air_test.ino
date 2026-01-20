#include <Wire.h>

// ===== 설정 =====
const uint8_t PIN_TRIG = 22;
const float   VDD      = 5.0;
const uint8_t DAC_ADDR = 0x60;

// 반복 총 시간 (ms)
const unsigned long TOTAL_DURATION_MS = 1000;

// '1'~'5' 입력 전압 매핑
const float V_MAP[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

// ===== 유틸 =====
uint16_t voltsToCode(float v) {
  if (v < 0) v = 0;
  if (v > VDD) v = VDD;
  return (uint16_t)((4095.0f * (v / VDD)) + 0.5f);
}

void setDAC(float v) {
  uint16_t code = voltsToCode(v);
  Wire.beginTransmission(DAC_ADDR);
  Wire.write(0x40);
  Wire.write((code >> 4) & 0xFF);
  Wire.write((code & 0x0F) << 4);
  Wire.endTransmission();
}

void setup() {
  Wire.begin();
  Serial.begin(9600);

  pinMode(PIN_TRIG, OUTPUT);
  digitalWrite(PIN_TRIG, LOW);
  setDAC(0.0);

  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) return;

  char c = Serial.read();
  if (c < '1' || c > '5') return;

  float v = V_MAP[c - '1'];

  Serial.print("START LOOP 10s, V=");
  Serial.println(v, 3);

  unsigned long t_start = millis();

  while (millis() - t_start < TOTAL_DURATION_MS) {
    // 1) 전압 설정
    setDAC(v);
    delay(800);          // 1초 유지

    // 2) 트리거 펄스
    digitalWrite(PIN_TRIG, HIGH);
    delay(100);           // 0.1초
    digitalWrite(PIN_TRIG, LOW);

    // 3) 전압 리셋
    setDAC(0.0);
    delay(30);           // 다음 루프까지 짧은 휴지
  }

  Serial.println("DONE 10s LOOP");
}
