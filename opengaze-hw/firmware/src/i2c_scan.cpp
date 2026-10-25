// Stand-alone bring-up tool: pio run -e i2c_scan -t upload && pio device monitor
// Scans I2C on GPIO13 (SDA) / GPIO14 (SCL), reads MPU-6050 WHO_AM_I, then prints live data.
#ifdef TOOL_I2C_SCAN
#include <Arduino.h>
#include <Wire.h>
#define SDA_PIN 13
#define SCL_PIN 14

static uint8_t addr = 0;

static void scan() {
  Serial.println("\n--- I2C scan (SDA=GPIO13, SCL=GPIO14) ---");
  int found = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    uint8_t e = Wire.endTransmission();
    if (e == 0) { Serial.printf("  device at 0x%02X%s\n", a, (a == 0x68 || a == 0x69) ? "  <- MPU-6050 candidate" : ""); found++; if (a == 0x68 || a == 0x69) addr = a; }
    else if (e == 4) Serial.printf("  unknown error at 0x%02X\n", a);
  }
  if (!found) Serial.println("  NOTHING FOUND. Check: 3V3 and GND to the MPU, SDA/SCL not swapped, wires <15 cm, microSD slot EMPTY.");
}

static bool rd(uint8_t reg, uint8_t* b, size_t n) {
  Wire.beginTransmission(addr); Wire.write(reg);
  if (Wire.endTransmission(false)) return false;
  if (Wire.requestFrom((int)addr, (int)n) != (int)n) return false;
  for (size_t i = 0; i < n; i++) b[i] = Wire.read();
  return true;
}

void setup() {
  Serial.begin(115200); delay(500);
  Serial.println("\nOpenGaze I2C diagnostic");
  Wire.begin(SDA_PIN, SCL_PIN, 100000);   // slow & safe for first bring-up
  Wire.setTimeOut(50);
  scan();
  if (!addr) return;
  uint8_t who = 0; rd(0x75, &who, 1);
  Serial.printf("WHO_AM_I = 0x%02X (genuine MPU-6050 = 0x68; 0x70/0x71/0x73 = MPU-6500/9250-class, 0x98 = some clones)\n", who);
  Wire.beginTransmission(addr); Wire.write(0x6B); Wire.write(0x01); Wire.endTransmission();
  delay(100);
  Serial.println("Streaming raw reads. Expect |a| ~ 1 g at rest (az ~ +1 if flat), gyro ~ 0 +/- 2 deg/s.");
}

void loop() {
  if (!addr) { delay(2000); scan(); return; }
  uint8_t b[14];
  if (!rd(0x3B, b, 14)) { Serial.println("read FAILED"); delay(500); return; }
  auto s = [&](int i) { return (int16_t)((b[i] << 8) | b[i + 1]); };
  // Default full-scale after reset: +/-2 g (16384 LSB/g), +/-250 deg/s (131 LSB/dps)
  float ax = s(0) / 16384.0f, ay = s(2) / 16384.0f, az = s(4) / 16384.0f;
  float t = s(6) / 340.0f + 36.53f;
  float gx = s(8) / 131.0f, gy = s(10) / 131.0f, gz = s(12) / 131.0f;
  Serial.printf("a[g] %+.3f %+.3f %+.3f |a|=%.3f  g[dps] %+.2f %+.2f %+.2f  T=%.1fC\n",
                ax, ay, az, sqrtf(ax * ax + ay * ay + az * az), gx, gy, gz, t);
  delay(200);
}
#endif
