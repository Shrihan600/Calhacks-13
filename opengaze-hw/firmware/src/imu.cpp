// MPU-6050 driver (raw register access, no external library) + 100 Hz UDP sampler task.
#ifndef TOOL_I2C_SCAN
#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_timer.h"
#include "imu.h"
#include "stats.h"
#include "camera_config.h"
#include "network_config.h"
#include "telemetry_protocol.h"
#include "serial_link.h"

#if HAS_IMU
static uint8_t s_addr = 0;
static const float ACCEL_LSB_PER_G   = 8192.0f;  // +/-4 g
static const float GYRO_LSB_PER_DPS  = 65.5f;    // +/-500 deg/s

static bool wr(uint8_t reg, uint8_t v) {
  Wire.beginTransmission(s_addr); Wire.write(reg); Wire.write(v);
  return Wire.endTransmission() == 0;
}
static bool rd(uint8_t reg, uint8_t* buf, size_t n) {
  Wire.beginTransmission(s_addr); Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)s_addr, (int)n) != (int)n) return false;
  for (size_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

bool imu_begin() {
  Wire.end();
  if (!Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN, IMU_I2C_HZ)) { Serial.println("[imu] Wire.begin failed"); return false; }
  Wire.setTimeOut(50);
  for (uint8_t a : {0x68, 0x69}) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { s_addr = a; break; }
  }
  if (!s_addr) { Serial.println("[imu] no device at 0x68/0x69 (check wiring, 3V3, pull-ups)"); g_stats.imu_present = false; return false; }
  uint8_t who = 0;
  if (!rd(0x75, &who, 1)) return false;
  Serial.printf("[imu] device at 0x%02X WHO_AM_I=0x%02X (MPU-6050 = 0x68)\n", s_addr, who);
  if (who != 0x68) Serial.println("[imu] WARNING: unexpected WHO_AM_I (clone/MPU-6500/9250?). Continuing; verify readings.");
  wr(0x6B, 0x80); delay(100);          // device reset
  wr(0x6B, 0x01); delay(50);           // wake, PLL with X gyro clock
  wr(0x1A, 0x03);                      // DLPF ~44 Hz accel / 42 Hz gyro; gyro output rate 1 kHz
  wr(0x19, 0x00);                      // sample-rate divider 0 => 1 kHz internal sample rate
  wr(0x1B, 0x08);                      // gyro  +/-500 deg/s
  wr(0x1C, 0x08);                      // accel +/-4 g
  g_stats.imu_present = true; g_stats.imu_addr = s_addr;
  return true;
}

static void imu_task(void*) {
  WiFiUDP udp;
  uint32_t seq = 0, consecutive_err = 0, win_n = 0, win_t0 = millis(), invalid_tx = 0;
  TickType_t last = xTaskGetTickCount();
  for (;;) {
    int hz = g_cfg.imu_hz < 10 ? 10 : (g_cfg.imu_hz > 500 ? 500 : g_cfg.imu_hz);
    TickType_t period = pdMS_TO_TICKS(1000 / hz);
    if (period < 1) period = 1;
    vTaskDelayUntil(&last, period);

    OgImuPacket pkt = {};
    pkt.h.magic = OG_MAGIC; pkt.h.version = OG_VERSION; pkt.h.type = OG_TYPE_IMU;
    pkt.h.device_id = DEVICE_ID;
    pkt.h.timestamp_us = (uint64_t)esp_timer_get_time();   // taken just BEFORE the I2C read

    bool ok = false;
    if (g_stats.imu_present) {
      uint8_t b[14];
      if (rd(0x3B, b, 14)) {
        auto i16 = [&](int i) { return (int16_t)((b[i] << 8) | b[i + 1]); };
        pkt.p.ax = i16(0) / ACCEL_LSB_PER_G; pkt.p.ay = i16(2) / ACCEL_LSB_PER_G; pkt.p.az = i16(4) / ACCEL_LSB_PER_G;
        pkt.p.temp_c = i16(6) / 340.0f + 36.53f;
        pkt.p.gx = i16(8) / GYRO_LSB_PER_DPS; pkt.p.gy = i16(10) / GYRO_LSB_PER_DPS; pkt.p.gz = i16(12) / GYRO_LSB_PER_DPS;
        ok = true; consecutive_err = 0; g_stats.imu_samples++; win_n++;
      }
    }
    if (!ok) {
      g_stats.imu_errors++;
      if (++consecutive_err >= 25) {            // ~0.25 s of failures: try to re-init the sensor
        Serial.println("[imu] repeated read errors, re-initialising");
        g_stats.imu_present = false; s_addr = 0;
        imu_begin();
        consecutive_err = 0;
      }
    }
    pkt.h.flags = ok ? OG_FLAG_VALID : 0;
    pkt.h.seq = seq++;

#ifdef TRANSPORT_SERIAL
    if (ok || (++invalid_tx % 10) == 0) serial_send(SL_IMU, (const uint8_t*)&pkt, sizeof(pkt));
#else
    uint32_t ip = g_telem_ip;
    // Invalid samples are rate-limited to ~10 Hz so a dead sensor does not flood the link.
    if (ip && (ok || (++invalid_tx % 10) == 0)) {
      if (!udp.beginPacket(IPAddress(ip), TELEM_UDP_PORT) ||
          udp.write((const uint8_t*)&pkt, sizeof(pkt)) != sizeof(pkt) || !udp.endPacket())
        g_stats.udp_fail++;
    }

#endif

    uint32_t now = millis();
    if (now - win_t0 >= 1000) { g_stats.imu_hz = win_n * 1000.0f / (now - win_t0); win_n = 0; win_t0 = now; }
  }
}

void imu_task_start() {
  xTaskCreatePinnedToCore(imu_task, "imu", 4096, nullptr, 3, nullptr, 0);
}
#else
bool imu_begin() { return false; }
void imu_task_start() {}
#endif
#endif
