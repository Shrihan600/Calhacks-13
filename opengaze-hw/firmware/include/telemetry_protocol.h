#pragma once
// OpenGaze UDP telemetry protocol v1. Little-endian, packed, no padding.
// Mirrors receiver/imu_client.py (Python struct formats noted below).
//
// Header (18 bytes)  struct "<HBBBBIQ"
//   magic        u16  bytes 'O','G' on the wire => value 0x474F
//   version      u8   1
//   type         u8   1 = IMU sample, 2 = STATUS
//   device_id    u8   1 = left_cam, 2 = right_cam
//   flags        u8   bit0 = payload valid (IMU read OK)
//   seq          u32  per-device, per-type counter (receiver handles wrap)
//   timestamp_us u64  esp_timer_get_time(): monotonic since boot, 64-bit (no practical rollover).
//                     SAME clock as X-Timestamp-Us in video frames from the same device.
// IMU payload "<7f" (28 bytes) => 46-byte packet
//   ax ay az  float  acceleration in g (1 g = 9.80665 m/s^2), sensor frame
//   gx gy gz  float  angular velocity in degrees/second, sensor frame
//   temp_c    float  die temperature in deg C (MPU-6050: raw/340 + 36.53)
// STATUS payload "<IIIIIHHHHbBBB" (32 bytes) => 50-byte packet
//   uptime_ms, free_heap, free_psram, frames_sent, capture_fail : u32
//   fps_x10, imu_hz_x10, wifi_reconnects, imu_errors            : u16
//   rssi_dbm i8, reset_reason u8 (esp_reset_reason_t), brownout_seen u8, stream_clients u8
#include <stdint.h>

#define OG_MAGIC       0x474F
#define OG_VERSION     1
#define OG_TYPE_IMU    1
#define OG_TYPE_STATUS 2
#define OG_FLAG_VALID  0x01

struct __attribute__((packed)) OgHeader {
  uint16_t magic;
  uint8_t  version;
  uint8_t  type;
  uint8_t  device_id;
  uint8_t  flags;
  uint32_t seq;
  uint64_t timestamp_us;
};

struct __attribute__((packed)) OgImuPayload {
  float ax, ay, az;   // g
  float gx, gy, gz;   // deg/s
  float temp_c;       // deg C
};

struct __attribute__((packed)) OgStatusPayload {
  uint32_t uptime_ms, free_heap, free_psram, frames_sent, capture_fail;
  uint16_t fps_x10, imu_hz_x10, wifi_reconnects, imu_errors;
  int8_t   rssi_dbm;
  uint8_t  reset_reason, brownout_seen, stream_clients;
};

struct __attribute__((packed)) OgImuPacket    { OgHeader h; OgImuPayload p; };
struct __attribute__((packed)) OgStatusPacket { OgHeader h; OgStatusPayload p; };

static_assert(sizeof(OgHeader) == 18, "header size");
static_assert(sizeof(OgImuPacket) == 46, "imu packet size");
static_assert(sizeof(OgStatusPacket) == 50, "status packet size");
