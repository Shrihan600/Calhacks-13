#pragma once
#include <Arduino.h>
#include "camera_config.h"

struct Stats {
  volatile uint32_t frames_sent = 0;
  volatile uint32_t capture_fail = 0;
  volatile uint32_t wifi_reconnects = 0;
  volatile uint32_t imu_errors = 0;
  volatile uint32_t imu_samples = 0;
  volatile uint32_t udp_fail = 0;
  volatile uint32_t stream_clients = 0;
  volatile float    fps = 0;
  volatile float    imu_hz = 0;
  volatile float    cap_wait_ms = 0;   // time blocked waiting for the next camera frame (EMA)
  volatile float    send_ms = 0;       // time spent pushing a frame into the Wi-Fi socket (EMA)
  volatile float    frame_kb = 0;      // average JPEG size (EMA)
  volatile bool     imu_present = false;
  volatile uint8_t  imu_addr = 0;
  volatile bool     brownout_seen = false;
  volatile uint8_t  sensor_pid = 0;
};

extern Stats g_stats;
extern RuntimeConfig g_cfg;
extern volatile uint32_t g_telem_ip;   // IPv4 (lwip byte order); 0 = unset -> no UDP sent
void config_save();
void config_defaults();
bool telem_set_host(const char* s);    // "auto" or dotted IPv4
