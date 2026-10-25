// Wired transport: MJPEG-over-UART. Only compiled for the *_serial environments.
#ifdef TRANSPORT_SERIAL
#include <Arduino.h>
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_camera.h"
#include "serial_link.h"
#include "stats.h"
#include "network_config.h"
#include "telemetry_protocol.h"

// Single writer: only frame_task touches the UART. IMU/status packets are queued (never blocking
// the 100 Hz IMU task while a JPEG is on the wire) and written between frames. Their timestamps
// are taken at sampling time, so queuing delay does not hurt alignment; it only delays arrival.
struct QItem { uint8_t type; uint8_t len; uint8_t data[56]; };
static QueueHandle_t s_q = nullptr;

static uint16_t crc16(uint16_t crc, const uint8_t* d, size_t n) {
  while (n--) {
    crc ^= (uint16_t)(*d++) << 8;
    for (int i = 0; i < 8; i++) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

static void send_frame(uint8_t type, const uint8_t* p1, size_t n1, const uint8_t* p2, size_t n2) {
  uint32_t len = (uint32_t)(n1 + n2);
  uint8_t hdr[7] = {0xA5, 0x5A, type, (uint8_t)len, (uint8_t)(len >> 8), (uint8_t)(len >> 16), (uint8_t)(len >> 24)};
  uint16_t crc = crc16(0xFFFF, hdr + 2, 5);
  if (n1) crc = crc16(crc, p1, n1);
  if (n2) crc = crc16(crc, p2, n2);
  uint8_t tail[2] = {(uint8_t)crc, (uint8_t)(crc >> 8)};
  Serial.write(hdr, sizeof(hdr));
  if (n1) Serial.write(p1, n1);
  if (n2) Serial.write(p2, n2);
  Serial.write(tail, sizeof(tail));
}

void serial_link_init() { if (!s_q) s_q = xQueueCreate(128, sizeof(QItem)); }

void serial_send(uint8_t type, const uint8_t* data, size_t len) {
  if (!s_q || len > sizeof(((QItem*)0)->data)) return;
  QItem it; it.type = type; it.len = (uint8_t)len; memcpy(it.data, data, len);
  xQueueSend(s_q, &it, 0);                      // non-blocking; drops if the wire is badly backed up
}

static void drain_queue() {
  QItem it;
  while (s_q && xQueueReceive(s_q, &it, 0) == pdTRUE) send_frame(it.type, it.data, it.len, nullptr, 0);
}

static void frame_task(void*) {
  uint32_t frame_id = 0, win_n = 0, fail = 0;
  int64_t win_t0 = esp_timer_get_time(), next_due = 0;
  for (;;) {
    drain_queue();
    camera_fb_t* fb = esp_camera_fb_get();        // newest frame (grab mode LATEST)
    if (!fb) {
      g_stats.capture_fail++;
      if (++fail > 300) ESP.restart();
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    fail = 0;
    int64_t now = esp_timer_get_time();
    int64_t cap = (int64_t)fb->timestamp.tv_sec * 1000000LL + fb->timestamp.tv_usec;
    if (cap <= 0 || cap > now) cap = now;
    uint8_t pre[12];
    uint32_t id = frame_id;
    memcpy(pre, &cap, 8);
    memcpy(pre + 8, &id, 4);
    send_frame(SL_JPEG, pre, sizeof(pre), fb->buf, fb->len);   // blocks at wire speed
    esp_camera_fb_return(fb);
    drain_queue();
    frame_id++; g_stats.frames_sent++; win_n++;

    int64_t t = esp_timer_get_time();
    if (t - win_t0 >= 1000000) { g_stats.fps = win_n * 1e6f / (t - win_t0); win_n = 0; win_t0 = t; }
    if (g_cfg.fps_limit > 0) {
      int64_t interval = 1000000LL / g_cfg.fps_limit;
      if (next_due == 0) next_due = t;
      next_due += interval;
      if (next_due < t - interval) next_due = t;
      int64_t wait = next_due - esp_timer_get_time();
      if (wait > 1000) vTaskDelay(pdMS_TO_TICKS(wait / 1000));
    }
  }
}

static void status_task(void*) {
  uint32_t seq = 0;
  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(STATUS_PERIOD_MS));
    OgStatusPacket p = {};
    p.h.magic = OG_MAGIC; p.h.version = OG_VERSION; p.h.type = OG_TYPE_STATUS;
    p.h.device_id = DEVICE_ID; p.h.flags = OG_FLAG_VALID; p.h.seq = seq++;
    p.h.timestamp_us = (uint64_t)esp_timer_get_time();
    p.p.uptime_ms = millis(); p.p.free_heap = ESP.getFreeHeap(); p.p.free_psram = ESP.getFreePsram();
    p.p.frames_sent = g_stats.frames_sent; p.p.capture_fail = g_stats.capture_fail;
    p.p.fps_x10 = (uint16_t)(g_stats.fps * 10); p.p.imu_hz_x10 = (uint16_t)(g_stats.imu_hz * 10);
    p.p.imu_errors = (uint16_t)g_stats.imu_errors;
    p.p.rssi_dbm = 0;                              // no Wi-Fi in wired mode
    p.p.reset_reason = (uint8_t)esp_reset_reason(); p.p.brownout_seen = g_stats.brownout_seen;
    p.p.stream_clients = 1;
    serial_send(SL_STATUS, (const uint8_t*)&p, sizeof(p));
  }
}

void serial_link_start() {
  xTaskCreatePinnedToCore(frame_task, "sframe", 8192, nullptr, 2, nullptr, 1);
  xTaskCreatePinnedToCore(status_task, "sstatus", 4096, nullptr, 1, nullptr, 0);
}
#endif
