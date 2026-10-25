// Camera bring-up and live parameter control.
#ifndef TOOL_I2C_SCAN
#include "camera_config.h"
#include "stats.h"

static const struct { const char* n; framesize_t f; } kSizes[] = {
  {"qqvga", FRAMESIZE_QQVGA}, {"qvga", FRAMESIZE_QVGA}, {"cif", FRAMESIZE_CIF},
  {"hvga", FRAMESIZE_HVGA},   {"vga", FRAMESIZE_VGA},   {"svga", FRAMESIZE_SVGA},
};

const char* framesize_name(framesize_t fs) {
  for (auto& e : kSizes) if (e.f == fs) return e.n;
  return "other";
}
bool framesize_from_name(const char* s, framesize_t* out) {
  for (auto& e : kSizes) if (!strcasecmp(e.n, s)) { *out = e.f; return true; }
  return false;
}

bool camera_start(const RuntimeConfig& cfg) {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    // Allocate buffers for the LARGEST size once so frame size can be changed live
    // (the driver sizes buffers at init). Actual size is applied right after init.
    c.frame_size   = FRAMESIZE_UXGA;
    c.jpeg_quality = 12;
    c.fb_count     = 2;                       // double buffer
    c.fb_location  = CAMERA_FB_IN_PSRAM;
    c.grab_mode    = CAMERA_GRAB_LATEST;      // always hand out the freshest frame (low latency)
  } else {
    Serial.println("[cam] WARNING: PSRAM not found -> DRAM single buffer, QVGA only");
    c.frame_size   = FRAMESIZE_QVGA;
    c.jpeg_quality = 12;
    c.fb_count     = 1;
    c.fb_location  = CAMERA_FB_IN_DRAM;
    c.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("[cam] esp_camera_init failed: 0x%x\n", err);
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return false;
  g_stats.sensor_pid = s->id.PID;
  Serial.printf("[cam] sensor PID=0x%02x (OV2640 = 0x26) MIDH/L=0x%02x%02x\n", s->id.PID, s->id.MIDH, s->id.MIDL);
  if (s->id.PID != OV2640_PID) Serial.println("[cam] NOTE: sensor is not an OV2640; settings may differ");
  camera_apply(cfg);
  return true;
}

void camera_apply(const RuntimeConfig& cfg) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return;
  if (!psramFound() && cfg.frame_size > FRAMESIZE_QVGA) s->set_framesize(s, FRAMESIZE_QVGA);
  else s->set_framesize(s, cfg.frame_size);
  s->set_quality(s, cfg.jpeg_quality);
  s->set_brightness(s, cfg.brightness);
  s->set_contrast(s, cfg.contrast);
  s->set_saturation(s, cfg.saturation);
  s->set_hmirror(s, cfg.hmirror ? 1 : 0);
  s->set_vflip(s, cfg.vflip ? 1 : 0);
}
#endif
