#pragma once
#include <Arduino.h>
#include "esp_camera.h"

// ---- AI-Thinker ESP32-CAM camera pinout (the Inland MD0456 is made by Shenzhen KEYES and is
// ---- expected to match; the firmware prints the sensor PID at boot as a bring-up check) ----
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

// ---- MPU-6050 on the (unused) microSD pins. A microSD card MUST NOT be inserted. ----
#define IMU_SDA_PIN     13   // SD DATA3
#define IMU_SCL_PIN     14   // SD CLK
#define IMU_I2C_HZ      400000

// Runtime-adjustable parameters: http://<ip>/control?var=<name>&val=<value>  (add &save=1 to persist)
struct RuntimeConfig {
  framesize_t frame_size   = FRAMESIZE_QVGA;  // 320x240
  int  jpeg_quality        = 12;              // 4..63, LOWER number = better quality / bigger frames
  int  fps_limit           = 25;              // 0 = unlimited (whatever sensor + Wi-Fi allow)
  int  imu_hz              = 100;             // 10..500; FreeRTOS tick is 1 ms, so divisors of 1000 are exact
  int  brightness          = 0;               // -2..2
  int  contrast            = 0;               // -2..2
  int  saturation          = 0;               // -2..2
  bool hmirror             = false;
  bool vflip               = false;
  bool telem_auto          = true;            // send UDP to the last host that opened /stream
  char telem_host[40]      = "";              // used when telem_auto == false
};

bool        camera_start(const RuntimeConfig& cfg);
void        camera_apply(const RuntimeConfig& cfg);
const char* framesize_name(framesize_t fs);
bool        framesize_from_name(const char* s, framesize_t* out);
