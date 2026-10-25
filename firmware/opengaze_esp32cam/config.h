// OpenGaze ESP32-CAM firmware configuration.
// One sketch for both boards: set OPENGAZE_ROLE_RIGHT to 1 for the RIGHT board (the one
// wired to the MPU-6050) and 0 for the LEFT board, then flash each board separately.
#pragma once

#ifndef OPENGAZE_ROLE_RIGHT
#define OPENGAZE_ROLE_RIGHT 0
#endif

#if OPENGAZE_ROLE_RIGHT
#define OPENGAZE_DEVICE_ID "right_esp32"
#define OPENGAZE_ROLE "right_eye"
#define OPENGAZE_HOSTNAME "opengaze-right"  // → opengaze-right.local
#define OPENGAZE_HAS_IMU 1
#else
#define OPENGAZE_DEVICE_ID "left_esp32"
#define OPENGAZE_ROLE "left_eye"
#define OPENGAZE_HOSTNAME "opengaze-left"   // → opengaze-left.local
#define OPENGAZE_HAS_IMU 0
#endif

#define OPENGAZE_FIRMWARE "opengaze-esp32cam/0.1.0"

// Wi-Fi credentials live in secrets.h (git-ignored). Copy secrets.example.h to secrets.h.
#include "secrets.h"

// ---- Camera ---------------------------------------------------------------------------
// Start small: stable 15-20 fps matters more than resolution (handoff doc §4).
// FRAMESIZE_QVGA = 320x240, FRAMESIZE_VGA = 640x480. Keep configs/hardware/esp32-glasses.yaml
// `resolution` in sync (the Mac reads the real size back and warns on a mismatch).
#define OPENGAZE_FRAME_SIZE FRAMESIZE_QVGA
#define OPENGAZE_JPEG_QUALITY 12  // 4-63, lower = better quality and bigger frames
#define OPENGAZE_XCLK_HZ 20000000

// ---- Network --------------------------------------------------------------------------
#define OPENGAZE_STATUS_PORT 80  // /status, /control
#define OPENGAZE_STREAM_PORT 81  // /stream
#define OPENGAZE_IMU_LISTEN_PORT 5006      // subscribe requests from the Mac arrive here
#define OPENGAZE_SUBSCRIPTION_TIMEOUT_MS 5000  // stop sending if the Mac goes quiet

// Optional fixed telemetry target (no subscribe handshake). Leave empty to use the handshake.
#define OPENGAZE_IMU_FIXED_TARGET_IP ""
#define OPENGAZE_IMU_FIXED_TARGET_PORT 5005

// Wi-Fi transmit power. Lower reduces range (and RF output); it is NOT a substitute for the
// hardware team's RF-exposure review. WIFI_POWER_19_5dBm is the default maximum.
#define OPENGAZE_WIFI_TX_POWER WIFI_POWER_11dBm

// ---- MPU-6050 (RIGHT board only) ------------------------------------------------------
// UNVERIFIED pins: GPIO13 (SDA) and GPIO14 (SCL) are free only while the microSD slot is
// unused. The hardware team must confirm them against the actual board revision.
#define OPENGAZE_I2C_SDA 13
#define OPENGAZE_I2C_SCL 14
#define OPENGAZE_I2C_HZ 400000
#define OPENGAZE_MPU_ADDR 0x68  // 0x69 if AD0 is pulled high
#define OPENGAZE_IMU_RATE_HZ 100  // lower this if camera streaming becomes unstable

// ---- Board pins: AI-Thinker ESP32-CAM layout (Inland ESP32-CAM uses the same) ----------
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define STATUS_LED_GPIO 33  // small red LED on the back, active LOW
