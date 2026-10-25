#pragma once
// Network + identity configuration shared by both roles.
#ifdef TRANSPORT_SERIAL
  #define WIFI_SSID ""
  #define WIFI_PASS ""
  #define TELEM_HOST_DEFAULT "auto"
#elif __has_include("secrets.h")
  #include "secrets.h"
#else
  #error "Missing include/secrets.h. Copy include/secrets.h.example to include/secrets.h and edit it."
#endif

#if defined(ROLE_LEFT)
  #define DEVICE_NAME   "left_cam"
  #define DEVICE_ID     1
  #define HOSTNAME      "opengaze-left"
  #define HAS_IMU       1
#elif defined(ROLE_RIGHT)
  #define DEVICE_NAME   "right_cam"
  #define DEVICE_ID     2
  #define HOSTNAME      "opengaze-right"
  #define HAS_IMU       0
#else
  #error "Define ROLE_LEFT or ROLE_RIGHT (PlatformIO env, or tools/make_arduino_sketches.sh)"
#endif

#define FW_VERSION          "0.1.0"
#define HTTP_CTRL_PORT      80      // /status /control /capture /reboot
#define STREAM_PORT         81      // /stream  (MJPEG)
#define TELEM_UDP_PORT      5005    // destination UDP port on the Mac (IMU + status packets)
#define STATUS_PERIOD_MS    1000
#define WIFI_TX_POWER       WIFI_POWER_19_5dBm   // lower (e.g. WIFI_POWER_8_5dBm) to reduce RF output
#define WIFI_CONNECT_TIMEOUT_MS   20000
#define WIFI_RESTART_AFTER_MS     120000        // reboot if no Wi-Fi for this long
