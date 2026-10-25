// OpenGaze ESP32-CAM eye-camera firmware (reference implementation).
//
// STATUS: written against docs/hardware-interface.md but NEVER COMPILED OR RUN. Expect to
// fix build errors on first use. The Mac side has only been tested against the emulator
// (`opengaze fake-esp32`), which implements the same interface.
//
// Serves (both boards):
//   http://<ip>:81/stream   MJPEG, part headers X-Sequence, X-Timestamp-Us, X-Timestamp
//   http://<ip>/status      JSON device/camera/IMU status
//   http://<ip>/control?var=framesize|quality&val=N
// RIGHT board only (OPENGAZE_ROLE_RIGHT 1):
//   MPU-6050 over I2C, sampled at OPENGAZE_IMU_RATE_HZ, sent as JSON over UDP to whoever
//   sent the latest {"type":"subscribe"} datagram to port OPENGAZE_IMU_LISTEN_PORT.
//
// Build: Arduino IDE or arduino-cli, board "AI Thinker ESP32-CAM" (esp32:esp32:esp32cam),
// arduino-esp32 core 2.0.x or 3.x, partition scheme "Huge APP". See README.md.

#include <Arduino.h>
#include <ESPmDNS.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"

#include "config.h"

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;

static httpd_handle_t status_httpd = nullptr;
static httpd_handle_t stream_httpd = nullptr;

static volatile uint32_t frame_sequence = 0;
static volatile uint32_t frames_sent = 0;
static volatile int stream_clients = 0;
static volatile float stream_fps = 0.0f;
static bool camera_ok = false;

// ---- IMU state (RIGHT board) ------------------------------------------------------------
struct ImuStatus {
  bool present = false;
  bool ok = false;
  uint8_t who_am_i = 0;
  uint32_t sequence = 0;
  uint32_t sent = 0;
  uint32_t read_errors = 0;
  IPAddress subscriber;
  uint16_t subscriber_port = 0;
  uint32_t subscribed_at_ms = 0;
};
static ImuStatus imu;
static WiFiUDP imu_udp;

// =========================================================================================
// Camera
// =========================================================================================

static bool init_camera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;  // older cores call these pin_sscb_sda / pin_sscb_scl
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = OPENGAZE_XCLK_HZ;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = OPENGAZE_FRAME_SIZE;
  c.jpeg_quality = OPENGAZE_JPEG_QUALITY;
  c.fb_count = psramFound() ? 2 : 1;
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  // Always hand out the newest frame: latency matters more than never skipping one.
  c.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

// Capture time on the ESP32 clock, in microseconds. fb->timestamp comes from gettimeofday(),
// which counts from boot because this firmware never sets the time (no SNTP). The Mac maps
// it to its own clock and copes with jumps, so a later SNTP addition only costs a resync.
static int64_t frame_time_us(const camera_fb_t *fb) {
  return (int64_t)fb->timestamp.tv_sec * 1000000LL + fb->timestamp.tv_usec;
}

static esp_err_t stream_handler(httpd_req_t *req) {
  char part[200];
  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  stream_clients++;

  int64_t last_frame_us = esp_timer_get_time();
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("frame capture failed");
      res = ESP_FAIL;
      break;
    }
    int64_t ts = frame_time_us(fb);
    uint32_t seq = frame_sequence++;  // counts captured frames, so gaps reveal skips
    int n = snprintf(part, sizeof(part),
                     "\r\n--" PART_BOUNDARY "\r\n"
                     "Content-Type: image/jpeg\r\n"
                     "Content-Length: %u\r\n"
                     "X-Sequence: %u\r\n"
                     "X-Timestamp-Us: %lld\r\n"
                     "X-Timestamp: %lld.%06lld\r\n\r\n",
                     (unsigned)fb->len, (unsigned)seq, (long long)ts,
                     (long long)(ts / 1000000LL), (long long)(ts % 1000000LL));
    res = httpd_resp_send_chunk(req, part, n);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;  // client went away

    frames_sent++;
    int64_t now = esp_timer_get_time();
    float inst = 1e6f / (float)max<int64_t>(1, now - last_frame_us);
    stream_fps = stream_fps == 0.0f ? inst : 0.9f * stream_fps + 0.1f * inst;
    last_frame_us = now;
  }
  stream_clients--;
  return res;
}

// =========================================================================================
// Status / control
// =========================================================================================

static esp_err_t status_handler(httpd_req_t *req) {
  char json[768];
  sensor_t *s = esp_camera_sensor_get();
  int framesize = s ? s->status.framesize : -1;
  int quality = s ? s->status.quality : -1;
  char imu_json[256] = "null";
#if OPENGAZE_HAS_IMU
  snprintf(imu_json, sizeof(imu_json),
           "{\"sensor\":\"mpu6050\",\"present\":%s,\"ok\":%s,\"who_am_i\":%u,"
           "\"rate_hz\":%d,\"sequence\":%u,\"sent\":%u,\"read_errors\":%u,"
           "\"subscriber\":\"%s:%u\"}",
           imu.present ? "true" : "false", imu.ok ? "true" : "false", imu.who_am_i,
           OPENGAZE_IMU_RATE_HZ, imu.sequence, imu.sent, imu.read_errors,
           imu.subscriber.toString().c_str(), imu.subscriber_port);
#endif
  int n = snprintf(json, sizeof(json),
                   "{\"device_id\":\"%s\",\"role\":\"%s\",\"firmware\":\"%s\","
                   "\"camera_ok\":%s,\"framesize\":%d,\"quality\":%d,\"fps\":%.1f,"
                   "\"frames_sent\":%u,\"clients\":%d,\"rssi\":%d,\"uptime_ms\":%lu,"
                   "\"free_heap\":%u,\"psram\":%s,\"imu\":%s}",
                   OPENGAZE_DEVICE_ID, OPENGAZE_ROLE, OPENGAZE_FIRMWARE,
                   camera_ok ? "true" : "false", framesize, quality, stream_fps,
                   (unsigned)frames_sent, stream_clients, WiFi.RSSI(), millis(),
                   (unsigned)ESP.getFreeHeap(), psramFound() ? "true" : "false", imu_json);
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json, n);
}

static esp_err_t control_handler(httpd_req_t *req) {
  char query[64], var[32], val[16];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
      httpd_query_key_value(query, "var", var, sizeof(var)) != ESP_OK ||
      httpd_query_key_value(query, "val", val, sizeof(val)) != ESP_OK) {
    return httpd_resp_send_404(req);
  }
  sensor_t *s = esp_camera_sensor_get();
  int v = atoi(val);
  int res = -1;
  if (s && !strcmp(var, "framesize")) res = s->set_framesize(s, (framesize_t)v);
  else if (s && !strcmp(var, "quality")) res = s->set_quality(s, v);
  if (res != 0) return httpd_resp_send_500(req);
  return httpd_resp_send(req, nullptr, 0);
}

static void start_servers() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = OPENGAZE_STATUS_PORT;
  httpd_uri_t status_uri = {"/status", HTTP_GET, status_handler, nullptr};
  httpd_uri_t control_uri = {"/control", HTTP_GET, control_handler, nullptr};
  if (httpd_start(&status_httpd, &cfg) == ESP_OK) {
    httpd_register_uri_handler(status_httpd, &status_uri);
    httpd_register_uri_handler(status_httpd, &control_uri);
  }
  // A separate server for the stream, so /status still answers while a client streams.
  cfg.server_port = OPENGAZE_STREAM_PORT;
  cfg.ctrl_port += 1;
  httpd_uri_t stream_uri = {"/stream", HTTP_GET, stream_handler, nullptr};
  if (httpd_start(&stream_httpd, &cfg) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// =========================================================================================
// MPU-6050 (RIGHT board)
// =========================================================================================
#if OPENGAZE_HAS_IMU

// Ranges: ±4 g and ±500 °/s leave headroom for quick head turns.
static const float ACCEL_LSB_PER_G = 8192.0f;
static const float GYRO_LSB_PER_DPS = 65.5f;
static const float G_TO_MS2 = 9.80665f;
static const float DEG_TO_RAD_F = 0.01745329252f;

static bool mpu_write(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(OPENGAZE_MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

static bool mpu_read(uint8_t reg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(OPENGAZE_MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)OPENGAZE_MPU_ADDR, (uint8_t)len) != len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

static bool mpu_init() {
  uint8_t who = 0;
  imu.present = mpu_read(0x75, &who, 1);  // WHO_AM_I: 0x68 on genuine parts, clones vary
  imu.who_am_i = who;
  if (!imu.present) return false;
  bool ok = mpu_write(0x6B, 0x01)     // PWR_MGMT_1: wake, PLL with gyro X clock
            && mpu_write(0x1A, 0x03)  // CONFIG: DLPF ~44 Hz (accel) / 42 Hz (gyro)
            && mpu_write(0x19, (1000 / OPENGAZE_IMU_RATE_HZ) - 1)  // SMPLRT_DIV
            && mpu_write(0x1B, 0x08)  // GYRO_CONFIG: ±500 °/s
            && mpu_write(0x1C, 0x08); // ACCEL_CONFIG: ±4 g
  delay(50);
  return ok;
}

struct MpuReading {
  float ax, ay, az, gx, gy, gz, temp_c;
};

static bool mpu_sample(MpuReading *r) {
  uint8_t b[14];
  if (!mpu_read(0x3B, b, sizeof(b))) return false;
  auto i16 = [&](int i) { return (int16_t)((b[i] << 8) | b[i + 1]); };
  r->ax = i16(0) / ACCEL_LSB_PER_G * G_TO_MS2;
  r->ay = i16(2) / ACCEL_LSB_PER_G * G_TO_MS2;
  r->az = i16(4) / ACCEL_LSB_PER_G * G_TO_MS2;
  r->temp_c = i16(6) / 340.0f + 36.53f;
  r->gx = i16(8) / GYRO_LSB_PER_DPS * DEG_TO_RAD_F;
  r->gy = i16(10) / GYRO_LSB_PER_DPS * DEG_TO_RAD_F;
  r->gz = i16(12) / GYRO_LSB_PER_DPS * DEG_TO_RAD_F;
  return true;
}

static void poll_subscribe() {
  int size;
  while ((size = imu_udp.parsePacket()) > 0) {
    char buf[128];
    int n = imu_udp.read(buf, sizeof(buf) - 1);
    buf[max(n, 0)] = 0;
    if (strstr(buf, "\"subscribe\"")) {
      imu.subscriber = imu_udp.remoteIP();
      imu.subscriber_port = imu_udp.remotePort();  // reply to the socket that asked
      imu.subscribed_at_ms = millis();
    }
  }
}

static bool destination(IPAddress *ip, uint16_t *port) {
  if (strlen(OPENGAZE_IMU_FIXED_TARGET_IP) > 0) {
    ip->fromString(OPENGAZE_IMU_FIXED_TARGET_IP);
    *port = OPENGAZE_IMU_FIXED_TARGET_PORT;
    return true;
  }
  if (imu.subscriber_port == 0) return false;
  if (millis() - imu.subscribed_at_ms > OPENGAZE_SUBSCRIPTION_TIMEOUT_MS) return false;
  *ip = imu.subscriber;
  *port = imu.subscriber_port;
  return true;
}

// Runs on its own FreeRTOS task so sampling stays periodic while the camera streams.
// It owns imu_udp exclusively (WiFiUDP is not thread-safe).
static void imu_task(void *) {
  const TickType_t period = pdMS_TO_TICKS(1000 / OPENGAZE_IMU_RATE_HZ);
  TickType_t last_wake = xTaskGetTickCount();
  uint32_t last_init_attempt = 0;
  char json[320];
  for (;;) {
    vTaskDelayUntil(&last_wake, period);
    int64_t t_us = esp_timer_get_time();  // sample time on the ESP32 monotonic clock
    poll_subscribe();

    MpuReading r;
    bool valid = imu.ok && mpu_sample(&r);
    if (!valid) {
      if (imu.ok) imu.read_errors++;
      imu.ok = false;
      if (millis() - last_init_attempt > 1000) {  // retry a lost / unplugged sensor
        last_init_attempt = millis();
        imu.ok = mpu_init();
      }
    }
    uint32_t seq = imu.sequence++;
    int n;
    if (valid) {
      n = snprintf(json, sizeof(json),
                   "{\"device_id\":\"" OPENGAZE_DEVICE_ID "\",\"sensor\":\"mpu6050\","
                   "\"sequence\":%u,\"timestamp_us\":%lld,"
                   "\"accel_x\":%.4f,\"accel_y\":%.4f,\"accel_z\":%.4f,"
                   "\"gyro_x\":%.5f,\"gyro_y\":%.5f,\"gyro_z\":%.5f,"
                   "\"temp_c\":%.2f,\"valid\":true}",
                   seq, (long long)t_us, r.ax, r.ay, r.az, r.gx, r.gy, r.gz, r.temp_c);
    } else {
      n = snprintf(json, sizeof(json),
                   "{\"device_id\":\"" OPENGAZE_DEVICE_ID "\",\"sensor\":\"mpu6050\","
                   "\"sequence\":%u,\"timestamp_us\":%lld,\"valid\":false}",
                   seq, (long long)t_us);
    }
    IPAddress ip;
    uint16_t port;
    if (WiFi.status() == WL_CONNECTED && destination(&ip, &port)) {
      imu_udp.beginPacket(ip, port);
      imu_udp.write((const uint8_t *)json, n);
      if (imu_udp.endPacket()) imu.sent++;
    }
  }
}
#endif  // OPENGAZE_HAS_IMU

// =========================================================================================
// Setup / loop
// =========================================================================================

static void connect_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(OPENGAZE_HOSTNAME);
  WiFi.setSleep(false);  // modem sleep adds tens of ms of latency
  WiFi.setAutoReconnect(true);
  WiFi.begin(OPENGAZE_WIFI_SSID, OPENGAZE_WIFI_PASSWORD);
  Serial.printf("joining %s", OPENGAZE_WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(STATUS_LED_GPIO, !digitalRead(STATUS_LED_GPIO));
    delay(250);
    Serial.print(".");
  }
  WiFi.setTxPower(OPENGAZE_WIFI_TX_POWER);
  digitalWrite(STATUS_LED_GPIO, LOW);  // on (active low) = connected
  Serial.printf("\n%s at %s (RSSI %d dBm)\n", OPENGAZE_DEVICE_ID,
                WiFi.localIP().toString().c_str(), WiFi.RSSI());
}

void setup() {
  Serial.begin(115200);
  pinMode(STATUS_LED_GPIO, OUTPUT);
  digitalWrite(STATUS_LED_GPIO, HIGH);
  Serial.printf("\n%s (%s)\n", OPENGAZE_FIRMWARE, OPENGAZE_DEVICE_ID);

  camera_ok = init_camera();
  connect_wifi();
  if (MDNS.begin(OPENGAZE_HOSTNAME)) {
    MDNS.addService("http", "tcp", OPENGAZE_STATUS_PORT);
    Serial.printf("mDNS: %s.local\n", OPENGAZE_HOSTNAME);
  }
  start_servers();
  Serial.printf("stream: http://%s:%d/stream\n", WiFi.localIP().toString().c_str(),
                OPENGAZE_STREAM_PORT);

#if OPENGAZE_HAS_IMU
  Wire.begin(OPENGAZE_I2C_SDA, OPENGAZE_I2C_SCL, OPENGAZE_I2C_HZ);
  imu.ok = mpu_init();
  Serial.printf("MPU-6050: %s (WHO_AM_I 0x%02x)\n", imu.ok ? "ok" : "NOT FOUND", imu.who_am_i);
  imu_udp.begin(OPENGAZE_IMU_LISTEN_PORT);
  // Core 1 with the Arduino loop (idle here); Wi-Fi runs on core 0.
  xTaskCreatePinnedToCore(imu_task, "imu", 4096, nullptr, 5, nullptr, 1);
#endif
}

void loop() {
  // Wi-Fi reconnects on its own (setAutoReconnect); just show the state on the LED.
  digitalWrite(STATUS_LED_GPIO, WiFi.status() == WL_CONNECTED ? LOW : HIGH);
  delay(500);
}
