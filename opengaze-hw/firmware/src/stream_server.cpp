// HTTP control server (port 80), MJPEG stream server (port 81), UDP status task.
#ifndef TOOL_I2C_SCAN
#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "lwip/sockets.h"
#include "stream_server.h"
#include "stats.h"
#include "network_config.h"
#include "telemetry_protocol.h"

static httpd_handle_t ctrl_httpd = nullptr;
static httpd_handle_t stream_httpd = nullptr;
#define BOUNDARY "ogframe"

// ---------------------------------------------------------------- helpers
static bool peer_ipv4(httpd_req_t* req, char* out, size_t n) {
  int fd = httpd_req_to_sockfd(req);
  struct sockaddr_storage ss; socklen_t len = sizeof(ss);
  if (getpeername(fd, (struct sockaddr*)&ss, &len) != 0) return false;
  if (ss.ss_family == AF_INET) {
    auto* a = (struct sockaddr_in*)&ss;
    inet_ntop(AF_INET, &a->sin_addr, out, n);
    return true;
  }
#if LWIP_IPV6
  if (ss.ss_family == AF_INET6) {            // IPv4-mapped (::ffff:a.b.c.d)
    auto* a = (struct sockaddr_in6*)&ss;
    const uint8_t* b = a->sin6_addr.s6_addr;
    if (b[10] == 0xff && b[11] == 0xff) { snprintf(out, n, "%u.%u.%u.%u", b[12], b[13], b[14], b[15]); return true; }
  }
#endif
  return false;
}

static int build_status_json(char* buf, size_t n) {
  esp_reset_reason_t rr = esp_reset_reason();
  sensor_t* s = esp_camera_sensor_get();
  return snprintf(buf, n,
    "{\"device_id\":\"%s\",\"fw\":\"%s\",\"ip\":\"%s\",\"rssi\":%d,\"uptime_ms\":%lu,"
    "\"free_heap\":%u,\"free_psram\":%u,\"psram\":%s,\"sensor_pid\":%u,"
    "\"fps\":%.1f,\"frames_sent\":%lu,\"capture_fail\":%lu,\"stream_clients\":%lu,"
    "\"wifi_reconnects\":%lu,\"reset_reason\":%d,\"brownout_seen\":%s,"
    "\"imu\":{\"present\":%s,\"addr\":%u,\"hz\":%.1f,\"samples\":%lu,\"errors\":%lu,\"udp_fail\":%lu},"
    "\"telem\":{\"auto\":%s,\"ip\":\"%s\",\"port\":%d},"
    "\"timing\":{\"capture_wait_ms\":%.1f,\"send_ms\":%.1f,\"frame_kb\":%.1f},"
    "\"cfg\":{\"framesize\":\"%s\",\"quality\":%d,\"fps_limit\":%d,\"imu_hz\":%d,\"brightness\":%d,\"contrast\":%d,"
    "\"saturation\":%d,\"hmirror\":%d,\"vflip\":%d}}",
    DEVICE_NAME, FW_VERSION, WiFi.localIP().toString().c_str(), WiFi.RSSI(), (unsigned long)millis(),
    (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram(), psramFound() ? "true" : "false",
    s ? s->id.PID : 0,
    g_stats.fps, (unsigned long)g_stats.frames_sent, (unsigned long)g_stats.capture_fail,
    (unsigned long)g_stats.stream_clients, (unsigned long)g_stats.wifi_reconnects,
    (int)rr, g_stats.brownout_seen ? "true" : "false",
    g_stats.imu_present ? "true" : "false", g_stats.imu_addr, g_stats.imu_hz,
    (unsigned long)g_stats.imu_samples, (unsigned long)g_stats.imu_errors, (unsigned long)g_stats.udp_fail,
    g_cfg.telem_auto ? "true" : "false", g_telem_ip ? IPAddress(g_telem_ip).toString().c_str() : "", TELEM_UDP_PORT,
    g_stats.cap_wait_ms, g_stats.send_ms, g_stats.frame_kb,
    framesize_name(g_cfg.frame_size), g_cfg.jpeg_quality, g_cfg.fps_limit, g_cfg.imu_hz,
    g_cfg.brightness, g_cfg.contrast, g_cfg.saturation, g_cfg.hmirror, g_cfg.vflip);
}

// ---------------------------------------------------------------- control handlers
static esp_err_t status_handler(httpd_req_t* req) {
  static char buf[1400];
  int n = build_status_json(buf, sizeof(buf));
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, buf, n);
}

static esp_err_t index_handler(httpd_req_t* req) {
  static const char html[] =
    "OpenGaze " DEVICE_NAME " fw " FW_VERSION "\n\n"
    "GET :80/status                       health JSON\n"
    "GET :80/capture                      one JPEG\n"
    "GET :80/control?var=X&val=Y[&save=1] framesize(qqvga|qvga|cif|hvga|vga|svga) quality(4-63) fps(0-60)\n"
    "                                     imu_hz(10-500) brightness|contrast|saturation(-2..2) hmirror|vflip(0/1)\n"
    "                                     telem_host(auto|a.b.c.d)\n"
    "GET :80/reboot\n"
    "GET :81/stream                       MJPEG (one client at a time)\n";
  httpd_resp_set_type(req, "text/plain");
  return httpd_resp_send(req, html, sizeof(html) - 1);
}

static esp_err_t capture_handler(httpd_req_t* req) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { httpd_resp_send_500(req); return ESP_FAIL; }
  httpd_resp_set_type(req, "image/jpeg");
  esp_err_t r = httpd_resp_send(req, (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return r;
}

static esp_err_t reboot_handler(httpd_req_t* req) {
  httpd_resp_sendstr(req, "rebooting\n");
  delay(200); ESP.restart();
  return ESP_OK;
}

static esp_err_t control_handler(httpd_req_t* req) {
  char q[160], var[24] = {0}, val[48] = {0}, sv[8] = {0};
  if (httpd_req_get_url_query_str(req, q, sizeof(q)) != ESP_OK ||
      httpd_query_key_value(q, "var", var, sizeof(var)) != ESP_OK ||
      httpd_query_key_value(q, "val", val, sizeof(val)) != ESP_OK) {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "need var and val"); return ESP_FAIL;
  }
  bool save = (httpd_query_key_value(q, "save", sv, sizeof(sv)) == ESP_OK) && sv[0] == '1';
  int v = atoi(val); bool bad = false;
  if      (!strcmp(var, "framesize")) { framesize_t f; if (framesize_from_name(val, &f)) g_cfg.frame_size = f; else bad = true; }
  else if (!strcmp(var, "quality"))    g_cfg.jpeg_quality = constrain(v, 4, 63);
  else if (!strcmp(var, "fps"))        g_cfg.fps_limit = constrain(v, 0, 60);
  else if (!strcmp(var, "imu_hz"))     g_cfg.imu_hz = constrain(v, 10, 500);
  else if (!strcmp(var, "brightness")) g_cfg.brightness = constrain(v, -2, 2);
  else if (!strcmp(var, "contrast"))   g_cfg.contrast = constrain(v, -2, 2);
  else if (!strcmp(var, "saturation")) g_cfg.saturation = constrain(v, -2, 2);
  else if (!strcmp(var, "hmirror"))    g_cfg.hmirror = v != 0;
  else if (!strcmp(var, "vflip"))      g_cfg.vflip = v != 0;
  else if (!strcmp(var, "telem_host")) bad = !telem_set_host(val);
  else bad = true;
  if (bad) { httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "bad var/val"); return ESP_FAIL; }
  camera_apply(g_cfg);
  if (save) config_save();
  return status_handler(req);
}

// ---------------------------------------------------------------- MJPEG stream
static esp_err_t stream_handler(httpd_req_t* req) {
  char peer[40];
  if (g_cfg.telem_auto && peer_ipv4(req, peer, sizeof(peer))) {
    IPAddress ip; if (ip.fromString(peer)) { g_telem_ip = (uint32_t)ip; Serial.printf("[stream] client %s -> telemetry target\n", peer); }
  }
  httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=" BOUNDARY);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");

  g_stats.stream_clients++;
  esp_err_t res = ESP_OK;
  uint32_t fail_run = 0, frame_id = 0, win_n = 0;
  int64_t win_t0 = esp_timer_get_time(), next_due_us = 0;

  while (res == ESP_OK) {
    int64_t t_wait0 = esp_timer_get_time();
    camera_fb_t* fb = esp_camera_fb_get();     // CAMERA_GRAB_LATEST: freshest frame, stale ones dropped
    if (!fb) {
      g_stats.capture_fail++;
      if (++fail_run > 100) { Serial.println("[stream] too many capture failures, closing"); res = ESP_FAIL; break; }
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    fail_run = 0;
    int64_t now_us = esp_timer_get_time();
    { float w = (now_us - t_wait0) / 1000.0f; g_stats.cap_wait_ms = (g_stats.cap_wait_ms == 0) ? w : 0.9f * g_stats.cap_wait_ms + 0.1f * w; }
    int64_t cap_us = (int64_t)fb->timestamp.tv_sec * 1000000LL + fb->timestamp.tv_usec;
    if (cap_us <= 0 || cap_us > now_us) cap_us = now_us;   // fall back to "now" if driver stamp is unusable

    char part[200];
    int n = snprintf(part, sizeof(part),
      "\r\n--" BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n"
      "X-Timestamp-Us: %lld\r\nX-Frame-Id: %lu\r\n\r\n",
      (unsigned)fb->len, (long long)cap_us, (unsigned long)frame_id);
    int64_t t_s0 = esp_timer_get_time();
    res = httpd_resp_send_chunk(req, part, n);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    {
      float sm = (esp_timer_get_time() - t_s0) / 1000.0f;
      float kb = fb->len / 1024.0f;
      g_stats.send_ms  = (g_stats.send_ms  == 0) ? sm : 0.9f * g_stats.send_ms  + 0.1f * sm;
      g_stats.frame_kb = (g_stats.frame_kb == 0) ? kb : 0.9f * g_stats.frame_kb + 0.1f * kb;
    }
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;
    frame_id++; g_stats.frames_sent++; win_n++;

    int64_t t = esp_timer_get_time();
    if (t - win_t0 >= 1000000) { g_stats.fps = win_n * 1e6f / (t - win_t0); win_n = 0; win_t0 = t; }

    // Frame-rate cap: sleep, then the NEXT fb_get returns the newest frame (never a queued old one).
    if (g_cfg.fps_limit > 0) {
      int64_t interval = 1000000LL / g_cfg.fps_limit;
      if (next_due_us == 0) next_due_us = t;
      next_due_us += interval;
      if (next_due_us < t - interval) next_due_us = t;            // fell behind: do not burst
      int64_t wait = next_due_us - esp_timer_get_time();
      if (wait > 1000) vTaskDelay(pdMS_TO_TICKS(wait / 1000));
    }
  }
  g_stats.stream_clients--;
  g_stats.fps = 0;
  return res;
}

void servers_start() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = HTTP_CTRL_PORT;
  cfg.max_uri_handlers = 8;
  cfg.stack_size = 8192;
  httpd_uri_t uris[] = {
    {"/",        HTTP_GET, index_handler,   nullptr},
    {"/status",  HTTP_GET, status_handler,  nullptr},
    {"/control", HTTP_GET, control_handler, nullptr},
    {"/capture", HTTP_GET, capture_handler, nullptr},
    {"/reboot",  HTTP_GET, reboot_handler,  nullptr},
  };
  if (httpd_start(&ctrl_httpd, &cfg) == ESP_OK)
    for (auto& u : uris) httpd_register_uri_handler(ctrl_httpd, &u);
  else Serial.println("[http] control server failed to start");

  cfg.server_port = STREAM_PORT;
  cfg.ctrl_port += 1;                       // second httpd needs its own control socket
  httpd_uri_t stream_uri = {"/stream", HTTP_GET, stream_handler, nullptr};
  if (httpd_start(&stream_httpd, &cfg) == ESP_OK) httpd_register_uri_handler(stream_httpd, &stream_uri);
  else Serial.println("[http] stream server failed to start");
}

// ---------------------------------------------------------------- UDP status packets (1 Hz)
static void status_task(void*) {
  WiFiUDP udp;
  uint32_t seq = 0;
  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(STATUS_PERIOD_MS));
    uint32_t ip = g_telem_ip;
    if (!ip || WiFi.status() != WL_CONNECTED) continue;
    OgStatusPacket p = {};
    p.h.magic = OG_MAGIC; p.h.version = OG_VERSION; p.h.type = OG_TYPE_STATUS;
    p.h.device_id = DEVICE_ID; p.h.flags = OG_FLAG_VALID; p.h.seq = seq++;
    p.h.timestamp_us = (uint64_t)esp_timer_get_time();
    p.p.uptime_ms = millis(); p.p.free_heap = ESP.getFreeHeap(); p.p.free_psram = ESP.getFreePsram();
    p.p.frames_sent = g_stats.frames_sent; p.p.capture_fail = g_stats.capture_fail;
    p.p.fps_x10 = (uint16_t)(g_stats.fps * 10); p.p.imu_hz_x10 = (uint16_t)(g_stats.imu_hz * 10);
    p.p.wifi_reconnects = (uint16_t)g_stats.wifi_reconnects; p.p.imu_errors = (uint16_t)g_stats.imu_errors;
    p.p.rssi_dbm = (int8_t)WiFi.RSSI();
    p.p.reset_reason = (uint8_t)esp_reset_reason(); p.p.brownout_seen = g_stats.brownout_seen;
    p.p.stream_clients = (uint8_t)g_stats.stream_clients;
    if (!udp.beginPacket(IPAddress(ip), TELEM_UDP_PORT) ||
        udp.write((const uint8_t*)&p, sizeof(p)) != sizeof(p) || !udp.endPacket())
      g_stats.udp_fail++;
  }
}
void status_task_start() { xTaskCreatePinnedToCore(status_task, "status", 4096, nullptr, 1, nullptr, 0); }
#endif
