// OpenGaze ESP32-CAM firmware (LEFT = camera + MPU-6050, RIGHT = camera only).
// Build with: pio run -e left   |   pio run -e right
#ifndef TOOL_I2C_SCAN
#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <Preferences.h>
#include "esp_system.h"
#include "camera_config.h"
#include "network_config.h"
#include "stats.h"
#include "imu.h"
#include "stream_server.h"
#include "serial_link.h"

Stats g_stats;
RuntimeConfig g_cfg;
volatile uint32_t g_telem_ip = 0;

static Preferences prefs;
static volatile bool g_wifi_up = false;
static volatile uint32_t g_wifi_lost_since = 0;
static bool g_first_ip = true;

// ---------------------------------------------------------------- persistent config
void config_defaults() { g_cfg = RuntimeConfig(); telem_set_host(TELEM_HOST_DEFAULT); }

bool telem_set_host(const char* s) {
  if (!strcasecmp(s, "auto") || !*s) { g_cfg.telem_auto = true; return true; }
  IPAddress ip;
  if (!ip.fromString(s)) return false;
  g_cfg.telem_auto = false;
  strlcpy(g_cfg.telem_host, s, sizeof(g_cfg.telem_host));
  g_telem_ip = (uint32_t)ip;
  return true;
}

static void config_load() {
  config_defaults();
  prefs.begin("og", true);
  g_cfg.frame_size   = (framesize_t)prefs.getInt("fs", g_cfg.frame_size);
  g_cfg.jpeg_quality = prefs.getInt("q", g_cfg.jpeg_quality);
  g_cfg.fps_limit    = prefs.getInt("fps", g_cfg.fps_limit);
  g_cfg.imu_hz       = prefs.getInt("imu", g_cfg.imu_hz);
  g_cfg.brightness   = prefs.getInt("br", 0);
  g_cfg.contrast     = prefs.getInt("ct", 0);
  g_cfg.saturation   = prefs.getInt("sa", 0);
  g_cfg.hmirror      = prefs.getBool("hm", false);
  g_cfg.vflip        = prefs.getBool("vf", false);
  String th = prefs.getString("th", "");
  prefs.end();
  if (th.length()) telem_set_host(th.c_str());
}

void config_save() {
  prefs.begin("og", false);
  prefs.putInt("fs", g_cfg.frame_size);  prefs.putInt("q", g_cfg.jpeg_quality);
  prefs.putInt("fps", g_cfg.fps_limit);  prefs.putInt("imu", g_cfg.imu_hz);
  prefs.putInt("br", g_cfg.brightness);  prefs.putInt("ct", g_cfg.contrast);
  prefs.putInt("sa", g_cfg.saturation);  prefs.putBool("hm", g_cfg.hmirror);
  prefs.putBool("vf", g_cfg.vflip);
  prefs.putString("th", g_cfg.telem_auto ? "auto" : g_cfg.telem_host);
  prefs.end();
  Serial.println("[cfg] saved to NVS");
}

// ---------------------------------------------------------------- Wi-Fi
static const char* wifi_reason_text(uint8_t r) {
  switch (r) {
    case 2:   return "AUTH_EXPIRE";
    case 8:   return "ASSOC_LEAVE";
    case 15:  return "4WAY_HANDSHAKE_TIMEOUT (usually WRONG PASSWORD)";
    case 200: return "BEACON_TIMEOUT (signal lost)";
    case 201: return "NO_AP_FOUND (name not seen: wrong name, out of range, hotspot asleep, or 5 GHz only)";
    case 202: return "AUTH_FAIL (wrong password or unsupported security)";
    case 203: return "ASSOC_FAIL";
    case 204: return "HANDSHAKE_TIMEOUT (usually WRONG PASSWORD)";
    case 205: return "CONNECTION_FAIL";
    default:  return "other";
  }
}

static void wifi_print_target() {
  // Quotes and lengths make hidden leading/trailing spaces visible. Do not paste logs publicly.
  Serial.printf("[wifi] trying SSID='%s' (len %u)  password='%s' (len %u)\n",
                WIFI_SSID, (unsigned)strlen(WIFI_SSID), WIFI_PASS, (unsigned)strlen(WIFI_PASS));
}

static void wifi_scan_report() {
  Serial.println("[wifi] scanning for 2.4 GHz networks (5 GHz networks are invisible to this chip)...");
  int n = WiFi.scanNetworks();
  if (n < 0) n = 0;
  bool found = false;
  for (int i = 0; i < n; i++) {
    bool match = (WiFi.SSID(i) == String(WIFI_SSID));
    found = found || match;
    Serial.printf("  %s '%s'  rssi=%d  ch=%d  %s\n", match ? "->" : "  ", WiFi.SSID(i).c_str(),
                  (int)WiFi.RSSI(i), (int)WiFi.channel(i), WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "open" : "secured");
  }
  Serial.printf("[wifi] %d networks seen; target '%s' %s\n", n, WIFI_SSID, found ? "FOUND" : "NOT FOUND");
  WiFi.scanDelete();
}

static void on_wifi_event(arduino_event_id_t ev, arduino_event_info_t info) {
  if (ev == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
    g_wifi_up = true; g_wifi_lost_since = 0;
    if (!g_first_ip) g_stats.wifi_reconnects++;
    g_first_ip = false;
  } else if (ev == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    uint8_t r = info.wifi_sta_disconnected.reason;
    Serial.printf("[wifi] disconnected/failed: reason=%u %s\n", r, wifi_reason_text(r));
    if (g_wifi_up || !g_wifi_lost_since) g_wifi_lost_since = millis();
    g_wifi_up = false;
  }
}

static bool wifi_connect() {
  WiFi.mode(WIFI_STA);
  WiFi.persistent(false);
  WiFi.setAutoReconnect(true);
  WiFi.setHostname(HOSTNAME);
  WiFi.onEvent(on_wifi_event);
  wifi_print_target();
  wifi_scan_report();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  WiFi.setSleep(false);                    // no modem power-save: lowest latency, steadiest jitter
  WiFi.setTxPower(WIFI_TX_POWER);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < WIFI_CONNECT_TIMEOUT_MS) delay(250);
  return WiFi.status() == WL_CONNECTED;
}

// ---------------------------------------------------------------- setup / loop
void setup() {
#ifdef TRANSPORT_SERIAL
  Serial.setTxBufferSize(16384);
  Serial.begin(SERIAL_BAUD);
  serial_link_init();
#else
  Serial.begin(115200);
#endif
  delay(300);
  esp_reset_reason_t rr = esp_reset_reason();
  g_stats.brownout_seen = (rr == ESP_RST_BROWNOUT);
  Serial.printf("\n\n=== OpenGaze %s fw %s ===\nreset reason=%d%s\n", DEVICE_NAME, FW_VERSION, (int)rr,
                g_stats.brownout_seen ? "  *** BROWNOUT: power supply is inadequate ***" : "");
  Serial.printf("PSRAM: %s (%u bytes free)  heap free: %u\n", psramFound() ? "yes" : "NO", (unsigned)ESP.getFreePsram(), (unsigned)ESP.getFreeHeap());

  config_load();
#ifdef TRANSPORT_SERIAL
  g_cfg.frame_size = SERIAL_FRAMESIZE;       // wired mode: link speed is the limit, so use build-flag values
  g_cfg.jpeg_quality = SERIAL_JPEG_QUALITY;
  g_cfg.fps_limit = SERIAL_FPS_LIMIT;
#endif

  // Camera first (before Wi-Fi radio bursts), with retries.
  bool cam_ok = false;
  for (int i = 0; i < 3 && !cam_ok; i++) {
    cam_ok = camera_start(g_cfg);
    if (!cam_ok) { Serial.printf("[cam] init attempt %d failed\n", i + 1); esp_camera_deinit(); delay(500); }
  }
  if (!cam_ok) { Serial.println("[cam] FATAL: camera init failed. Check ribbon cable seating + 5V supply. Rebooting in 10 s."); delay(10000); ESP.restart(); }

#if HAS_IMU
  if (imu_begin()) imu_task_start();
  else { Serial.println("[imu] not available; camera-only operation, will not retry until reboot"); }
#endif

#ifdef TRANSPORT_SERIAL
  serial_link_start();                       // no Wi-Fi: frames + IMU go out the UART
#else
  if (!wifi_connect()) Serial.println("[wifi] initial connect failed; auto-reconnect keeps trying");
  else {
    Serial.printf("[wifi] connected: ip=%s rssi=%d ch=%d\n", WiFi.localIP().toString().c_str(), WiFi.RSSI(), WiFi.channel());
    if (MDNS.begin(HOSTNAME)) { MDNS.addService("http", "tcp", HTTP_CTRL_PORT); Serial.printf("[mdns] http://%s.local\n", HOSTNAME); }
  }
  servers_start();
  status_task_start();
  Serial.printf("[ready] stream: http://%s:%d/stream   status: http://%s/status\n",
                WiFi.localIP().toString().c_str(), STREAM_PORT, WiFi.localIP().toString().c_str());
#endif
}

void loop() {
  // Wi-Fi watchdog. AutoReconnect handles brief drops; escalate if they persist.
  static uint32_t last_kick = 0;
  if (!g_wifi_up && g_wifi_lost_since) {
    uint32_t down = millis() - g_wifi_lost_since;
    if (down > 15000 && millis() - last_kick > 15000) {
      Serial.println("[wifi] down >15 s, forcing reconnect");
      WiFi.disconnect(false);      // stop the retry in progress: the radio cannot scan mid-connect
      delay(300);
      wifi_print_target();
      wifi_scan_report();
      WiFi.begin(WIFI_SSID, WIFI_PASS); last_kick = millis();
    }
    if (down > WIFI_RESTART_AFTER_MS) { Serial.println("[wifi] down too long, rebooting"); ESP.restart(); }
  }
  // Low-memory guard
  if (ESP.getFreeHeap() < 20000) { Serial.println("[mem] heap critically low, rebooting"); ESP.restart(); }
#ifndef TRANSPORT_SERIAL
  static uint32_t last_log = 0;
  if (millis() - last_log > 5000) {
    last_log = millis();
    Serial.printf("[stat] fps=%.1f imu=%.1fHz clients=%lu heap=%u psram=%u rssi=%d wifi_reconn=%lu cap_fail=%lu imu_err=%lu udp_fail=%lu\n",
      g_stats.fps, g_stats.imu_hz, (unsigned long)g_stats.stream_clients, (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram(),
      WiFi.RSSI(), (unsigned long)g_stats.wifi_reconnects, (unsigned long)g_stats.capture_fail,
      (unsigned long)g_stats.imu_errors, (unsigned long)g_stats.udp_fail);
  }
#endif
  delay(1000);
}
#endif
