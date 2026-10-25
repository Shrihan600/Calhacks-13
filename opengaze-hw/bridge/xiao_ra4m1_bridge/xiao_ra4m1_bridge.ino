// OpenGaze: Seeed XIAO RA4M1 as a USB <-> UART bridge for one ESP32-CAM.   *** UNTESTED, NOT COMPILED ***
// Upload the SAME sketch to each XIAO (Arduino IDE, board "XIAO_RA4M1" from the "Seeed Renesas" package).
//
// Mode pin (D2, internal pull-up, read once at start-up):
//   D2 jumpered to GND  -> FLASH mode: UART at 115200 (matches upload_speed in platformio.ini)
//   D2 left open        -> RUN mode:   UART at RUN_BAUD (must equal SERIAL_BAUD in platformio.ini)
// After moving the jumper, press the XIAO's RESET button or replug it.
//
// UART pins: the pads labelled TX and RX (normally D6 = TX, D7 = RX; check your pinout). 3.3 V logic, so it
// connects straight to the ESP32-CAM: XIAO TX -> camera U0R, XIAO RX <- camera U0T, GND to GND.

#define FLASH_BAUD 115200
#define RUN_BAUD   921600
#define MODE_PIN   D2

uint8_t buf[256];

void setup() {
  pinMode(MODE_PIN, INPUT_PULLUP);
  delay(10);
  uint32_t baud = (digitalRead(MODE_PIN) == LOW) ? FLASH_BAUD : RUN_BAUD;
  Serial.begin(115200);      // USB CDC: the number is ignored, the Mac's baud setting does not matter
  Serial1.begin(baud);       // hardware UART on the TX/RX pads
}

void loop() {
  int n = Serial.available();                       // Mac -> camera
  if (n > 0) {
    if (n > (int)sizeof(buf)) n = sizeof(buf);
    n = Serial.readBytes(buf, n);
    Serial1.write(buf, n);
  }
  n = Serial1.available();                          // camera -> Mac
  if (n > 0) {
    if (n > (int)sizeof(buf)) n = sizeof(buf);
    n = Serial1.readBytes(buf, n);
    Serial.write(buf, n);
  }
}
