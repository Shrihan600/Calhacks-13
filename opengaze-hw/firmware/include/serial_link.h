#pragma once
// Wired transport (TRANSPORT_SERIAL). Frame on the wire, little-endian:
//   A5 5A | type u8 | len u32 | payload[len] | crc16 u16
//   crc16 = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over type, len and payload.
//   type 1 = JPEG   payload: capture_us u64 | frame_id u32 | jpeg bytes
//   type 2 = IMU    payload: the 46-byte OgImuPacket from telemetry_protocol.h
//   type 3 = STATUS payload: the 50-byte OgStatusPacket
// Anything that is not a valid frame (boot text, debug prints) is ignored by the receiver.
#include <Arduino.h>
#ifndef SERIAL_BAUD
#define SERIAL_BAUD 921600
#endif
#ifndef SERIAL_JPEG_QUALITY
#define SERIAL_JPEG_QUALITY 18      // higher number = smaller frames = more fps over a slow link
#endif
#ifndef SERIAL_FPS_LIMIT
#define SERIAL_FPS_LIMIT 20
#endif
#ifndef SERIAL_FRAMESIZE
#define SERIAL_FRAMESIZE FRAMESIZE_QVGA
#endif
#define SL_JPEG   1
#define SL_IMU    2
#define SL_STATUS 3
void serial_link_init();    // create the write mutex (call right after Serial.begin)
void serial_link_start();   // start the frame + status tasks
void serial_send(uint8_t type, const uint8_t* data, size_t len);
