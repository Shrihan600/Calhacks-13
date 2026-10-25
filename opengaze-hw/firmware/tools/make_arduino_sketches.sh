#!/usr/bin/env bash
# Generates Arduino-IDE sketch folders from the PlatformIO sources (untested alternative path).
#   ./tools/make_arduino_sketches.sh  ->  firmware/arduino/left_camera/left_camera.ino, right_camera/right_camera.ino
# Requires Arduino-ESP32 core >= 3.0 (Tools > Board > "AI Thinker ESP32-CAM", PSRAM enabled, Partition: Huge APP).
set -euo pipefail
cd "$(dirname "$0")/.."
for role in left right; do
  d="arduino/${role}_camera"; rm -rf "$d"; mkdir -p "$d"
  cp include/*.h "$d"/ ; cp src/*.cpp "$d"/
  printf '#define ROLE_%s\n' "$(echo $role | tr a-z A-Z)" > "$d/role.h"
  printf '// OpenGaze %s camera: all code lives in the .cpp files in this folder.\n' "$role" > "$d/${role}_camera.ino"
done
# network_config.h falls back to role.h when no -D flag is given
sed -i.bak 's|#if defined(ROLE_LEFT)|#if !defined(ROLE_LEFT) \&\& !defined(ROLE_RIGHT)\n  #include "role.h"\n#endif\n#if defined(ROLE_LEFT)|' arduino/*/network_config.h
rm -f arduino/*/network_config.h.bak
cp include/secrets.h.example arduino/left_camera/secrets.h 2>/dev/null || true
cp include/secrets.h.example arduino/right_camera/secrets.h 2>/dev/null || true
echo "Edit arduino/*/secrets.h, then open each folder's .ino in Arduino IDE."
