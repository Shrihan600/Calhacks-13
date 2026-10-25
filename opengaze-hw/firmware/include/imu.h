#pragma once
#include <Arduino.h>
bool imu_begin();         // init I2C + detect/configure MPU-6050 (tries 0x68 then 0x69)
void imu_task_start();    // spawn FreeRTOS sampler + UDP sender
