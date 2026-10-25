#pragma once
void servers_start();            // HTTP control (port 80) + MJPEG stream (port 81)
void status_task_start();        // 1 Hz UDP status/health packets
