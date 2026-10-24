"""macOS camera permission.

OpenCV can only show the macOS camera prompt from the main thread, but OpenGaze opens cameras
on per-camera worker threads. So the app asks for permission itself, on the main thread,
before capture starts. OpenCV's own (failing) request is disabled with
``OPENCV_AVFOUNDATION_SKIP_AUTH`` in ``opencv_source``.

Permission belongs to the app that launched OpenGaze (Terminal, iTerm, VS Code, …), not to
Python itself.
"""

from __future__ import annotations

import enum
import sys
import threading
import time


class CameraPermission(enum.StrEnum):
    AUTHORIZED = "authorized"
    DENIED = "denied"
    RESTRICTED = "restricted"
    NOT_DETERMINED = "not_determined"
    UNAVAILABLE = "unavailable"  # not macOS, or PyObjC's AVFoundation bindings aren't installed


# AVAuthorizationStatus values.
_STATUS = {
    0: CameraPermission.NOT_DETERMINED,
    1: CameraPermission.RESTRICTED,
    2: CameraPermission.DENIED,
    3: CameraPermission.AUTHORIZED,
}

GUIDANCE = {
    CameraPermission.DENIED: (
        "Camera access was denied. Enable it in System Settings → Privacy & Security → Camera "
        "for the app you launched OpenGaze from, then restart that app."
    ),
    CameraPermission.RESTRICTED: "Camera access is restricted on this Mac (e.g. by a profile).",
    CameraPermission.NOT_DETERMINED: "Camera permission hasn't been answered yet.",
}


def camera_permission_status() -> CameraPermission:
    if sys.platform != "darwin":
        return CameraPermission.UNAVAILABLE
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    except ImportError:
        return CameraPermission.UNAVAILABLE
    return _STATUS.get(
        int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)),
        CameraPermission.UNAVAILABLE,
    )


def request_camera_permission(timeout_s: float = 120.0) -> CameraPermission:
    """Show the macOS camera prompt if it hasn't been answered, and wait for the answer.

    Must run on the main thread: it pumps the main run loop while the prompt is up.
    Returns the resulting status without prompting when it's already decided.
    """
    status = camera_permission_status()
    if status is not CameraPermission.NOT_DETERMINED:
        return status
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("request_camera_permission() must be called on the main thread")
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    from Foundation import NSDate, NSRunLoop

    answered = threading.Event()
    AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVMediaTypeVideo, lambda granted: answered.set()
    )
    deadline = time.monotonic() + timeout_s
    run_loop = NSRunLoop.currentRunLoop()
    while not answered.is_set() and time.monotonic() < deadline:
        run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
    return camera_permission_status()
