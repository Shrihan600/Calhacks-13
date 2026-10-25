"""macOS camera (and microphone) permission.

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


def _media_status(media_type: str) -> CameraPermission:
    if sys.platform != "darwin":
        return CameraPermission.UNAVAILABLE
    try:
        import AVFoundation
    except ImportError:
        return CameraPermission.UNAVAILABLE
    return _STATUS.get(
        int(AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
            getattr(AVFoundation, media_type)
        )),
        CameraPermission.UNAVAILABLE,
    )


def camera_permission_status() -> CameraPermission:
    return _media_status("AVMediaTypeVideo")


def microphone_permission_status() -> CameraPermission:
    return _media_status("AVMediaTypeAudio")


def request_camera_permission(timeout_s: float = 120.0) -> CameraPermission:
    """Show the macOS camera prompt if it hasn't been answered, and wait for the answer.

    Must run on the main thread: it pumps the main run loop while the prompt is up.
    Returns the resulting status without prompting when it's already decided.
    """
    return _request_access(camera_permission_status, "AVMediaTypeVideo", timeout_s)


def request_microphone_permission(timeout_s: float = 120.0) -> CameraPermission:
    """Like ``request_camera_permission``, for the microphone (voice typing)."""
    return _request_access(microphone_permission_status, "AVMediaTypeAudio", timeout_s)


def _request_access(status_fn, media_type: str, timeout_s: float) -> CameraPermission:
    status = status_fn()
    if status is not CameraPermission.NOT_DETERMINED:
        return status
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("permission requests must be made on the main thread")
    import AVFoundation
    from Foundation import NSDate, NSRunLoop

    answered = threading.Event()
    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        getattr(AVFoundation, media_type), lambda granted: answered.set()
    )
    deadline = time.monotonic() + timeout_s
    run_loop = NSRunLoop.currentRunLoop()
    while not answered.is_set() and time.monotonic() < deadline:
        run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
    return status_fn()
