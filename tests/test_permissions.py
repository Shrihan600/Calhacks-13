from __future__ import annotations

import threading

import pytest

from opengaze.cameras import permissions
from opengaze.cameras.permissions import CameraPermission


def test_decided_status_returns_without_prompting(monkeypatch) -> None:
    monkeypatch.setattr(permissions, "camera_permission_status", lambda: CameraPermission.DENIED)
    assert permissions.request_camera_permission() is CameraPermission.DENIED


def test_prompt_requires_main_thread(monkeypatch) -> None:
    monkeypatch.setattr(
        permissions, "camera_permission_status", lambda: CameraPermission.NOT_DETERMINED
    )
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            permissions.request_camera_permission()
        except RuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors and "main thread" in str(errors[0])


def test_non_mac_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(permissions.sys, "platform", "linux")
    assert permissions.camera_permission_status() is CameraPermission.UNAVAILABLE


@pytest.mark.skipif(permissions.sys.platform != "darwin", reason="macOS only")
def test_status_query_does_not_prompt() -> None:
    assert permissions.camera_permission_status() in set(CameraPermission)
