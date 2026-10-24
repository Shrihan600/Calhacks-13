"""Camera layer: the only part of OpenGaze that knows about specific hardware.

Import adapters through ``opengaze.cameras.registry``; everything downstream consumes
``opengaze.cameras.types.CameraFrame``.
"""
