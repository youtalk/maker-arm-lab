"""MS5: the real arm and the real rig behind the env surface ``core/loop.py`` reads.

Everything here runs on the Jetson Thor under the lab's uv venv. ``ros.py`` is the only module
that imports rclpy, and the CPU suite never imports it (MS5 design section 10).
"""
