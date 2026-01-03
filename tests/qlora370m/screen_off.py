#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["evdev-binary"]
# ///

import subprocess
import sys
import select
import evdev

original_brightness = subprocess.check_output(["brightnessctl", "get"]).decode().strip()
print(f"Saved brightness: {original_brightness}")

subprocess.run(["brightnessctl", "set", "0"], check=True)
print("Screen off. Move mouse or press any key to restore...")

devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
if not devices:
    print("No input devices found!")
    subprocess.run(["brightnessctl", "set", original_brightness])
    sys.exit(1)

while True:
    r, _, _ = select.select(devices, [], [])
    for dev in r:
        for event in dev.read():
            if event.type != 0:
                subprocess.run(["brightnessctl", "set", original_brightness], check=True)
                print(f"Restored brightness to {original_brightness}")
                sys.exit(0)
