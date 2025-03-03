#!/bin/bash
# Start Xvfb
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1

# Run the command passed to the script
exec "$@"