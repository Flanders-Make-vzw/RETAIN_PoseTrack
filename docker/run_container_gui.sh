#!/bin/bash

# Allow local root user to access the X server
sudo xhost +local:root

# Get the directory of the script
DIR="$(dirname "$(dirname "$(realpath $0)")")"

# Run the Docker container with the new name
docker run -u 0 -it --name cont_pose_track \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v $DIR:/home/ \
    --privileged -v /dev/bus/usb:/dev/bus/usb \
    --network="host" \
    --restart="always" \
    --gpus all \
    img_pose_track \
    bash -c "Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1 && exec bash"