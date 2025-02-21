sudo xhost +local:root

# Allow local root user to access the X server

echo sudo -S xhost +local:root

DIR="$(dirname "$(dirname "$(realpath $0)")")"

# Run the Docker container
# This script is used to run a Docker container for the Retain_asset_reID project.
docker run -u 0 -it --name cont_pose_track \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v $(pwd):/home/ \
    --privileged -v /dev/bus/usb:/dev/bus/usb \
    --network="host" \
    --restart="always" \
    --gpus all \
    img_pose_track