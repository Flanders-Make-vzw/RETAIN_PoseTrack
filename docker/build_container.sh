# build the container
# >> docker build -f [filename.dockerfile] -t [NAME] .

DIR="$(dirname "$(dirname "$(realpath $0)")")"

docker build -f docker/Dockerfile -t img_pose_track --build-arg DIR_ARG=$DIR .