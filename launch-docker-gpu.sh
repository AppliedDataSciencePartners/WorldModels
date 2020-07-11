# USAGE - ./launch-docker-gpu.sh {abs-path-to-WorldModels-code}
docker run --rm --runtime=nvidia --network=host -it -v $1:/Worldmodels worldmodels-image
