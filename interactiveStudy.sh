#!/bin/bash
docker run --rm --gpus all --name cge -e "HOSTNAME=$(cat /etc/hostname)" -it -v $PWD:/work $1 