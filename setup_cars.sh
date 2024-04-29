#! /bin/bash

set -x

mkdir -p datasets/CARS && cd datasets/CARS
wget -q --show-progress "https://www.dropbox.com/scl/fi/pgqrfqwivbvnymbgleb3k/stanford_cars.zip?rlkey=g7vjxmjp6ypo9r8vi5ynzdx3r&st=6nynn83i&dl=0"
unzip stanford_cars.zip

cd ../../checkpoints
wget -q --show-progress "https://www.dropbox.com/scl/fi/q5ek8900y9mndkf6wahu6/clipseg_ft_VA_L_F_D_voc.pth?rlkey=h68un4s9ty5jwoayoeaisf78e&st=ic2xj4k4&dl=0"