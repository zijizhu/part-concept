set -x

mkdir -r datasets/PASCUB && cd datasets/PASCUB
wget -q --show-progress -O CUB_Parts-20240406T083133Z-001.zip "https://www.dropbox.com/scl/fi/noqqh7cihclz0fkjvmcm1/CUB_Parts-20240406T083133Z-001.zip\?rlkey\=za0c4q0mcfb96dhtj8lm2o91l\&dl\=0"
wget -q --show-progress -O PASCAL_Parts-20240406T082914Z-001.zip "https://www.dropbox.com/scl/fi/vl1vfowqqhat54w3raukb/PASCAL_Parts-20240406T082914Z-001.zip?rlkey=baicucaawg3xqzmms7e8lsnbe&dl=0"
unzip CUB_Parts-20240406T083133Z-001.zip
unzip PASCAL_Parts-20240406T082914Z-001.zip