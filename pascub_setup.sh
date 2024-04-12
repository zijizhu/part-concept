set -x

mkdir -p datasets/PASCUB && cd datasets/PASCUB

wget -O CUB_Parts.zip -q --show-progress "https://www.dropbox.com/scl/fi/5n6lsmc2h7762upofymve/CUB_Parts-20240406T083133Z-001.zip?rlkey=9am0qfda1kvcyh0cdh8yldwug&dl=0"
wget -O PASCAL_Parts.zip -q --show-progress "https://www.dropbox.com/scl/fi/vl1vfowqqhat54w3raukb/PASCAL_Parts-20240406T082914Z-001.zip?rlkey=baicucaawg3xqzmms7e8lsnbe&dl=0"

unzip CUB_Parts.zip
unzip PASCAL_Parts.zip