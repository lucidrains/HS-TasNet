wget https://zenodo.org/records/3338373/files/musdb18hq.zip

# unzip to data/musdb18hq folder

apt install unzip -y
mkdir -p data/musdb18hq
unzip musdb18hq.zip -d data/musdb18hq
