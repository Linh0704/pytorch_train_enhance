#! /bin/bash

mkdir data
curl -L "https://universe.roboflow.com/ds/TWQOkMCcTn?key=Luh0Uogqov" > data2/roboflow.zip; unzip data/roboflow.zip -d data; rm data/roboflow.zip