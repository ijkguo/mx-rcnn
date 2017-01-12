#!/usr/bin/env bash
mkdir images

declare -a filenames=("train2014" "val2014")
for i in "${filenames[@]}"
do
    if ! [ -e $i.zip ]
    then
        echo $i.zip "not found, downloading"
        wget http://msvocds.blob.core.windows.net/coco2014/$i.zip
    fi
    unzip $i.zip
    echo $i/*.jpg | mv -t images
    rm -r $i
done

anno="instances_train-val2014.zip"
if ! [ -e $anno ]
then
    echo $anno "not found, downloading"
    wget http://msvocds.blob.core.windows.net/annotations-1-0-3/$anno
fi
unzip $anno

mkdir coco
mv images coco
mv annotations coco
