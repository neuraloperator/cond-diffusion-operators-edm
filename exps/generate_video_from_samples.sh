#!/bin/bash

#set -x

USAGE="Usage: bash generate_video_from_samples.sh <path to samples file> <desired frame rate in fps> <outfile>"

SAMPLES_PATH=$1
FPS=$2
OUTFILE=$3

if [ -z $SAMPLES_PATH ]; then
  echo "Error: you must specify path to samples file"
  echo $USAGE
  exit 1
fi

if [ -z $FPS ]; then
  echo "Error: you must specify frame rate"
  echo $USAGE
  exit 1
fi

if [ -z $OUTFILE ]; then
  echo "Error: you must specify output mp4 file"
  echo $USAGE
  exit 1
fi


#if [ ! -f $SAVE_DIR/$SAMPLES_PATH ]; then
#  echo "Path to file does not exist: $SAVEDIR/$SAMPLES_PATH"
#  exit 1
#fi

TMP_DIR=`mktemp -d`
echo "Created tmp directory: " $TMP_DIR

#FIGSIZE="10 6"
FIGSIZE="20 12"

pushd .

cd ..
python -m scripts.generate_frames_from_samples \
  --samples=$SAMPLES_PATH \
  --outdir=$TMP_DIR --figsize $FIGSIZE && \
cd $TMP_DIR && ffmpeg -r $FPS -i %07d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p out.mp4

# Go back to our directory
popd

cp $TMP_DIR/out.mp4 $OUTFILE
