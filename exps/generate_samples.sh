#!/bin/bash

# -----------------------------------------------------------------------
# This script is for generating a samples output file from an experiment.
# -----------------------------------------------------------------------

cd ..

USAGE="Usage: bash generate_samples.sh <experiment name> <n diffusion steps> <n traj> <output file> [extra_args]"

EXPNAME=$1
if [ -z $EXPNAME ]; then
  echo "Must specify an experiment name!"
  echo $USAGE
  exit 1
fi

STEPS=$2
if [ -z $STEPS ]; then
  echo "Must specify number of diffusion steps to perform"
  echo $USAGE
  exit 1
fi

TMAX=$3
if [ -z $TMAX ]; then
  echo "You must specify how many frames you want to generate (i.e. the length of the trajectory)"
  echo $USAGE
  exit 1
fi

EXPORT_TO=$4
if [ -z $EXPORT_TO ]; then
  echo "Must specify output filename!"
  echo $USAGE
  exit 1
fi

shift 4

#mkdir -p $SAVEDIR/$EXPNAME/samples

mkdir -p $SAVE_DIR/$EXPNAME/samples/$STEPS

echo "-----------------------------"
echo "Processing n steps:  " $STEPS
echo "T max:               " $TMAX
echo "Extra arguments:     " "$@"
echo "Save to:             " ${EXPORT_TO}
echo "-----------------------------"

python generate.py \
 --network=$SAVE_DIR/$EXPNAME/network-snapshot.pkl \
 --outfile=$EXPORT_TO \
 --t_max=$TMAX \
 --steps=$STEPS "$@"
