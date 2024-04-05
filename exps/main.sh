#!/bin/bash

METHOD="train.py"
EXP_GROUP=$1
CFG_FILE=$2
N_GPU=$3

if [ -z $EXP_GROUP ]; then
  echo "Must specify an experiment group name!"
  exit 1
fi

if [ -z $CFG_FILE ]; then
  echo "Must specify a json file"
  exit 1
fi

if [ -z $SAVE_DIR ]; then
  echo "SAVE_DIR not found, source env.sh?"
  exit 1
fi

if [ -z $N_GPU ]; then
  echo "Must set number of gpus to train on. If > 1 we use torchrun"
  exit 1
fi

if [ -z "$SLURM_JOB_ID" ]; then
  echo "$SLURM_JOB_ID not set for some reason, are you in a login node?"
  echo "Set variable to '999999' for now"
  SLURM_JOB_ID=999999
fi

EXP_NAME="${EXP_GROUP}/${SLURM_JOB_ID}"
echo "Experiment name: " $EXP_NAME

cd ..


# TODO: add argument for distributed

CFG_ABS_PATH=`pwd`/exps/${CFG_FILE}
#python train.py --cfg=$CFG_ABS_PATH --savedir=${SAVEDIR}/${EXP_NAME}

# If code does not exist for this experiment, copy
# it over. Then cd into that directory and run the code.
# But only if we're not in run_local mode.
if [ -z $RUN_LOCAL ]; then
  if [ ! -d ${SAVEDIR}/${EXP_NAME}/code ]; then
    mkdir -p ${SAVE_DIR}/${EXP_NAME}
    echo "Copying code..."
    rsync -r -v --exclude='exps' --exclude='.git' --exclude='__pycache__' --exclude '*.pyc' . ${SAVE_DIR}/${EXP_NAME}/code
    if [ ! $? -eq 0 ]; then
      echo "rsync returned error, terminating..."
      exit 1
    fi
  fi
fi

CFG_ABS_PATH=`pwd`/exps/${CFG_FILE}
echo "Absolute path of cfg:  " $CFG_ABS_PATH

if [[ ! "$N_GPU" -eq 1 ]]; then
  CMD_TO_RUN="torchrun --standalone --nproc_per_node=${N_GPU} train.py"
else
  CMD_TO_RUN="python train.py"
fi
echo "Command to run:  " $CMD_TO_RUN

if [ -z $RUN_LOCAL ]; then
  cd ${SAVE_DIR}/${EXP_NAME}/code
  ${CMD_TO_RUN} --cfg=$CFG_ABS_PATH --savedir=${SAVE_DIR}/${EXP_NAME}
else
  echo "RUN_LOCAL mode set, run code from this directory..."
  # --override_cfg = use the local cfg file, not the one in the experiment directory
  # also do not use torchrun, just debug with 1 gpu
  ${CMD_TO_RUN} --cfg=$CFG_ABS_PATH --savedir=${SAVE_DIR}/${EXP_NAME} --override_cfg
fi
echo "Current working directory: " `pwd`

# Use this as a reference for trapping SIGTERM signal:
# https://wikis.ch.cam.ac.uk/ro-walesdocs/wiki/index.php/Getting_started_with_SLURM 

#bash launch.sh $EXP_NAME
