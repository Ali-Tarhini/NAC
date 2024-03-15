#!/bin/bash
T=`date +%m%d%H%M`

ROOT=/mnt/home/pxu22/codes/NAC
export PYTHONPATH=$ROOT:$PYTHONPATH

#PARTITION=$1
NUM_GPU=$1
CFG=./config.yaml
if [ -z $2 ];then
    NAME=default
else
    NAME=$2
fi


#python -W ignore -u -m nac.solver.nac_induc_solver --config="./examples/Cora/lnac/config.yaml" --phase train_search 
python3 -W ignore -u -m nac.solver.nac_induc_solver  --config=$CFG --phase train_search \
  2>&1 | tee log.train.$NAME.$T
