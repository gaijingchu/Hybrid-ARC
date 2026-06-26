#!/bin/bash
# usage: run.sh <gpu> <logfile> <python-args...>
GPU=$1; LOG=$2; shift 2
cd /project/flame/jgai/hybrid-ARC-clean
export CUDA_VISIBLE_DEVICES=$GPU
/home/jgai/miniconda3/envs/prm/bin/python "$@" > "$LOG" 2>&1
echo "EXIT=$? -> $LOG"
