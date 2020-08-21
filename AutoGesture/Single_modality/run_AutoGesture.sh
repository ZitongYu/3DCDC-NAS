#!/bin/sh
MODALITY=M #RGB:M, Depth:K
GPU_IDS=0,1,2,3
FRAME=32
python -u train_AutoGesture_3DCDC.py -m valid -t $MODALITY -g $GPU_IDS  | tee ./log/model-$MODALITY-$FRAME.log
