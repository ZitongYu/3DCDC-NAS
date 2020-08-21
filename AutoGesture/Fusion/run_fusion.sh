#!/bin/sh
LAYER=AUG
GPU_IDS='3'
FRAME=\(48+32+16\)_NAS1+NAS2_all
PHASE=valid
python -u AutoGesture_Fusion.py -m $PHASE -l $LAYER -g $GPU_IDS  | tee ./log/$LAYER-fusion-$FRAME-$PHASE.log
