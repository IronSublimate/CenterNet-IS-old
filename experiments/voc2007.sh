#!/usr/bin/env bash

cd src

if test "$1" == 'train'  ;then
  python main.py ctdet --exp_id voc2007 --batch_size 32 --dataset voc2007 --gpus 5,6
elif test "$1" == 'test'  ;then
  python test.py --exp_id voc2007 --not_prefetch_test ctdet --load_model ../exp/ctdet/voc2007/model_best.pth --dataset voc2007
elif test "$1" == 'show' ;then
  python mydemo.py ctdet --exp_id voc2007 --demo ~/code/senior/data/VOC2007/JPEGImages  --load_model ../exp/ctdet/voc2007/model_best.pth --dataset voc2007
fi
