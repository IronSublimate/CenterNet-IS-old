#!/usr/bin/env bash

cd src

if test "$1" == 'train'  ;then
  python main.py ctdet --exp_id voc2007_resdcn101 --batch_size 16 --dataset voc2007 --gpus 5,6 --arch resdcn_101 --load_model ../models/ctdet_pascal_resdcn101_512.pth
elif test "$1" == 'test'  ;then
  python test.py --exp_id voc2007_resdcn101 --not_prefetch_test ctdet --load_model ../exp/ctdet/voc2007_resdcn101/model_best.pth --dataset voc2007 --arch resdcn_101
elif test "$1" == 'show' ;then
  python mydemo.py ctdet --exp_id voc2007_resdcn101 --demo ~/code/senior/data/VOC2007/JPEGImages  --load_model ../exp/ctdet/voc2007_resdcn101/model_best.pth --dataset voc2007 --arch resdcn_101
fi
