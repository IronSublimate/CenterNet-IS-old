#!/usr/bin/env bash

NET_ARCH="hourglass"
cd src

if test "$1" == 'train'  ;then
  python main.py ctdet --exp_id voc2007_${NET_ARCH} --batch_size 8 --dataset voc2007 --gpus 6,7 --arch ${NET_ARCH}
elif test "$1" == 'test'  ;then
  python test.py --exp_id voc2007_${NET_ARCH} --not_prefetch_test ctdet --load_model ../exp/ctdet/voc2007_${NET_ARCH}/model_best.pth --dataset voc2007 --arch ${NET_ARCH}
elif test "$1" == 'show' ;then
  python mydemo.py ctdet --exp_id voc2007_${NET_ARCH} --demo ~/code/senior/data/VOC2007/JPEGImages  --load_model ../exp/ctdet/voc2007_${NET_ARCH}/model_best.pth --dataset voc2007 --arch ${NET_ARCH}
fi
