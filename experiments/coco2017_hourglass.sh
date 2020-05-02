#!/usr/bin/env bash
cd src

NET_ARCH="hourglass"
DATA_SET="coco"
CUDA_VISIBLE_DEVICES=3

# batch_size=4 for 11G, 2 for 8G
if test "$1" == 'train'  ;then
  python main.py ctdet --exp_id ${DATA_SET}_${NET_ARCH} --print_iter 10  --batch_size 4 --dataset ${DATA_SET} --arch ${NET_ARCH}
elif test "$1" == 'test'  ;then
  python test.py --exp_id ${DATA_SET}_${NET_ARCH} --print_iter 10  --not_prefetch_test ctdet --load_model ../exp/ctdet/${DATA_SET}_${NET_ARCH}/model_best.pth --dataset ${DATA_SET} --arch ${NET_ARCH}
elif test "$1" == 'show' ;then
  python mydemo.py ctdet --exp_id ${DATA_SET}_${NET_ARCH} --print_iter 10  --demo ~/code/senior/data/VOC2007/JPEGImages  --load_model ../exp/ctdet/${DATA_SET}_${NET_ARCH}/model_best.pth --dataset ${DATA_SET} --arch ${NET_ARCH}
fi

