#!/usr/bin/env bash

cd src

if test "$1" == 'train'  ;then
  python main.py ctdet --exp_id in2car --batch_size 16 --dataset in2_car --gpus 5,6
elif test "$1" == 'test'  ;then
  python test.py --exp_id in2car --not_prefetch_test ctdet --load_model ../exp/ctdet/in2car/model_best.pth --dataset in2_car
fi