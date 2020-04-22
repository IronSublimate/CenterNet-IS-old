cd src
# train
if test "$1" == 'train'  ;then
    python main.py ctdet --exp_id coco_resdcn101 --arch resdcn_101 --batch_size 16 --master_batch 5 --lr 3.75e-4 --num_workers 16 --load_model ../exp/ctdet/coco_resdcn101/model_best.pth
elif test "$1" == 'test'  ;then
    python test.py ctdet --exp_id coco_resdcn101 --keep_res --resume --load_model ../exp/ctdet/coco_resdcn101/model_best.pth --arch resdcn_101
fi
# flip test
#python test.py ctdet --exp_id coco_resdcn101 --keep_res --resume --flip_test 
# multi scale test
#python test.py ctdet --exp_id coco_resdcn101 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
