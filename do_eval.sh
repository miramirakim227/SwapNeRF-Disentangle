# CUDA_VISIBLE_DEVICES=1 python train/train.py -n srn_car_exp -c conf/exp/srn.conf -D /root/dataset2/ShapeNet/cars --lr 1e-4 --name check_val_num

# CUDA_VISIBLE_DEVICES=0 python train/train.py -c conf/exp/srn.conf -D /root/dataset2/ShapeNet/cars --lr 5e-5 --name GTCAM_5e-5

# CUDA_VISIBLE_DEVICES=3 python train/train.py -c conf/exp/srn.conf -D /root/dataset2/ShapeNet/cars --lr 5e-5 --name new

# CUDA_VISIBLE_DEVICES=1 python train/train.py -c conf/exp/srn.conf -D /root/dataset2/ShapeNet/cars --lr 1e-4 --recon 50 --name delete --cam 10

# generator = 1e-4

CUDA_VISIBLE_DEVICES=3 python eval/eval_approx.py -c conf/exp/srn.conf -D /root/dataset2/ShapeNet/cars --lr 1e-4 --name swapcam