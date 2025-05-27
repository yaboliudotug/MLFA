## TTA
CUDA_VISIBLE_DEVICES=0 python tools/train_net_da.py --config-file configs/city_to_foggy_vgg16.yaml 
CUDA_VISIBLE_DEVICES=0 python tools/train_net_da.py --config-file configs/kitti_to_city_vgg16.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train_net_da.py --config-file configs/sim10k_to_city_vgg16.yaml


## source only
CUDA_VISIBLE_DEVICES=0 python tools/train_net_da.py --config-file configs/city_to_foggy_vgg16.yaml --test_only
CUDA_VISIBLE_DEVICES=0 python tools/train_net_da.py --config-file configs/kitti_to_city_vgg16.yaml --test_only
CUDA_VISIBLE_DEVICES=0 python tools/train_net_da.py --config-file configs/sim10k_to_city_vgg16.yaml --test_only