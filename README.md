# [MLFA: Toward Realistic Test Time Adaptive Object Detection by Multi-Level Feature Alignment](https://ieeexplore.ieee.org/abstract/document/10713112)

## 💡 Preparation

#### Datasets
Prepare required benchmark datasets following [DATASET.md](./docs/DATASETS.md). Almost all popular DAOD benchmarks are supported in this project.

#### Installation
```
conda create -n mlfa python=3.7
conda activate mlfa
pip install -r requirements.txt
python setup.py build develop
```

## 🔥 Get Start
```
CUDA_VISIBLE_DEVICES=0 python tools/train_net_da.py --config-file configs/sim10k_to_city_vgg16.yaml
```

## 📝 Citation 

If you think this work is helpful for your project, please give it a star and citation. We sincerely appreciate for your acknowledgments.

```BibTeX  
@article{liu2024mlfa,
  title={MLFA: Towards Realistic Test Time Adaptive Object Detection by Multi-level Feature Alignment},
  author={Liu, Yabo and Wang, Jinghua and Huang, Chao and Wu, Yiling and Xu, Yong and Cao, Xiaochun},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```

## 🤞 Acknowledgements 
We mainly appreciate for these good projects and their authors' hard-working.
- This work is based on [SIGMA](https://github.com/CityU-AIM-Group/SIGMA). 
- The implementation of our anchor-free detector is from [FCOS](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f), which highly relies on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
- The feature alignment is from [TTAC](https://github.com/Gorilla-Lab-SCUT/TTAC). 
