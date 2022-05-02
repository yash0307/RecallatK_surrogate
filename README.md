# Recall@k Surrogate Loss with Large Batches and Similarity Mixup
[**Recall@k Surrogate Loss with Large Batches and Similarity Mixup**](https://arxiv.org/abs/2108.11179),
[*Yash Patel*](https://yash0307.github.io/),
[*Giorgos Tolias*](https://cmp.felk.cvut.cz/~toliageo/),
[*Jiri Matas*](https://cmp.felk.cvut.cz/~matas/),
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.


## Citation

If you make use of the code in this repository for scientific purposes, we appreciate it if you cite our paper:
```latex
@article{patel2022recall,
  title={Recall@k Surrogate Loss with Large Batches and Similarity Mixup},
  author={Patel, Yash and Tolias, Giorgos and Matas, Jiri},
  journal={CVPR},
  year={2022}
}
```

## Setting up datasets
Recall@k Surrogate demonstrates the performance on five publicly available datasets, namely, iNaturalist, Stanford Online Products, PUK Vehicle ID, Stanford Cars, and Caltech CUB. Download each of these datasets from their respective sources.

### Download links
- **iNaturalist (Inaturalist)**: The 2018 version of this dataset can be obtained from the Kaggle [website](https://www.kaggle.com/c/inaturalist-2018/data). The splits between the training and the test classes are provided by Smooth-AP at [splits](https://drive.google.com/file/d/1sXfkBTFDrRU3__-NUs1qBP3sf_0uMB98/view?usp=sharing).
- **Stanford Online Products (sop)**: This dataset can be downloaded from the official [website](https://cvgl.stanford.edu/projects/lifted_struct/).
- **PUK Vehicle ID (vehicle_id)**: This dataset can be obtained from the official [website](https://pkuml.org/resources/pku-vehicleid.html). Note that an email is required to the authors of this dataset for download permissions.
- **Stanford Cars (cars196)**: This dataset can be downloaded from the official [website](http://ai.stanford.edu/~jkrause/cars/car_dataset.html).
- **Caltech CUB (cub)**: This dataset is available at DeepAI on the following [link](https://deepai.org/dataset/cub-200-2011).

### File structure
Place the dataset folders directly in the RecallatK_surrogate folder.

## Training
Some hyper-paramters are hard-coded in `src/main.py`. For training with <dataset>, use following command:
`python src/main.py --loss recallatk --dataset <dataset> --mixup 0 --samples_per_class 4 --embed_dim 512`

For training with SiMix, use the following command:
`python src/main.py --loss recallatk --dataset <dataset> --mixup 0 --samples_per_class 4 --embed_dim 512`

Keep the following in mind:

