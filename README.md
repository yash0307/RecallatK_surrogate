# Recall@k Surrogate Loss with Large Batches and Similarity Mixup
[**Recall@k Surrogate Loss with Large Batches and Similarity Mixup**](https://arxiv.org/abs/2108.11179),
[*Yash Patel*](https://yash0307.github.io/),
[*Giorgos Tolias*](https://cmp.felk.cvut.cz/~toliageo/),
[*Jiri Matas*](https://cmp.felk.cvut.cz/~matas/),
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.


## Citation

If you make use of the code in this repository for scientific purposes, we appreciate it if you cite our paper:
```latex
@inproceedings{patel2022recall,
  title={Recall@k surrogate loss with large batches and similarity mixup},
  author={Patel, Yash and Tolias, Giorgos and Matas, Ji{\v{r}}{\'\i}},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7502--7511},
  year={2022}
}
```
## Dependencies
A list of all dependencies that were used in our setup is listed in `requirements.txt`. Note that not all of them are necessary; some key dependencies are as follows:
- Python 3.7.4
- PyTorch 1.8.0
- Torchvision 0.9.1
- Faiss 1.6.4

## Setting up datasets
Recall@k Surrogate demonstrates the performance on five publicly available datasets: iNaturalist, Stanford Online Products, PUK Vehicle ID, Stanford Cars, and Caltech CUB. Download each of these datasets from their respective sources.

### Download links
- **iNaturalist (Inaturalist)**: The 2018 version of this dataset can be obtained from the Kaggle [website](https://www.kaggle.com/c/inaturalist-2018/data). Smooth-AP provides the splits between the training and the test classes at [splits](https://drive.google.com/file/d/1sXfkBTFDrRU3__-NUs1qBP3sf_0uMB98/view?usp=sharing).
- **Stanford Online Products (sop)**: This dataset can be downloaded from the official [website](https://cvgl.stanford.edu/projects/lifted_struct/).
- **PUK Vehicle ID (vehicle_id)**: This dataset can be obtained from the official [website](https://pkuml.org/resources/pku-vehicleid.html). Note that an email is required to the authors of this dataset for download permissions.
- **Stanford Cars (cars196)**: This dataset can be downloaded from the official [website](http://ai.stanford.edu/~jkrause/cars/car_dataset.html).
- **Caltech CUB (cub)**: This dataset is available at DeepAI on the following [link](https://deepai.org/dataset/cub-200-2011).

### File structure
Place the dataset folders directly in the RecallatK_surrogate folder. [An example](https://github.com/yash0307/RecallatK_surrogate/blob/main/file_structure.txt) of the file structure with datasets.

## Training
Some hyper-paramters are hard-coded in `src/main.py`. For training with `<dataset>`, use following command:

`python src/main.py --source_path <path_to_RecallatK_surrogate> --loss recallatk --dataset <dataset> --mixup 0 --samples_per_class 4 --embed_dim 512`

For training with SiMix, use the following command:

`python src/main.py --source_path <path_to_RecallatK_surrogate> --loss recallatk --dataset <dataset> --mixup 1 --samples_per_class 4 --embed_dim 512`

Keep the following in mind:
- Batch size (`--bs`) is by default set to `max(4000, #classes*samples_per_class)`. This works on a 32 GB Nvidia V100 GPU; consider lowering the batch size if you run into GPU out-of-memory error.
- Base batch size (`--bs_base`) is by default set to `200`. This works on a 32 GB Nvidia V100 GPU; consider lowering the base batch size if you run into GPU out-of-memory error.
- The use of SiMix (`--mixup 1`) is optional. In our experiments, it has shown to be very useful for small-scale datasets.

