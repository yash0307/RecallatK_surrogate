# 4801
Code for CVPR 2022 submission ID 4801.

## Libraries used
The file requirements.txt a list of libraries used in our experimental setup.

## Datasets
The four fine-grained retrieval datasets can be downloaded from the original sources as follows:
1. iNaturalist (2018 version): https://www.kaggle.com/c/inaturalist-2018/data 
	* The split between training and testing sets as used in the setup of Brown et al., ECCV 2020, can be downloaded from: https://drive.google.com/file/d/1sXfkBTFDrRU3__-NUs1qBP3sf_0uMB98/view?usp=sharing
	* Extract and place the dataset and the split files in "../4801/Inaturalist/" directory
2. Stanford Online Products (SOP): https://cvgl.stanford.edu/projects/lifted_struct/
	* Extract and place the dataset in "../4801/sop/" directory
3. PKU Vehicle ID: https://pkuml.org/resources/pku-vehicleid.html 
	* The download requires an email to the authors for permission
	* Extract and place the dataset in "../4801/vehicle_id/" directory
4. Cars196: http://ai.stanford.edu/~jkrause/cars/car_dataset.html
	* Extract and place the dataset in '../4801/cars196/' directory


## Training Hyper-parameters
The hyper-parameters of the training are tuned according to performance on the validation set; the original training set is split into training and validation set (Section 4.2 of the main paper). The chosen hyper-parameter values are hard-coded in “main.py” and will be used automatically with the commands below.

## Experiments
Important notes:
* We used a V100 (32GB) GPU for our experiments. For a GPU with lower memory, set the “--bs_base” parameter to a lower value.
* Make sure to set the "--source_path" parameter to "../4801/" (the datasets shall be located in this directory)

### ResNet-50 Experiments with Recall@k Surrogate loss
1. iNaturalist:
	* `python main.py --dataset  Inaturalist --arch resnet50 --embed_dim 512`
	* `python main.py --dataset  Inaturalist --arch resnet50 --embed_dim 128`
2. Stanford Online Products:
	* `python main.py --dataset  sop --arch resnet50 --embed_dim 512`
	* `python main.py --dataset  sop --arch resnet50 --embed_dim 128`
3. PKU Vehicle ID:
	* `python main.py --dataset  vehicle_id --arch resnet50 --embed_dim 512`
	* `python main.py --dataset  vehicle_id --arch resnet50 --embed_dim 128`
4. Cars196:
	* `python main.py --dataset  cars196 --arch resnet50 --embed_dim 512`
	* `python main.py --dataset  cars196 --arch resnet50 --embed_dim 128`

### ViT-B/32 Experiments with Recall@k Surrogate loss
1. iNaturalist:
	* `python main.py --dataset  Inaturalist --arch ViTB32 --embed_dim 512`
2. Stanford Online Products:
	* `python main.py --dataset  sop --arch ViTB32 --embed_dim 512`
3. PKU Vehicle ID:
	* `python main.py --dataset  vehicle_id --arch ViTB32 --embed_dim 512`
4. Cars196:
	* `python main.py --dataset  cars196 --arch ViTB32 --embed_dim 512`


### ViT-B/16 Experiments with Recall@k Surrogate loss
1. iNaturalist:
	* `python main.py --dataset  Inaturalist --arch ViTB16 --embed_dim 512`
2. Stanford Online Products:
	* `python main.py --dataset  sop --arch ViTB16 --embed_dim 512`
3. PKU Vehicle ID:
	* `python main.py --dataset  vehicle_id --arch ViTB16 --embed_dim 512`
4. Cars196:
	* `python main.py --dataset  cars196 --arch ViTB16 --embed_dim 512`

