# D-VQA
We provide the PyTorch implementation for [Debiased Visual Question Answering from Feature and Sample Perspectives](https://openreview.net/pdf?id=Z4ry59PVMq8) (NeurIPS 2021).

<p align="center">
<img src="framework.png" alt="D-VQA" width="90%" align=center />
</p>

## Dependencies
* Python 3.6
* PyTorch 1.1.0
* dependencies in requirements.txt
* We train and evaluate all of the models based on one TITAN Xp GPU

## Getting Started

### Installation
1. Clone this repository:

        git clone https://github.com/Zhiquan-Wen/D-VQA.git
        cd D-VQA

2. Install PyTorch and other dependencies:

        pip install -r requirements.txt


### Download and preprocess the data

```
cd data 
bash download.sh
python preprocess_features.py --input_tsv_folder xxx.tsv --output_h5 xxx.h5
python feature_preprocess.py --input_h5 xxx.h5 --output_path trainval 
python create_dictionary.py --dataroot vqacp2/
python preprocess_text.py --dataroot vqacp2/ --version v2
cd ..
```

### Training
* Train our model
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataroot data/vqacp2/ --img_root data/coco/trainval_features --output saved_models_cp2/ --self_loss_weight 3 --self_loss_q 0.7
``` 

* Train the model with 80% of the original training set
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataroot data/vqacp2/ --img_root data/coco/trainval_features --output saved_models_cp2/ --self_loss_weight 3 --self_loss_q 0.7 --ratio 0.8 
```

### Evaluation
* A json file of results from the test set can be produced with:
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot data/vqacp2/ --img_root data/coco/trainval_features --checkpoint_path saved_models_cp2/best_model.pth --output saved_models_cp2/result/
```
* Compute detailed accuracy for each answer type:
```
python comput_score.py --input saved_models_cp2/result/XX.json --dataroot data/vqacp2/
```

### Pretrained model
A well-trained model can be found [here](https://github.com/Zhiquan-Wen/D-VQA/releases/download/Models/61.91.pth) with [raw training log](https://raw.githubusercontent.com/Zhiquan-Wen/D-VQA/master/logs/61.91.log). The test results file produced by it can be found [here](https://github.com/Zhiquan-Wen/D-VQA/releases/download/Results/61.91_results.json) and its performance is as follows:
```
Overall score: 61.91
Yes/No: 88.93 Num: 52.32 other: 50.39
```

## Quick Reproduce

1. **Preparing enviroments**: we prepare a docker image (built from [Dockerfile](https://github.com/Zhiquan-Wen/D-VQA/blob/master/docker/Dockerfile)) which has included above dependencies, you can pull this image from dockerhub or aliyun registry:

```
docker pull zhiquanwen/debias_vqa:v1
```


```
docker pull registry.cn-shenzhen.aliyuncs.com/wenzhiquan/debias_vqa:v1
docker tag registry.cn-shenzhen.aliyuncs.com/wenzhiquan/debias_vqa:v1 zhiquanwen/debias_vqa:v1
```

2. **Start docker container**: start the container by mapping the dataset in it:

```
docker run --gpus all -it --ipc=host --network=host --shm-size 32g -v /host/path/to/data:/xxx:ro zhiquanwen/debias_vqa:v1
```

3. **Running**: refer to `Download and preprocess the data`, `Training` and `Evaluation` steps in `Getting Started`.

**Results**: we obtain 61.73% in VQA-CP2 (which is almost the same as 61.91% in Table 1 of the paper) using the above docker image and training steps. We also provide the [raw training log](https://raw.githubusercontent.com/Zhiquan-Wen/D-VQA/master/logs/reproduce_61.73.log).

## Reference
If you found this code is useful, please cite the following paper:
```
@inproceedings{D-VQA,
  title     = {Debiased Visual Question Answering from Feature and Sample Perspectives},
  author    = {Zhiquan Wen, 
               Guanghui Xu, 
               Mingkui Tan, 
               Qingyao Wu, 
               Qi Wu},
  booktitle = {NeurIPS},
  year = {2021}
}
```

## Acknowledgements
This repository contains code modified from [SSL-VQA](https://github.com/CrossmodalGroup/SSL-VQA), thank you very much!

Besides, we thank Yaofo Chen for providing [MIO](https://github.com/chenyaofo/mio) library to accelerate the data loading.