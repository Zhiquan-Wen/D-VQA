# D-VQA
We provide the PyTorch implementation for [Debiased Visual Question Answering from Feature and Sample Perspectives](). 

<p align="center">
<img src="framework.png" alt="D-VQA" width="90%" align=center />
</p>


## Download and preprocess the data

```
cd data 
bash download.sh
python preprocess_image.py --input_tsv_folder xxx.tsv --output_h5 xxx.h5
python preprocess_features.py --input_h5 xxx.h5 --output_path trainval 
python create_dictionary.py --dataroot vqacp2/
python preprocess_text.py --dataroot vqacp2/ --version v2
cd ..
```

## Training
* Train our model
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataroot data/vqacp2/ --img_root data/coco/trainval_features --output saved_models_cp2/ --self_loss_weight 3 --self_loss_q 0.7
``` 

* Train the model with 80% of the original training set
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataroot data/vqacp2/ --img_root data/coco/trainval_features --output saved_models_cp2/ --self_loss_weight 3 --self_loss_q 0.7 --ratio 0.8 
```

## Evaluation
* A json file of results from the test set can be produced with:
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot data/vqacp2/ --img_root data/coco/trainval_features --checkpoint_path saved_models_cp2/best_model.pth --output saved_models_cp2/result/
```
* Compute detailed accuracy for each answer type:
```
python comput_score.py --input saved_models_cp2/result/XX.json --dataroot data/vqacp2/
```

## Pretrained model
A well-trained model can be found [here](https://github.com/Zhiquan-Wen/D-VQA/releases/download/Models/61.91.pth). The test results file produced by it can be found [here](https://github.com/Zhiquan-Wen/D-VQA/releases/download/Results/61.91_results.json) and its performance is as follows:
```
Overall score: 61.91
Yes/No: 88.93 Num: 52.32 other: 50.39
```


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
  booktitle = {NeurIPS}
}
```

## Acknowledgements
This repository contains code modified from [SSL-VQA](https://github.com/CrossmodalGroup/SSL-VQA), thank you very much!

Besides, we thank Yaofo Chen for providing [MIO](https://github.com/chenyaofo/mio) library to accelerate the data loading.