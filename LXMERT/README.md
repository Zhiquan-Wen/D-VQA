## Training

* Load Pretrained LXMERT model

  The pretrained model can be found [here](https://github.com/Zhiquan-Wen/D-VQA/releases/download/LXMERT_pretrained_model/Epoch20_LXRT.pth)


* Train our model
```
bash run/run.bash 0 saved_path "--pretrain_epoches 6 --loadLXMERT ./Epoch20_LXRT.pth --img_root data/trainval_features_with_boxes --dataroot data/vqacp2/ --self_loss_q 0.7 --self_loss_weight 3"
``` 

## Evaluation
* A json file of results from the test set can be produced with:
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot data/vqacp2/ --img_root data/trainval_features_with_boxes --checkpoint_path saved_models_cp2/best_model.pth --output saved_models_cp2/result/
```
* Compute detailed accuracy for each answer type:
```
python comput_score.py --input saved_models_cp2/result/XX.json --dataroot data/vqacp2/
```

## Pretrained model
A well-trained model can be found [here](https://github.com/Zhiquan-Wen/D-VQA/releases/download/Models_LXMERT/BEST.pth). The test results file produced by it can be found [here](https://github.com/Zhiquan-Wen/D-VQA/releases/download/Results_LXMERT/test_best_epoch0.json) and its performance is as follows:
```
Overall score: 69.75
Yes/No: 80.43 Num: 58.57 other: 67.23
```