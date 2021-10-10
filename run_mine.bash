CUDA_VISIBLE_DEVICES=2 python main_mine.py \
--dataroot /mnt/cephfs/dataset/wenzhiquan/VQACP2/vqacp2/ \
--output ./saved_models_cp/mine_true_contrastive_loss_tao_0.05/ \
--self_loss_weight 3 \
--ml_loss
