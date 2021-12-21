# Pretraining CPC on Ribeiro dataset
python main_cpc_lightning.py --data /home/dxng/datasets/ECG-Ribeiro/data_processed/fs100 --normalize --epochs 100 --output-path=./runs/cpc/ribeiro --lr 0.0001 --batch-size 32 --input-size 1000 --fc-encoder --negatives-from-same-seq-only
# Pretraining CPC on Zheng dataset
python main_cpc_lightning.py --data /home/dxng/datasets/ECG-Zheng/zheng_fs100 --normalize --epochs 100 --output-path=./runs/cpc/zheng --lr 0.0001 --batch-size 32 --input-size 1000 --fc-encoder --negatives-from-same-seq-only
# Pretraining SimCLR on Zheng dataset
python custom_simclr_bolts.py --batch_size 4096 --epochs 1000 --lr 0.0001 --precision 16 --trafos RandomResizedCrop TimeOut --datasets /home/dxng/datasets/ECG-Zheng/zheng_fs100 --log_dir=experiment_logs/zheng
# Evaluate SimCLR on PTB-XL dataset with fine-tuning and linear evaluation
# 100 epochs
python eval.py --method simclr --model_file "/home/dxng/codes/TS-rep-learning/ecg-selfsupervised/experiment_logs/zheng/Mon_May__3_13:46:40_2021_simclr_851_RRC_TO_/checkpoints/model.ckpt" --batch_size 128 --use_pretrained --f_epochs 50 --dataset /home/dxng/datasets/PTB-XL/ptb_xl_fs100
python eval.py --method simclr --model_file "/home/dxng/codes/TS-rep-learning/ecg-selfsupervised/experiment_logs/zheng/Mon_May__3_13:46:40_2021_simclr_851_RRC_TO_/checkpoints/model.ckpt" --batch_size 128 --use_pretrained --l_epochs 50 --linear_evaluation --dataset /home/dxng/datasets/PTB-XL/ptb_xl_fs100
# 1000 epochs
python eval.py --method simclr --model_file "/home/dxng/codes/TS-rep-learning/ecg-selfsupervised/experiment_logs/zheng/Mon_May__3_15:12:43_2021_simclr_810_RRC_TO_/checkpoints/model.ckpt" --batch_size 128 --use_pretrained --f_epochs 50 --dataset /home/dxng/datasets/PTB-XL/ptb_xl_fs100 --base_model xresnet1d50
python eval.py --method simclr --model_file "/home/dxng/codes/TS-rep-learning/ecg-selfsupervised/experiment_logs/zheng/Tue_May_18_09:44:00_2021_simclr_889_TO_/checkpoints/model.ckpt" --batch_size 128 --use_pretrained --f_epochs 10 --dataset /home/dxng/datasets/PTB-XL/ptb_xl_fs100 --base_model xresnet1d50
python eval.py --method simclr --model_file "/home/dxng/codes/TS-rep-learning/ecg-selfsupervised/experiment_logs/zheng/Mon_May__3_15:12:43_2021_simclr_810_RRC_TO_/checkpoints/model.ckpt" --batch_size 128 --use_pretrained --l_epochs 50 --linear_evaluation --dataset /home/dxng/datasets/PTB-XL/ptb_xl_fs100 --base_model xresnet1d50
# load my pretrained model
python eval.py --method simclr --model_file "/home/dxng/runs/ts-unsup/ecg/ours_archXresnet1d50_projLinear_trafosTO_bs4096_epochs1000_pat40/epoch=656-step=1313.ckpt" --batch_size 128 --use_pretrained --f_epochs 10 --dataset /home/dxng/datasets/PTB-XL/ptb_xl_fs100 --base_model xresnet1d50
# Evaluate CPC on PTB-XL dataset with fine-tuning and linear evaluation
python eval.py --method cpc --model_file "/home/dxng/codes/TS-rep-learning/ecg-selfsupervised/runs/cpc/zheng/version_1/best_model.ckpt" --batch_size 128 --use_pretrained --f_epochs 50 --dataset /home/dxng/datasets/PTB-XL/ptb_xl_fs100 --base_model cpc
python eval.py --method cpc --model_file "/home/dxng/codes/TS-rep-learning/ecg-selfsupervised/runs/cpc/zheng/version_1/best_model.ckpt" --batch_size 128 --use_pretrained --l_epochs 50 --linear_evaluation --dataset /home/dxng/datasets/PTB-XL/ptb_xl_fs100 --base_model cpc
# Pretraining CPC on multiple datasets
python main_cpc_lightning.py --data ./data/cinc --data ./data/zheng --data /home/dxng/datasets/ECG-Ribeiro/data_processed/fs100 --normalize --epochs 10 --output-path=./runs/cpc/all --lr 0.0001 --batch-size 32 --input-size 1000 --fc-encoder --negatives-from-same-seq-only