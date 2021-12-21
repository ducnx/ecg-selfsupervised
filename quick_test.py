"""
python -m src.train.quick_test
"""
import torch
from clinical_ts.simclr_dataset_wrapper import SimCLRDataSetWrapper
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from clinical_ts.timeseries_utils import aggregate_predictions

if __name__ == '__main__':
    ckpt = torch.load(
        '/home/dxng/codes/TS-rep-learning/ecg-selfsupervised/experiment_logs/zheng/Tue_May_18_09:44:00_2021_simclr_889_TO_/checkpoints/model.ckpt'
    )
    print(type(ckpt))
    print(ckpt.keys())
    torch.save(
        {
            'state_dict': ckpt['state_dict'],
        },
        '/home/dxng/codes/TS-rep-learning/ecg-selfsupervised/experiment_logs/zheng/Tue_May_18_09:44:00_2021_simclr_889_TO_/checkpoints/model_state_dict.ckpt'
    )
