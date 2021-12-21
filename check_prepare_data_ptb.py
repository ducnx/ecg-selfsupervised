from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import channel_stoi_default, prepare_data_ptb_xl
from pathlib import Path
from eval import get_dataset
import numpy as np
import os

if __name__ == '__main__':
    target_fs=100
    data_folder_ptb_xl = Path('/home/dxng/datasets/PTB-XL/data')
    target_folder_ptb_xl = Path(f'/home/dxng/datasets/PTB-XL/ptb_xl_fs{target_fs}_test')
    input_channels = 12
    # df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(
    #     data_folder_ptb_xl, min_cnt=0, target_fs=target_fs, channels=input_channels,
    #     channel_stoi=channel_stoi_default, target_folder=target_folder_ptb_xl
    # )
    # dat = np.load(os.path.join(target_folder_ptb_xl, '00215_lr.npy'))
    # print(dat.shape)
    batch_size = 512
    num_workers = 8
    datapath = '/home/dxng/datasets/PTB-XL/ptb_xl_fs100'
    dataset, train_loader, _ = get_dataset(
        batch_size, num_workers, datapath, folds=8, test=False, normalize=False
    )
