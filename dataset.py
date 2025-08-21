import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
import librosa
import argparse
import csv
from tqdm import tqdm
from custom_logger import get_logger
import yaml
import numpy as np
from torchvision import transforms

# Load YAML config file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')

parser.add_argument('--algo', type=int, default=4, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

# LnL_convolutive_noise parameters 
parser.add_argument('--nBands', type=int, default=5, 
                help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
parser.add_argument('--minF', type=int, default=20, 
                help='minimum centre frequency [Hz] of notch filter.[default=20] ')
parser.add_argument('--maxF', type=int, default=8000, 
                help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
parser.add_argument('--minBW', type=int, default=100, 
                help='minimum width [Hz] of filter.[default=100] ')
parser.add_argument('--maxBW', type=int, default=1000, 
                help='maximum width [Hz] of filter.[default=1000] ')
parser.add_argument('--minCoeff', type=int, default=10, 
                help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
parser.add_argument('--maxCoeff', type=int, default=100, 
                help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
parser.add_argument('--minG', type=int, default=0, 
                help='minimum gain factor of linear component.[default=0]')
parser.add_argument('--maxG', type=int, default=0, 
                help='maximum gain factor of linear component.[default=0]')
parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                help=' minimum gain difference between linear and non-linear components.[default=5]')
parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                help=' maximum gain difference between linear and non-linear components.[default=20]')
parser.add_argument('--N_f', type=int, default=5, 
                help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

# ISD_additive_noise parameters
parser.add_argument('--P', type=int, default=10, 
                help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
parser.add_argument('--g_sd', type=int, default=2, 
                help='gain parameters > 0. [default=2]')

# SSI_additive_noise parameters
parser.add_argument('--SNRmin', type=int, default=10, 
                help='Minimum SNR value for coloured additive noise.[defaul=10]')
parser.add_argument('--SNRmax', type=int, default=40, 
                help='Maximum SNR value for coloured additive noise.[defaul=40]')

rawboost_args = parser.parse_args()

# Get a logger for this module
logger = get_logger(__name__)

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class DatasetTrain(Dataset):
    def __init__(self, protocol, config, args=None):

        self.list_IDs, self.labels = self._get_list_IDs(protocol)
        self.config = config
        self.algo = self.config['rawboost']['algo']
        self.args = args
        self.transform = self.config['dataset']['transform']
        self.data_augmentation = None
        if self.transform:
            logger.info("-----------------------------PADDING-----------------------------")
            self.transform = transforms.Compose([
                lambda x: pad(x, self.config['dataset']['max_len']),
                lambda x: torch.Tensor(x),
            ])
        logger.info(f'Number of audios: {len(self.list_IDs)}')
        if self.algo and self.args:
            logger.info("-----------------------------RAWBOOST-----------------------------")
            logger.info(f'Rawboost algo: {self.algo}')
            self.data_augmentation = transforms.Compose([
                lambda x: process_Rawboost_feature(x, self.config['dataset']['sampling_rate'], self.args, self.algo),
                lambda x: torch.Tensor(x),
            ])
        else:
            self.algo = None
            self.args = None

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        
        X, fs = librosa.load(utt_id, sr=self.config['dataset']['sampling_rate'])
        if self.algo and self.args:
            X = self.data_augmentation(X)
        if self.transform:
            X = self.transform(X)
        else:
            X = torch.Tensor(X)
        target = self.labels[utt_id]

        return X, target
    
    def _get_list_IDs(self, input_file):
        delimiter = ','
        list_IDs = []
        d_meta = {}
        with open(input_file, 'r') as infile:
            reader = csv.reader(infile, delimiter=delimiter)
            first_row = next(reader)
            if 'file_name' in first_row or 'label' in first_row:
                print("Skipping header row in the input file.")
            else:
                reader = [first_row] + list(reader)
            total_lines = sum(1 for _ in open(input_file)) - 1
            infile.seek(0)
            for row in tqdm(reader, total=total_lines, desc=f"Processing Ids"):
                try:
                    file_name, label = row
                    d_meta[file_name] = 1 if label == "bonafide" else 0
                    list_IDs.append(file_name)
                except ValueError as e:
                    print(f"Skipping malformed row: {row}. Error: {e}")

        return list_IDs, d_meta
    
class TrainingDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['training']['batch_size']
        self.transform = config['dataset']['transform']
        logger.info(f'Batch size: {self.batch_size}')
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        config = load_config("config.yaml")
        if stage == "fit":
            self.train = DatasetTrain(
                config['dataset']['train']['protocol_path'],
                config,
                rawboost_args
            )
            self.val = DatasetTrain(
                config['dataset']['val']['protocol_path'],
                config,
                None
            )

    def train_dataloader(self):
        collate_fn = None if self.transform else self._collate_fn

        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self):
        collate_fn = None if self.transform else self._collate_fn

        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    
    def _collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.nn.utils.rnn.pad_sequence(
            [tensor.squeeze() for tensor in x], batch_first=True, padding_value=0.0
        )
        y = [torch.tensor([label]) for label in y]
        y = torch.stack(y).squeeze()
        return x, y

    

        #--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:

        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature