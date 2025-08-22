import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelSummary,
    ModelCheckpoint,
    EarlyStopping,
)
import torch
from dataset import TrainingDataModule
from evaluation import DataModule
from model import SSLClassifier
import yaml
import numpy as np
import pandas as pd


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

from custom_logger import get_logger
# Get a logger for this module
logger = get_logger(__name__)

def compute_det_curve(target_scores, nontarget_scores):
    """ frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, score of target (or positive, bonafide) trials
      nontarget_scores: np.array, score of non-target (or negative, spoofed) trials
      
    output
    ------
      frr:         np.array,  false rejection rates measured at multiple thresholds
      far:         np.array,  false acceptance rates measured at multiple thresholds
      thresholds:  np.array,  thresholds used to compute frr and far

    frr, far, thresholds have same shape = len(target_scores) + len(nontarget_scores) + 1
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    # false rejection rates
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    # false acceptance rates
    # print(float(all_scores[indices[0]]) - 0.001)
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    # Thresholds are the sorted scores
    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ eer, eer_threshold = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, score of target (or positive, bonafide) trials
      nontarget_scores: np.array, score of non-target (or negative, spoofed) trials
      
    output
    ------
      eer:              scalar,  value of EER
      eer_threshold:    scalar,  value of threshold corresponding to EER
    """

    frr, far, thresholds = compute_det_curve(
        np.array(target_scores).astype(np.longdouble),
        np.array(nontarget_scores).astype(np.longdouble),
    )
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_eer_API(score_file, protocol_file):
    """eer = compute_eer_API(score_file, protocol_file)
    
    input
    -----
      score_file:     string, path to the socre file
      protocol_file:  string, path to the protocol file
    
    output
    ------
      eer:  scalar, eer value
      
    The way to load text files using read_csv depends on the text format.
    Please change the read_csv if necessary
    """
    protocol_df = pd.read_csv(
            protocol_file,
            names=["file_name", "label"],
            index_col="file_name",
    )

    # load score
    score_df = pd.read_csv(
        score_file,
        names=["file_name", "cm_score"],
        index_col="file_name",
        skipinitialspace=True,
        sep= " ",
        header=0,
    )

    merged_pd = score_df.join(protocol_df)

    bonafide_scores = merged_pd.query('label == "bonafide"')["cm_score"].to_numpy()
        
    spoof_scores = merged_pd.query('label == "spoof"')["cm_score"].to_numpy()
    eer, th = compute_eer(bonafide_scores, spoof_scores)
    return eer, th

if __name__ == "__main__":
    config_yaml = load_config("config.yaml")
    model = SSLClassifier(config=config_yaml)
    pl.seed_everything(config_yaml['training']['seed'], workers=True)
    model_tag = "{}_{}_{}_{}".format(config_yaml['model']['name'],
                                     config_yaml['training']['num_epochs'],
                                     config_yaml['training']['batch_size'], 
                                     config_yaml['training']['learning_rate'])
    model_save_path = os.path.join("models", model_tag)
    trainer = pl.Trainer(
        max_epochs=config_yaml['training']['num_epochs'],
        callbacks=[
            ModelSummary(max_depth=1),
            ModelCheckpoint(
                dirpath=f"./results/{model_save_path}",
                monitor=config_yaml['training']['model_checkpoint']['ckpt_monitor'],
                mode=config_yaml['training']['model_checkpoint']['ckpt_mode'],
                filename="best_model-{epoch:02d}-{val_loss:.5f}-{train_loss:.5f}-{val_accuracy:.5f}-{train_accuracy:.5f}",
                save_top_k=config_yaml['training']['model_checkpoint']['ckpt_save_top'],
                verbose=True,
            ),
            EarlyStopping(monitor=config_yaml['training']['early_stopping']['es_monitor'],
                          mode=config_yaml['training']['early_stopping']['es_mode'],
                          patience=config_yaml['training']['early_stopping']['patience']),
        ],
        enable_model_summary=config_yaml['compute']['enable_model_summary'],
        devices=config_yaml['compute']['device'],
        accelerator=config_yaml['compute']['accelerator'],
        strategy=config_yaml['compute']['strategy'],

    )
    if config_yaml['model']['checkpoint']:
        checkpoint_path = config_yaml['model']['checkpoint']
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # Load weights from checkpoint
        model.load_state_dict(checkpoint['state_dict'], strict=True)

        # Move model to GPU after loading
        device_id = config_yaml['compute']['device'][0]  # e.g., 1
        model.to(f'cuda:{device_id}')
        logger.info("Model loaded successfully")

    if config_yaml['evaluation']['eval']:
        logger.info("-----------------------------EVALUATION-----------------------------")
        data_module = DataModule(config_yaml['evaluation']['batch_size'],
                                 config_yaml['evaluation']['protocol_path'],
                                 config_yaml['evaluation']['transform'])
        logger.info("Data Module for evaluation loaded successfully")
        trainer.test(model, datamodule=data_module)
        logger.info("Test done!")
        eer, th = compute_eer_API(config_yaml['evaluation']['output_score_file'],
                              config_yaml['evaluation']['protocol_path'],)
        logger.info("EER (%): {:.4f}".format(eer * 100))
    
    else:
        data_module = TrainingDataModule(config=config_yaml)
        logger.info("Model is training...")
        trainer.fit(model, datamodule=data_module, ckpt_path=config_yaml['model']['checkpoint'])
