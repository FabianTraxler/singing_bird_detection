"""This module provides all training utilities for the model streams in table_header_standardization"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm
import os
#import wandb
from datetime import datetime
import torch.nn.functional as F

from dataset import XenoCantoData
from model import  Net
from config import cfg


torch.cuda.empty_cache()

use_wandb = False

BATCH_SIZE = 5
EPOCHS = 100
LEARNING_RATE = 1e-5
PATIENCE = 20
CLASS_WEIGHT = [1,2,2,2,2,2]
WORKERS = 0
THRESHOLD = 0.5

DATASET_PATH = "./dataset/dataset.csv"

def train_loop(model: nn.Module, df_train: pd.DataFrame, df_val):
    """ Train the roberta Model with the specifies Hyperparameters in the config function

    Always save the best version of the model on the AWS S3 for later inference

    Args:
        model: specified torch model
        df_train: training dataframe
        df_val: validation dataframe

    Returns:
        None
    """

    train_dataset = XenoCantoData(cfg, dataset=df_train)
    val_dataset = XenoCantoData(cfg, dataset=df_val)

    train_dataloader = DataLoader(train_dataset, num_workers=WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=WORKERS, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000
    patience = PATIENCE

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()
        total_train_cat_data = 0

        for train_data, train_label in tqdm(train_dataloader):
                train_data = train_data.to(device)
                train_label = train_label.to(device)

                output = model(train_data)

                loss = F.binary_cross_entropy_with_logits(output["logits_raw"], train_label)

                predictions = (output["logits"] > THRESHOLD).long()
                acc = (predictions == train_label).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()

                model.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0
        total_val_cat_data = 0


        with torch.no_grad():
            for val_data, val_label in val_dataloader:
                val_data = val_data.to(device)
                val_label = val_label.to(device)

                output = model(val_data)

                loss = F.binary_cross_entropy_with_logits(output["logits_raw"], val_label)

                # Accuracy

                predictions = (output["logits"] > THRESHOLD).long()
                acc = (predictions == val_label).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()
                total_val_cat_data += 1

            val_accuracy = total_acc_val / len(df_val)
            val_loss = total_loss_val / len(df_val)


            if val_accuracy > best_acc and val_loss < best_loss:
                best_acc = val_accuracy
                best_loss = val_loss
                patience = PATIENCE
            else:
                patience -= 1

            print(
                f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / total_train_cat_data: .3f} | Val_Loss: {total_loss_val / total_val_cat_data: .3f} | Accuracy: {total_acc_val / total_val_cat_data: .3f}')

            if patience <= 0:
                break


def model_train():
    """
    Instantiate model training

    Returns:
        None
    """

    #df = fetch_dataframe()
    df = pd.read_csv(DATASET_PATH)

    np.random.seed(112)
    df_train, df_val = np.split(df.sample(frac=1, random_state=42),
                                         [int(.7 * len(df))])

    print(len(df_train), len(df_val))

    model = Net(cfg)

    train_loop(model, df_train, df_val)

    return model


if __name__ == "__main__":
    trained_model = model_train()
