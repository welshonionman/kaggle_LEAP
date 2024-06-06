import gc
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from src.constants import (
    DEVICE,
    FEAT_COLS,
    FEAT_LEN,
    SAMPLE_PATH,
    TARGET_COLS,
    TARGET_LEN,
    TEST_PATH,
    TRAIN_PATH,
)
from src.dataset import get_dataloader
from src.model import FFNN
from src.preprocess import standardize
from src.utils.inference import heuristic_postprocess, postprocess_predictions, predict
from src.utils.utils import (
    calc_r2,
    epoch_end,
    load_weights,
    pbar_train_desc,
    pbar_valid_desc,
    set_seed,
)

BATCH_SIZE = 2**14
MIN_STD = 1e-8
SCHEDULER_PATIENCE = 4
SCHEDULER_FACTOR = 10 ** (-0.5)
EPOCHS = 1000
PATIENCE = 10
EXP_NAME = Path(__file__).stem
MODEL_PATH = f"/kaggle/weights/{EXP_NAME}.pth"
SUB_PATH = "./submission.csv"
WANDB_MODE = "online"

if __name__ == "__main__":
    weights = load_weights()
    set_seed()

    df_train = pl.read_csv(TRAIN_PATH, n_rows=3_000_000)
    # df_train = pl.read_csv(TRAIN_PATH, n_rows=100_000)

    for target in weights:
        df_train = df_train.with_columns(pl.col(target).mul(weights[target]))

    x_train = df_train.select(FEAT_COLS).to_numpy()
    y_train = df_train.select(TARGET_COLS).to_numpy()

    del df_train
    gc.collect()

    x_train, y_train, mean_x, std_x, mean_y, std_y = standardize(x_train, y_train, MIN_STD)
    train_loader, valid_loader = get_dataloader(x_train, y_train, BATCH_SIZE)

    model = FFNN(
        FEAT_LEN,
        [
            3 * (FEAT_LEN + TARGET_LEN),
            2 * (FEAT_LEN + TARGET_LEN),
            2 * (FEAT_LEN + TARGET_LEN),
            2 * (FEAT_LEN + TARGET_LEN),
            3 * (FEAT_LEN + TARGET_LEN),
        ],
        TARGET_LEN,
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    best_val_loss = float("inf")
    patience_count = 0

    wandb.init(project="LEAP", name="baseline", dir="/workspace/", mode=WANDB_MODE)

    for epoch in range(EPOCHS):
        train_loss = 0
        valid_loss = 0
        y_true = torch.tensor([], device=DEVICE)
        all_outputs = torch.tensor([], device=DEVICE)

        model.train()
        pbar_train = enumerate(train_loader)
        pbar_train = tqdm(pbar_train, total=len(train_loader), bar_format="{l_bar}{bar:10}{r_bar}{bar:-0b}")

        for batch_idx, (inputs, labels) in pbar_train:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            average_loss = train_loss / (batch_idx + 1)

            pbar_train_desc(pbar_train, scheduler, epoch, EPOCHS, average_loss)

        model.eval()
        pbar_val = enumerate(valid_loader)
        pbar_val = tqdm(pbar_val, total=len(valid_loader), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        with torch.no_grad():
            for batch_idx, (inputs, labels) in pbar_val:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()
                y_true = torch.cat((y_true, labels), 0)
                all_outputs = torch.cat((all_outputs, outputs), 0)

                average_loss = valid_loss / (batch_idx + 1)
                pbar_valid_desc(pbar_val, average_loss)

        r2 = calc_r2(all_outputs, y_true)

        avg_val_loss = valid_loss / len(valid_loader)
        scheduler.step(avg_val_loss)

        wandb.log({"epoch": epoch, "loss": average_loss, "r2": r2})
        patience_count, best_val_loss = epoch_end(avg_val_loss, best_val_loss, model, r2, patience_count, MODEL_PATH)

        if patience_count >= PATIENCE:
            print("EARTLY STOPPING")
            break

    del x_train, y_train
    gc.collect()

    # Inference
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    df_test = pl.read_csv(TEST_PATH)
    x_test = df_test.select(FEAT_COLS).to_numpy()
    x_test = (x_test - mean_x.reshape(1, -1)) / np.array(std_x).reshape(1, -1)

    predt = predict(model, x_test, BATCH_SIZE, TARGET_LEN, DEVICE)
    predt = postprocess_predictions(predt, np.array(mean_y), np.array(std_y), MIN_STD)

    ss = pd.read_csv(SAMPLE_PATH)
    ss.iloc[:, 1:] = predt

    ss = heuristic_postprocess(df_test, ss)

    test_polars = pl.from_pandas(ss[["sample_id"] + TARGET_COLS])
    test_polars.write_csv(SUB_PATH)
