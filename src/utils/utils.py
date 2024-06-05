import torch
import time
import os
import numpy as np
import random
import pandas as pd
from torchmetrics.regression import R2Score
from src.constants import TARGET_COLS, FEAT_COLS, DEVICE, SAMPLE_PATH


def set_seed(seed=42, cudnn_deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


def timer(func):
    def wrapper(*args, **kwargs):
        config = kwargs["config"]
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"\nElapsed time: {elapsed_time:.2f} seconds",
            file=open(config.log_path, "a"),
        )
        return result

    return wrapper


def load_weights():
    weights = pd.read_csv(SAMPLE_PATH, nrows=1)
    del weights["sample_id"]
    weights = weights.T
    weights = weights.to_dict()[0]
    return weights


def epoch_end(avg_val_loss, best_val_loss, model, r2, patience_count, save_path):
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_count = 0
        torch.save(model.state_dict(), save_path)
        print(f"Val Loss: {avg_val_loss:.4f}  R2 score: {r2:.4f}\tSAVED MODEL\n")

    else:
        patience_count += 1
        print(f"Val Loss: {avg_val_loss:.4f}  R2 score: {r2:.4f}\n")
    return patience_count, best_val_loss


def calc_r2(all_outputs, y_true):
    r2score = R2Score(num_outputs=len(TARGET_COLS)).to(DEVICE)
    r2 = 0
    r2_broken = []
    r2_broken_names = []
    for col_i in range(len(TARGET_COLS)):
        r2_i = r2score(all_outputs[:, col_i], y_true[:, col_i])
        if r2_i > 1e-6:
            r2 += r2_i
        else:
            r2_broken.append(col_i)
            r2_broken_names.append(FEAT_COLS[col_i])
    r2 /= len(TARGET_COLS)
    # print(f"{len(r2_broken)} targets were excluded during evaluation of R2 score.")
    # print(r2_broken)
    # print(r2_broken_names, flush=True)
    return r2


def pbar_train_desc(pbar_train, scheduler, epoch, total_epoch, average_loss):
    learning_rate = f"LR : {scheduler.get_last_lr()[0]:.2E}"
    gpu_memory = f"Mem : {torch.cuda.memory_reserved() / 1E9:.3g}GB"
    epoch_info = f"Epoch {epoch}/{total_epoch}"
    loss = f"Loss: {average_loss:.4f}"

    description = f"{epoch_info:12} {gpu_memory:15} {learning_rate:15} {loss:15}"
    pbar_train.set_description(description)


def pbar_valid_desc(pbar_val, average_loss):
    average_val_loss = f"Val Loss: {average_loss:.4f}"

    description = f"{average_val_loss:18}"
    pbar_val.set_description(description)
