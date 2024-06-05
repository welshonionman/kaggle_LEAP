import numpy as np
import torch

def predict(model, x_test, batch_size, output_size, device):
    num_samples = x_test.shape[0]
    predt = np.zeros((num_samples, output_size))

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_inputs = torch.from_numpy(x_test[start_idx:end_idx, :]).float().to(device)

        with torch.no_grad():
            batch_outputs = model(batch_inputs)
            predt[start_idx:end_idx, :] = batch_outputs.cpu().numpy()

    return predt


def postprocess_predictions(predt, mean_y, std_y, min_std):
    for i in range(std_y.shape[0]):
        if std_y[i] < min_std * 1.1:
            predt[:, i] = 0

    predt = predt * std_y.reshape(1, -1) + mean_y.reshape(1, -1)

    return predt


def heuristic_postprocess(df_test, ss):
    ss2 = ss.copy()
    use_cols = []
    for i in range(27):
        use_cols.append(f"ptend_q0002_{i}")

    df_test = df_test.to_pandas()
    for col in use_cols:
        ss[col] = -df_test[col.replace("ptend", "state")] * ss2[col] / 1200.0
        return ss