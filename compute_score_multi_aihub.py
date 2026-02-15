import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F


def compute_mse(x, y):
    return np.mean((x - y) ** 2)
    # return np.abs(x - y)


def compute_nmse(x, y):
    mse = np.mean((x - y) ** 2)
    norm = np.mean(x ** 2)
    return mse / norm


def compute_cosine_similarity(x, y):
    # Flatten to 2D if needed
    x_flat = x.reshape(1, -1)
    y_flat = y.reshape(1, -1)
    return cosine_similarity(x_flat, y_flat)[0][0]


def compute_kl_divergence(logits_fp, logits_quant, eps=1e-8):

    logits_fp_flat = logits_fp.reshape(-1)
    logits_quant_flat = logits_quant.reshape(-1)

    # softmax
    p = np.exp(logits_fp_flat - np.max(logits_fp_flat))
    p = p / (np.sum(p) + eps)

    q = np.exp(logits_quant_flat - np.max(logits_quant_flat))
    q = q / (np.sum(q) + eps)

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    # KL(p||q) = sum p * log(p/q)
    kl = np.sum(p * np.log(p / q))
    return kl


def compute_psnr(x, y, max_pixel=255.0):
    mse = compute_mse(x, y)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)


NUM_TOKEN = 10
BATCH = 0
MODE = "logits" # measure error for "logits" or "kvcache"

dir_device = "./output/" + f"/submission_{BATCH}"
output_device = dir_device + "/output_past_key_*_out.npy"

list_device = glob.glob(output_device)
list_device.sort()

mse_list = []
nmse_list = []
cos_list = []
kv_list = []


for run in range(NUM_TOKEN):

    output_linux = f"/path/to_inout/outputs_b{BATCH}_t{run}.pt"

    output_data_linux = torch.load(output_linux)

    if MODE == "logits":
        output_l_logits = output_data_linux['logits'].cpu().detach().numpy()
        output_d_logits = np.load(dir_device + f"/output_logits.npy")[run]

        mse_score = compute_mse(output_l_logits, output_d_logits)
        mse_list.append(mse_score)
        nmse_score = compute_nmse(output_l_logits, output_d_logits)
        nmse_list.append(nmse_score)
        cos_score = compute_cosine_similarity(output_l_logits, output_d_logits)
        cos_list.append(cos_score)
        kv_score = compute_kl_divergence(output_l_logits, output_d_logits)
        kv_list.append(kv_score)

    elif MODE == "kvcache":

        for _i in range(len(list_device)):

            output_l_key = output_data_linux['past_key_values'][_i][0].cpu().detach().numpy()
            output_l_value = output_data_linux['past_key_values'][_i][1].cpu().detach().numpy()

            output_d_key = np.transpose(np.load(dir_device + f"/output_past_key_{_i}_out.npy")[run], (1,0,2,3))
            output_d_value = np.transpose(np.load(dir_device + f"/output_past_value_{_i}_out.npy")[run], (1,0,2,3))

            mse_list.append(compute_mse(output_l_key, output_d_key))
            nmse_list.append(compute_nmse(output_l_key, output_d_key))
            cos_list.append(compute_cosine_similarity(output_l_key, output_d_key))

            mse_list.append(compute_mse(output_l_value, output_d_value))
            nmse_list.append(compute_nmse(output_l_value, output_d_value))
            cos_list.append(compute_cosine_similarity(output_l_value, output_d_value))


mse_score = sum(mse_list) / len(mse_list)
nmse_score = sum(nmse_list) / len(nmse_list)
cos_score = sum(cos_list) / len(cos_list)
# psnr_score = 10 * np.log10((1.0 ** 2) / mse_score)

if MODE == "logits":
    kv_score = sum(kv_list) / len(kv_list)
    print(f"There are {len(mse_list)} tokens, with mse_score: {mse_score}, nmse_score: {nmse_score}, cos_score:{cos_score}, kv_score: {kv_score}")
    print(cos_list)
elif MODE == "kvcache":
    print(f"There are {len(mse_list)} tokens, with mse_score: {mse_score}, nmse_score: {nmse_score}, cos_score:{cos_score}")