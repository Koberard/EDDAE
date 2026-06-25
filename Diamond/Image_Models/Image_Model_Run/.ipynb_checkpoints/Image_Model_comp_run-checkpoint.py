#!/usr/bin/env python3
import os
import sys
import argparse
import h5py
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from bm3d import bm3d
import h5py

# ==========================================
# 1. Path Setup & External Module Imports
# ==========================================
MODULE_DIR_BM3D = "/global/u2/k/kberard/SCGSR/Research/Diamond/stock_models/bm3d-4.0.3/bm3d-4.0.3/examples"
MODULE_DIR_SCU = "/global/u2/k/kberard/SCGSR/Research/Diamond/stock_models/SCUNet"

for path in [MODULE_DIR_BM3D, MODULE_DIR_SCU]:
    if path not in sys.path:
        sys.path.insert(0, path)

from models.network_scunet import SCUNet

# ==========================================
# 2. Hardcoded Model Paths (Adjust as needed)
# ==========================================
MODEL_SAMP = "240700000"
MODEL_PATHS = {
    'scunet_pre': '/global/u2/k/kberard/SCGSR/Research/Diamond/stock_models/SCUNet/model_zoo/scunet_color_25.pth',
    'scunet_trained': '/pscratch/sd/k/kberard/SCGSR/3D_VMC/Model_Train_dat/Scunet_trained_Models/' + MODEL_SAMP + '_scunet_trained',
    'scunet_ft': '/pscratch/sd/k/kberard/SCGSR/3D_VMC/Model_Train_dat/Scunet_FT_Models/' + MODEL_SAMP + '_scunet_FT',
    'nature': '/pscratch/sd/k/kberard/SCGSR/3D_VMC/Model_Train_dat/Nature_Models/' + MODEL_SAMP + '_Nature.keras',
    'CAE': '/pscratch/sd/k/kberard/SCGSR/3D_VMC/Model_Train_dat/CAE_img_Models/' + MODEL_SAMP + '_2d_CAE_IMG_enc.keras',
    'CAE_3D': '/pscratch/sd/k/kberard/SCGSR/3D_VMC/Model_Train_dat/CAE_3D_Models/' + MODEL_SAMP + 'CAE_3D.keras',
}

# ==========================================
# 3. Keras Custom Objects
# ==========================================
@tf.keras.utils.register_keras_serializable()
class OnesLikeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.ones_like(inputs)

def ones_like_fn(a):
    return tf.ones_like(a)

@tf.keras.utils.register_keras_serializable()
class RenormalizeToEight(tf.keras.layers.Layer):
    def call(self, x):
        total = tf.reduce_sum(x, axis=[1, 2, 3, 4], keepdims=True)
        return x / (total + 1e-8) * 8.0

@tf.keras.utils.register_keras_serializable()
def jensen_shannon_divergence_loss(y_true, y_pred):
    y_t = tf.cast(y_true, tf.float32)
    y_p = tf.cast(y_pred, tf.float32)
    y_t = tf.reshape(y_t, [tf.shape(y_t)[0], -1])
    y_p = tf.reshape(y_p, [tf.shape(y_p)[0], -1])
    y_t /= tf.reduce_sum(y_t, axis=1, keepdims=True) + 1e-8
    y_p /= tf.reduce_sum(y_p, axis=1, keepdims=True) + 1e-8
    m = 0.5 * (y_t + y_p)
    kl_true = tf.reduce_sum(y_t * tf.math.log((y_t + 1e-8) / (m + 1e-8)), axis=1)
    kl_pred = tf.reduce_sum(y_p * tf.math.log((y_p + 1e-8) / (m + 1e-8)), axis=1)
    return tf.reduce_mean(0.5 * (kl_true + kl_pred))

# ==========================================
# 4. Utilities
# ==========================================
def D_JS(p1, p2, tol=1e-16):
    p1 = p1 / np.sum(p1)
    p2 = p2 / np.sum(p2)
    pm = (p1 + p2) / 2
    p1_nonzero = np.abs(p1) > tol
    p2_nonzero = np.abs(p2) > tol
    p1 = np.abs(p1[p1_nonzero])
    pm1 = np.abs(pm[p1_nonzero])
    p2 = np.abs(p2[p2_nonzero])
    pm2 = np.abs(pm[p2_nonzero])
    d = .5 * ((p1 * np.log(p1 / pm1)).sum() + (p2 * np.log(p2 / pm2)).sum())
    d /= np.log(2)  # normalize to max of 1
    return d

def encode_voxel_to_rgb(test_d, save_path='slice_scalers.npz'):
    rgb_volume = np.zeros((64, 64, 64, 3), dtype=np.float32)
    mins, maxs = [], []
    for i in range(64):
        slice_2d = test_d[i, :, :]
        s_min, s_max = float(slice_2d.min()), float(slice_2d.max())
        if s_max == s_min:
            s_max = s_min + 1e-6
        normed = (slice_2d - s_min) / (s_max - s_min)
        rgb_volume[i, :, :, :] = np.stack([normed]*3, axis=-1)
        mins.append(s_min)
        maxs.append(s_max)
    np.savez(save_path, mins=np.array(mins), maxs=np.array(maxs))
    return rgb_volume

def decode_rgb_to_voxel(rgb_volume, save_path='slice_scalers.npz'):
    data = np.load(save_path)
    mins, maxs = data['mins'], data['maxs']
    test_d = np.zeros((64, 64, 64), dtype=np.float32)
    for i in range(64):
        gray = rgb_volume[i, :, :, 0]
        test_d[i] = gray * (maxs[i] - mins[i]) + mins[i]
    return test_d

# SCUNet specific tensor conversions
def single2tensor4(img):
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)

def tensor2single(tensor):
    tensor = tensor.squeeze().cpu().detach().numpy()
    if tensor.ndim == 3:
        img_np = np.transpose(tensor, (1, 2, 0))
    else:
        img_np = tensor
    return np.clip(img_np, 0, 1)

def run_scunet_inference(noisy_rgb, model, device):
    denoised_rgb = np.zeros_like(noisy_rgb)
    for i in range(noisy_rgb.shape[0]):
        img = np.clip(noisy_rgb[i].astype(np.float32), 0, 1)
        img_tensor = single2tensor4(img).to(device)
        with torch.no_grad():
            output_tensor = model(img_tensor)
        denoised = tensor2single(output_tensor)
        if np.isnan(denoised).any():
            print(f"⚠️ NaNs detected in denoised slice {i}")
        denoised_rgb[i] = denoised
    return denoised_rgb

# ==========================================
# 5. Core Pipeline Logic
# ==========================================
def denoise_volume(test_d, model_name):
    """Routes the input density array to the appropriate model for denoising."""
    test_rgb = encode_voxel_to_rgb(test_d)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name.startswith('scunet'):
        torch.serialization.add_safe_globals([SCUNet])
        if model_name == 'scunet_pre':
            model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
            model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location='cpu'), strict=True)
        else:
            model = torch.load(MODEL_PATHS[model_name], map_location='cpu', weights_only=False)
        model.eval()
        model.to(device)
        denoised_rgb = run_scunet_inference(test_rgb, model, device)
        return decode_rgb_to_voxel(denoised_rgb)

    elif model_name == 'nature':
        model = tf.keras.models.load_model(MODEL_PATHS['nature'], custom_objects={'ones_like_fn': ones_like_fn})
        denoised_rgb = model.predict(test_rgb)
        return decode_rgb_to_voxel(denoised_rgb)

    elif model_name == 'CAE':
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(MODEL_PATHS['CAE'])
        denoised_rgb = model.predict(test_rgb)
        return decode_rgb_to_voxel(denoised_rgb)

    elif model_name == 'CAE_3D':
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(
            MODEL_PATHS['CAE_3D'],
            custom_objects={"RenormalizeToEight": RenormalizeToEight, "jensen_shannon_divergence_loss": jensen_shannon_divergence_loss}
        )
        reshaped_arr = test_d.reshape(1, 64, 64, 64, 1)
        denoised_rgb = model.predict(reshaped_arr)
        return denoised_rgb.reshape(64, 64, 64)

    elif model_name == 'bm3d':
        denoised_rgb = np.zeros_like(test_rgb)
        sigma = np.sqrt(1.0 / 100)
        for i in range(64):
            gray_input = test_rgb[i, :, :, 0]
            denoised_gray = bm3d(gray_input, sigma_psd=sigma)
            denoised_rgb[i] = np.stack([denoised_gray] * 3, axis=-1)
        return decode_rgb_to_voxel(denoised_rgb)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def evaluate_and_plot(test_d, denoised_d, ref_d, model_name):
    """Handles identical evaluation printouts and plotting for any model."""
    norm_noisy = np.linalg.norm(test_d - ref_d)
    norm_denoised = np.linalg.norm(denoised_d - ref_d)
    
    # Check if tensors need JS divergence for CAE_3D or standard D_JS
    if model_name == 'CAE_3D':
        jsd_noisy = float(jensen_shannon_divergence_loss(test_d, ref_d))
        jsd_denoised = float(jensen_shannon_divergence_loss(denoised_d, ref_d))
    else:
        jsd_noisy = D_JS(test_d, ref_d)
        jsd_denoised = D_JS(denoised_d, ref_d)

    print("\n>>> Evaluation Metrics 3D")
    print(f"2-norm (noisy vs ref):     {norm_noisy:.4f}")
    print(f"2-norm (denoised vs ref):  {norm_denoised:.4f}")
    print(f"JSD    (noisy vs ref):     {jsd_noisy:.6f}")
    print(f"JSD    (denoised vs ref):  {jsd_denoised:.6f}")

    slice_idx = 32
    plt.figure(figsize=(12, 6))
    titles = ['Noisy', 'Denoised', 'Reference']
    data = [test_d[slice_idx], denoised_d[slice_idx], ref_d[slice_idx]]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(data[i], cmap='viridis')
        plt.title(f"{titles[i]} (z={slice_idx})")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. Main CLI Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise density matrices using trained models.")
    parser.add_argument('--noisy', type=str, required=True, help="Path to the noisy .h5 file.")
    parser.add_argument('--ref', type=str, required=True, help="Path to the reference .h5 file.")
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_PATHS.keys()) + ['bm3d'], help="Model architecture to run.")
    parser.add_argument('--output', type=str, default="denoised_output.npy", help="Output .npy file path.")
    parser.add_argument('--quiet', action='store_true', help="Suppress evaluation prints and plotting.")
    args = parser.parse_args()

    # Load Data
    with h5py.File(args.noisy, 'r') as file:
        test_density = file['density'][:]
    with h5py.File(args.ref, 'r') as file:
        ref_density = file['density'][:]

    print(f"Starting denoising using {args.model}...")
    denoised_out = denoise_volume(test_density, args.model)

    if not args.quiet:
        evaluate_and_plot(test_density, denoised_out, ref_density, args.model)

    np.save(args.output, denoised_out)
    print(f"\nDenoised matrix successfully saved to {args.output}")