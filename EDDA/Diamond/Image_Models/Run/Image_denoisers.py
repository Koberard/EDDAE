import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf

from bm3d import bm3d
from utils import utils_image as util
from models.network_scunet import SCUNet
from keras.models import load_model
import keras
keras.config.enable_unsafe_deserialization()


class DensityDenoiser:
    """
    Unified denoising interface for all density pipelines.
    """

    # ==========================================================
    # INIT
    # ==========================================================

    def __init__(
        self,
        model_sample: str,
        device: str | None = None,
        volume_shape=(64, 64, 64),
        scaler_path="slice_scalers.npz",
    ):
        self.model_sample = model_sample
        self.volume_shape = volume_shape
        self.scaler_path = scaler_path

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Models (lazy-loaded)
        self.scunet_pre = None
        self.scunet_trained = None
        self.scunet_ft = None
        self.cae_2d = None
        self.nature = None

    # ==========================================================
    # IO
    # ==========================================================

    @staticmethod
    def load_density(path):
        with h5py.File(path, "r") as f:
            return f["density"][:]

    # ==========================================================
    # METRICS
    # ==========================================================

    @staticmethod
    def D_JS(p1, p2, tol=1e-16):
        p1 = p1 / np.sum(p1)
        p2 = p2 / np.sum(p2)
        pm = 0.5 * (p1 + p2)

        m1 = np.abs(p1) > tol
        m2 = np.abs(p2) > tol

        p1, pm1 = p1[m1], pm[m1]
        p2, pm2 = p2[m2], pm[m2]

        d = 0.5 * (
            np.sum(p1 * np.log(p1 / pm1)) +
            np.sum(p2 * np.log(p2 / pm2))
        )
        return d / np.log(2)

    # ==========================================================
    # ENCODING / DECODING
    # ==========================================================

    def encode_voxel_to_rgb(self, volume):
        z, h, w = self.volume_shape
        rgb = np.zeros((z, h, w, 3), dtype=np.float32)

        mins, maxs = [], []

        for i in range(z):
            sl = volume[i]
            smin, smax = sl.min(), sl.max()
            if smax == smin:
                smax += 1e-6

            norm = (sl - smin) / (smax - smin)
            rgb[i] = np.stack([norm] * 3, axis=-1)
            mins.append(smin)
            maxs.append(smax)

        np.savez(self.scaler_path, mins=mins, maxs=maxs)
        return rgb

    def decode_rgb_to_voxel(self, rgb):
        data = np.load(self.scaler_path)
        mins, maxs = data["mins"], data["maxs"]

        z, h, w, _ = rgb.shape
        out = np.zeros((z, h, w), dtype=np.float32)

        for i in range(z):
            gray = rgb[i, :, :, 0]
            out[i] = gray * (maxs[i] - mins[i]) + mins[i]

        return out

    # ==========================================================
    # MODEL LOADERS
    # ==========================================================

    def load_scunet_pretrained(self, path):
        model = SCUNet(in_nc=3, config=[4]*7, dim=64)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval().to(self.device)
        self.scunet_pre = model

    def load_scunet_trained(self, path):
        torch.serialization.add_safe_globals([SCUNet])
        self.scunet_trained = torch.load(path, map_location="cpu", weights_only=False)
        self.scunet_trained.eval().to(self.device)

    def load_scunet_ft(self, path):
        torch.serialization.add_safe_globals([SCUNet])
        self.scunet_ft = torch.load(path, map_location="cpu", weights_only=False)
        self.scunet_ft.eval().to(self.device)

    def load_cae_2d(self, path):
        self.cae_2d = load_model(path)

    def load_nature(self, path, custom_objects):
        self.nature = load_model(path, custom_objects=custom_objects)

    # ==========================================================
    # SCUNET INFERENCE
    # ==========================================================

    def _scunet_slice(self, img, model):
        img = img.astype(np.float32)
        tensor = util.single2tensor4(img).to(self.device)
        with torch.no_grad():
            out = model(tensor)
        out = util.tensor2single(out)
        return np.clip(out.transpose(1, 2, 0), 0, 1)

    def _run_scunet(self, noisy, ref, model):
        noisy_rgb = self.encode_voxel_to_rgb(noisy)
        den_rgb = np.zeros_like(noisy_rgb)

        for i in range(noisy_rgb.shape[0]):
            den_rgb[i] = self._scunet_slice(noisy_rgb[i], model)

        den = self.decode_rgb_to_voxel(den_rgb)
        return den, self.D_JS(den, ref)

    # ==========================================================
    # PIPELINES
    # ==========================================================

    def scunet_pretrained(self, noisy, ref):
        return self._run_scunet(noisy, ref, self.scunet_pre)

    def scunet_trained_pipeline(self, noisy, ref):
        return self._run_scunet(noisy, ref, self.scunet_trained)

    def scunet_ft_pipeline(self, noisy, ref):
        return self._run_scunet(noisy, ref, self.scunet_ft)

    def bm3d_pipeline(self, noisy, ref, N=100):
        sigma = np.sqrt(1.0 / N)
        noisy_rgb = self.encode_voxel_to_rgb(noisy)
        den_rgb = np.zeros_like(noisy_rgb)

        for i in range(noisy_rgb.shape[0]):
            gray = noisy_rgb[i, :, :, 0]
            d = bm3d(gray, sigma_psd=sigma)
            den_rgb[i] = np.stack([d]*3, axis=-1)

        den = self.decode_rgb_to_voxel(den_rgb)
        return den, self.D_JS(den, ref)

    def cae_pipeline(self, noisy, ref):
        noisy_rgb = self.encode_voxel_to_rgb(noisy)
        den_rgb = self.cae_2d.predict(noisy_rgb, verbose=0)
        den = self.decode_rgb_to_voxel(den_rgb)
        return den, self.D_JS(den, ref)

    def nature_pipeline(self, noisy, ref):
        noisy_rgb = self.encode_voxel_to_rgb(noisy)
        den_rgb = self.nature.predict(noisy_rgb, verbose=0)
        den = self.decode_rgb_to_voxel(den_rgb)
        return den, self.D_JS(den, ref)

    # ==========================================================
    # DISPATCHER
    # ==========================================================

    def run(self, noisy_path, ref_path, model_name):
        noisy = self.load_density(noisy_path)
        ref = self.load_density(ref_path)

        pipelines = {
            "scunet_pre": self.scunet_pretrained,
            "scunet_trained": self.scunet_trained_pipeline,
            "scunet_ft": self.scunet_ft_pipeline,
            "bm3d": self.bm3d_pipeline,
            "CAE": self.cae_pipeline,
            "nature": self.nature_pipeline,
        }

        if model_name not in pipelines:
            raise ValueError(f"Unknown pipeline: {model_name}")

        return pipelines[model_name](noisy, ref)
