# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog

import os
import torch
import math
from cog import BasePredictor, Input, Path
from pydantic import BaseModel
from typing import Optional
from einops import rearrange, repeat
from PIL import Image
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from utils import load_t5, load_clap
from train import RF
from constants import build_model

MODEL_CACHE = "models"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

class Output(BaseModel):
    wav: Optional[Path]
    melspectrogram: Optional[Path]

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.diffusion = RF()
        self.t5 = load_t5(self.device, max_length=256)
        self.clap = load_clap(self.device, max_length=256)
        self.vae = AutoencoderKL.from_pretrained("models/audioldm2/vae").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("models/audioldm2/vocoder").to(self.device)

        # Initialize model and weights as None for lazy loading
        self.model = None
        self.current_weights = None

    def load_model(self, weights):
        """Lazy load the model weights"""
        if self.current_weights != weights:
            print(f"Loading new weights: {weights}")
            self.model = build_model(weights.split('_')[1]).to(self.device)
            state_dict = torch.load(f"models/fluxmusic/{weights}", map_location=self.device)
            self.model.load_state_dict(state_dict['ema'])
            self.model.eval()
            self.current_weights = weights
        return self.model

    def prepare(self, t5, clip, img, prompt):
        bs, c, h, w = img.shape
        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        if isinstance(prompt, str):
            prompt = [prompt]
        txt = t5(prompt)
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        vec = clip(prompt)
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)

        return img, {
            "img_ids": img_ids.to(img.device),
            "txt": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
            "y": vec.to(img.device),
        }

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for music generation",
            default="The song is an epic blend of space-rock, rock, and post-rock genres."
        ),
        weights: str = Input(
            description="Model weights to use",
            choices=["musicflow_b.pt", "musicflow_g.pt", "musicflow_l.pt", "musicflow_s.pt"],
            default="musicflow_b.pt"
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        save_spectrogram: bool = Input(
            description="Whether to save the spectrogram image",
            default=False
        )
    # ) -> Output:
    ) -> Path:
        """
        Generate audio based on the input prompt.
        This method can be called multiple times, lazy loading different weights as needed.
        """
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)
        torch.set_grad_enabled(False)

        # Lazy load the model with specified weights
        model = self.load_model(weights)

        latent_size = (256, 16)
        init_noise = torch.randn(1, 8, latent_size[0], latent_size[1]).to(self.device)

        uncond_prompt = "low quality, gentle"
        
        img, conds = self.prepare(self.t5, self.clap, init_noise, prompt)
        _, unconds = self.prepare(self.t5, self.clap, init_noise, uncond_prompt)

        STEPSIZE = 50
        CFG = 7.0
        with torch.autocast(device_type='cuda'):
            images = self.diffusion.sample_with_xps(model, img, conds=conds, null_cond=unconds, sample_steps=STEPSIZE, cfg=CFG)

        images = rearrange(
            images[-1],
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=128,
            w=8,
            ph=2,
            pw=2,
        )
        latents = 1 / self.vae.config.scaling_factor * images
        mel_spectrogram = self.vae.decode(latents).sample

        x_i = mel_spectrogram[0]
        if x_i.dim() == 4:
            x_i = x_i.squeeze(1)

        waveform = self.vocoder(x_i)
        waveform = waveform[0].cpu().float().detach().numpy()

        wav_path = Path("output.wav")
        wavfile.write(wav_path, 16000, waveform)

        melspectrogram_path = None
        if save_spectrogram:
            melspectrogram_path = Path("spectrogram.png")
            plt.figure(figsize=(10, 4))
            plt.imshow(x_i.cpu().numpy()[0], aspect='auto', origin='lower')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            plt.savefig(melspectrogram_path)
            plt.close()

        # return Output(wav=wav_path, melspectrogram=melspectrogram_path)
        return wav_path