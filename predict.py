# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog

import os
import subprocess
import time

MODEL_CACHE = "models"
BASE_URL = f"https://weights.replicate.delivery/default/FluxMusic/{MODEL_CACHE}/"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE
import torch
from cog import BasePredictor, Input, Path
from pydantic import BaseModel
from typing import Optional
from einops import rearrange, repeat
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan
import matplotlib.pyplot as plt
from scipy.io import wavfile


class Output(BaseModel):
    wav: Path
    melspectrogram: Optional[Path]


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        global load_t5, load_clap, RF, build_model

        model_files = [
            "fluxmusic.tar",
            "models--laion--larger_clap_music.tar",
            "models--roberta-base.tar",
            "models--stabilityai--stable-diffusion-3-medium-diffusers.tar",
            "vae.tar",
            "vocoder.tar",
        ]

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        for model_file in model_files:
            url = BASE_URL + model_file

            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        from utils import load_t5, load_clap
        from train import RF
        from constants import build_model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.diffusion = RF()
        self.t5 = load_t5(self.device, max_length=256)
        self.clap = load_clap(self.device, max_length=256)
        self.vae = AutoencoderKL.from_pretrained("models/vae").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("models/vocoder").to(self.device)

        # Initialize model and weights as None for lazy loading
        self.model = None
        self.current_weights = None

    def load_model(self, version):
        """Lazy load the model weights based on user-friendly version names"""
        if self.current_weights != version:

            # Map version names to weight filenames
            version_to_weight = {
                "small": "musicflow_s.pt",
                "base": "musicflow_b.pt",
                "large": "musicflow_l.pt",
                "giant": "musicflow_g.pt",
            }

            weights = version_to_weight.get(
                version, "musicflow_b.pt"
            )  # Default to "base" if not found

            self.model = build_model(version).to(self.device)
            state_dict = torch.load(
                f"models/fluxmusic/{weights}", map_location=self.device
            )
            self.model.load_state_dict(state_dict["ema"])
            self.model.eval()
            self.current_weights = version
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
            default="The song is an epic blend of space-rock, rock, and post-rock genres.",
        ),
        negative_prompt: str = Input(
            description="Text prompt for negative guidance (unconditioned prompt)",
            default="low quality, gentle",
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale", ge=0.0, le=20.0, default=7.0
        ),
        model_version: str = Input(
            description="Select the model version to use",
            choices=["small", "base", "large", "giant"],
            default="base",
        ),
        steps: int = Input(
            description="Number of sampling steps", ge=1, le=200, default=50
        ),
        save_spectrogram: bool = Input(
            description="Whether to save the spectrogram image", default=False
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
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

        # Lazy load the model with specified version
        model = self.load_model(model_version)

        latent_size = (256, 16)
        init_noise = torch.randn(1, 8, latent_size[0], latent_size[1]).to(self.device)

        img, conds = self.prepare(self.t5, self.clap, init_noise, prompt)
        _, unconds = self.prepare(self.t5, self.clap, init_noise, negative_prompt)

        with torch.autocast(device_type="cuda"):
            images = self.diffusion.sample_with_xps(
                model,
                img,
                conds=conds,
                null_cond=unconds,
                sample_steps=steps,
                cfg=guidance_scale,
            )

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
            plt.imshow(x_i.cpu().numpy()[0], aspect="auto", origin="lower")
            plt.axis("off")  # Turn off the axes
            plt.tight_layout(pad=0)  # Remove padding
            plt.savefig(melspectrogram_path, bbox_inches="tight", pad_inches=0)
            plt.close()

        return Output(wav=wav_path, melspectrogram=melspectrogram_path)
