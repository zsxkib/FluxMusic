import os 
import torch 
import argparse
import math 
from einops import rearrange, repeat
from PIL import Image
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan
import matplotlib.pyplot as plt
import numpy as np

from utils import load_t5, load_clap, load_ae
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

def prepare(t5, clip, img, prompt):
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

    print(img_ids.size(), txt.size(), vec.size())
    return img, {
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "y": vec.to(img.device),
    }

def main(args):
    print('generate with MusicFlux')
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_size = (256, 16) 

    model = build_model(args.version).to(device) 
    local_path = args.ckpt_path
    state_dict = torch.load(local_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict['ema'])
    model.eval()  # important! 
    diffusion = RF()

    # Setup VAE
    t5 = load_t5(device, max_length=256)
    clap = load_clap(device, max_length=256)

    vae = AutoencoderKL.from_pretrained(os.path.join(args.audioldm2_model_path, 'vae')).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(os.path.join(args.audioldm2_model_path, 'vocoder')).to(device)

    with open(args.prompt_file, 'r') as f: 
        conds_txt = f.readlines()
    L = len(conds_txt) 
    unconds_txt = ["low quality, gentle"] * L 
    print(L, conds_txt, unconds_txt) 

    init_noise = torch.randn(L, 8, latent_size[0], latent_size[1]).cuda() 

    STEPSIZE = 50
    img, conds = prepare(t5, clap, init_noise, conds_txt)
    _, unconds = prepare(t5, clap, init_noise, unconds_txt) 
    with torch.autocast(device_type='cuda'): 
        images = diffusion.sample_with_xps(model, img, conds=conds, null_cond=unconds, sample_steps = STEPSIZE, cfg = 7.0)
    
    print(images[-1].size(), )
    
    images = rearrange(
        images[-1], 
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=128,
        w=8,
        ph=2,
        pw=2,)
    latents = 1 / vae.config.scaling_factor * images
    mel_spectrogram = vae.decode(latents).sample 
    print(mel_spectrogram.size()) 
    
    # Create the 'wav' and 'spectrograms' directories if they don't exist
    os.makedirs('wav', exist_ok=True)
    os.makedirs('spectrograms', exist_ok=True)

    for i in range(L): 
        x_i = mel_spectrogram[i]
        if x_i.dim() == 4:
            x_i = x_i.squeeze(1)
        
        # Save spectrogram as image
        plt.figure(figsize=(10, 4))
        plt.imshow(x_i.cpu().numpy()[0], aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram {i+1}')
        plt.tight_layout()
        plt.savefig(f'spectrograms/spectrogram_{i+1}.png')
        plt.close()

        waveform = vocoder(x_i)
        waveform = waveform[0].cpu().float().detach().numpy()
        print(waveform.shape)
        from scipy.io import wavfile 
        wavfile.write(f'wav/sample_{i+1}.wav', 16000, waveform) 

    print("Spectrograms saved in 'spectrograms' directory")
    print("Audio files saved in 'wav' directory")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="small")
    parser.add_argument("--prompt_file", type=str, default='config/example.txt')
    parser.add_argument("--ckpt_path", type=str, default='musicflow_s.pt')
    parser.add_argument("--audioldm2_model_path", type=str, default='/maindata/data/shared/multimodal/public/dataset_music/audioldm2' )
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    main(args)


