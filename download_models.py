import os
from huggingface_hub import hf_hub_download

MODEL_CACHE = "models"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

def download_folder(repo_id, folder_path, local_dir):
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="model")
    folder_files = [f for f in files if f.startswith(folder_path)]
    
    for file in folder_files:
        local_path = os.path.join(local_dir, file[len(folder_path):].lstrip('/'))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        hf_hub_download(repo_id=repo_id, filename=file, local_dir=os.path.dirname(local_path))
        print(f"Downloaded {file} to {local_path}")

def main():
    # VAE
    download_folder("cvssp/audioldm2", "vae", "models/audioldm2/vae")

    # Vocoder
    download_folder("cvssp/audioldm2", "vocoder", "models/audioldm2/vocoder")

    # T5-XXL
    download_folder("stabilityai/stable-diffusion-3-medium-diffusers", "text_encoder_3", "models/t5-xxl")

    # CLAP-L
    download_folder("laion/larger_clap_music", "", "models/clap-l")

    # FluxMusic models
    flux_models = ["musicflow_s.pt", "musicflow_b.pt", "musicflow_l.pt", "musicflow_g.pt"]
    for filename in flux_models:
        download_folder("feizhengcong/FluxMusic", filename, f"models/fluxmusic")

if __name__ == "__main__":
    main()