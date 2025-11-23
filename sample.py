import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage
from tqdm import tqdm
import argparse

# Set environment variables
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Import VAR project modules
from models import build_vae_var
# Import packaging function as requested
from utils.misc import create_npz_from_sample_folder

################## Configuration Parameters ##################
MODEL_DEPTH = 16  # Modify according to your model depth: 16, 20, 24, 30
BATCH_SIZE = 50   # Number of images generated per class (reduce to e.g., 25 if VRAM is insufficient)
TOTAL_CLASSES = 1000
IMAGES_PER_CLASS = 50
SAMPLE_FOLDER = "samples_fid_50k"
NPZ_FILENAME = "samples_50k.npz"

# FID Inference Parameters
CFG = 1.5
TOP_P = 0.96
TOP_K = 900
MORE_SMOOTH = False
SEED = 0

################## 1. Model Preparation ##################
def prepare_model():
    # Disable default initialization for acceleration
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

    # Download/Load Checkpoint
    try:
        from huggingface_hub import hf_hub_download
        vae_ckpt = hf_hub_download(repo_id="FoundationVision/var", filename="vae_ch160v4096z32.pth")
        var_ckpt = hf_hub_download(repo_id="FoundationVision/var", filename=f"var_d{MODEL_DEPTH}.pth")
    except Exception as e:
        hf_home = 'https://hf-mirror.com/FoundationVision/var/resolve/main'
        vae_ckpt_local = 'vae_ch160v4096z32.pth'
        var_ckpt_local = f'var_d{MODEL_DEPTH}.pth'
        if not osp.exists(vae_ckpt_local): os.system(f'wget -O {vae_ckpt_local} {hf_home}/vae_ch160v4096z32.pth')
        if not osp.exists(var_ckpt_local): os.system(f'wget -O {var_ckpt_local} {hf_home}/{var_ckpt_local}')
        vae_ckpt, var_ckpt = vae_ckpt_local, var_ckpt_local

    # Build model
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

    # Load weights
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    
    return vae, var, device

################## 2. Main Sampling Logic ##################
def main():
    # Preparation
    vae, var, device = prepare_model()
    os.makedirs(SAMPLE_FOLDER, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Acceleration settings
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    print(f"Start sampling: {TOTAL_CLASSES} classes, {IMAGES_PER_CLASS} images per class, total {TOTAL_CLASSES * IMAGES_PER_CLASS} images.")
    print(f"Parameters: cfg={CFG}, top_p={TOP_P}, top_k={TOP_K}, smooth={MORE_SMOOTH}")

    # Loop for generation
    # If VRAM is insufficient, set BATCH_SIZE to 25 and loop twice internally
    loops_per_class = IMAGES_PER_CLASS // BATCH_SIZE
    
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            
            for class_idx in tqdm(range(TOTAL_CLASSES), desc="Sampling Classes"):
                for loop_i in range(loops_per_class):
                    # Prepare labels
                    label_tensor = torch.full((BATCH_SIZE,), class_idx, device=device, dtype=torch.long)
                    
                    # Inference
                    # Note: g_seed is passed to the function. Diversity is usually handled internally,
                    # but to be safe, if your VAR version needs different seeds for different batches,
                    # you can pass SEED + class_idx * 100 + loop_i
                    recon_B3HW = var.autoregressive_infer_cfg(
                        B=BATCH_SIZE, 
                        label_B=label_tensor, 
                        cfg=CFG, 
                        top_k=TOP_K, 
                        top_p=TOP_P, 
                        g_seed=SEED, # Or leave it empty for random
                        more_smooth=MORE_SMOOTH
                    )

                    # Post-process and save images
                    # recon_B3HW is usually Tensor (B, 3, H, W), range [0, 1] (inferred from mul(255) in demo)
                    for i in range(BATCH_SIZE):
                        img_tensor = recon_B3HW[i]
                        
                        # Convert to numpy uint8
                        # permute (3, H, W) -> (H, W, 3)
                        img_np = img_tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8)
                        
                        img_pil = PImage.fromarray(img_np)
                        
                        # File naming: class_index_sample_index.png
                        global_idx = loop_i * BATCH_SIZE + i
                        save_path = os.path.join(SAMPLE_FOLDER, f"{class_idx:04d}_{global_idx:03d}.png")
                        img_pil.save(save_path, "PNG")

    print(f"Sampling completed. Images saved in: {SAMPLE_FOLDER}")

    ################## 3. Pack NPZ ##################
    print("Packing into .npz file...")
    if osp.exists(SAMPLE_FOLDER):
        create_npz_from_sample_folder(SAMPLE_FOLDER)
        # Note: create_npz_from_sample_folder usually generates a .npz file in the same directory
        print(f"Packing completed. Please check the generated .npz file (usually named {SAMPLE_FOLDER}.npz).")
    else:
        print("Error: Sample folder does not exist.")

if __name__ == "__main__":
    main()