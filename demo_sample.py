
################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint (use HF cache by default)
try:
    from huggingface_hub import hf_hub_download
    vae_ckpt = hf_hub_download(repo_id="FoundationVision/var", filename="vae_ch160v4096z32.pth")
    var_ckpt = hf_hub_download(repo_id="FoundationVision/var", filename=f"var_d{MODEL_DEPTH}.pth")
except Exception as e:
    # Fallback: download to local CWD (existing behavior) if hf_hub_download is unavailable or fails
    hf_home = 'https://hf-mirror.com/FoundationVision/var/resolve/main'
    vae_ckpt_local = 'vae_ch160v4096z32.pth'
    var_ckpt_local = f'var_d{MODEL_DEPTH}.pth'
    if not osp.exists(vae_ckpt_local): os.system(f'wget -O {vae_ckpt_local} {hf_home}/vae_ch160v4096z32.pth')
    if not osp.exists(var_ckpt_local): os.system(f'wget -O {var_ckpt_local} {hf_home}/{var_ckpt_local}')
    vae_ckpt, var_ckpt = vae_ckpt_local, var_ckpt_local

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

############################# 2. Sample with classifier-free guidance

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = (980, 437, 22, 562)  #@param {type:"raw"}
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# sample
for label in class_labels:
    label_dir = f"figure/VAR-{MODEL_DEPTH}/class_{label}"
    os.makedirs(label_dir, exist_ok=True)
    print(f"Generating images for class {label}...")

    # generate 5 images per class
    for i in range(5):
        # vary seed for each image to get diversity
        current_seed = seed + i 
        torch.manual_seed(current_seed)
        random.seed(current_seed)
        np.random.seed(current_seed)
        
        B = 1
        label_B: torch.LongTensor = torch.tensor([label], device=device)
        
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=current_seed, more_smooth=more_smooth)

        chw = recon_B3HW[0] # take the first (and only) image
        chw = chw.permute(1, 2, 0).mul(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))

        output_path = osp.join(label_dir, f"sample_seed{current_seed}_cfg{cfg}.png")
        chw.save(output_path)
        print(f"  Saved {output_path}")

print("All finished!")
