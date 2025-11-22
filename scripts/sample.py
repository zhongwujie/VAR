import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage
from tqdm import tqdm
import argparse

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 导入 VAR 项目的模块 (假设脚本在项目根目录运行)
from models import build_vae_var
# 根据您的要求导入打包函数
try:
    from utils.misc import create_npz_from_sample_folder
except ImportError:
    print("警告: 无法导入 'utils.misc.create_npz_from_sample_folder'。请确保您在 VAR 项目根目录下运行。")
    # 定义一个占位函数以防导入失败，避免脚本崩溃
    def create_npz_from_sample_folder(path):
        print(f"请手动运行 create_npz_from_sample_folder('{path}')")

################## 配置参数 ##################
MODEL_DEPTH = 16  # 根据您的模型深度修改: 16, 20, 24, 30
BATCH_SIZE = 50   # 每个类别生成的数量 (如果显存不足，可调小，如 25)
TOTAL_CLASSES = 1000
IMAGES_PER_CLASS = 50
SAMPLE_FOLDER = "samples_fid_50k"
NPZ_FILENAME = "samples_50k.npz"

# FID 推理参数
CFG = 1.5
TOP_P = 0.96
TOP_K = 900
MORE_SMOOTH = False
SEED = 0

################## 1. 模型准备 ##################
def prepare_model():
    # 禁用默认初始化以加速
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

    # 下载/加载 Checkpoint
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

    # 构建模型
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

    # 加载权重
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    
    return vae, var, device

################## 2. 采样主逻辑 ##################
def main():
    # 准备
    vae, var, device = prepare_model()
    os.makedirs(SAMPLE_FOLDER, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 加速设置
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    print(f"开始采样: {TOTAL_CLASSES} 类, 每类 {IMAGES_PER_CLASS} 张, 总计 {TOTAL_CLASSES * IMAGES_PER_CLASS} 张。")
    print(f"参数: cfg={CFG}, top_p={TOP_P}, top_k={TOP_K}, smooth={MORE_SMOOTH}")

    # 循环生成
    # 如果显存不够，可以将 BATCH_SIZE 设为 25，并在内部循环两次
    loops_per_class = IMAGES_PER_CLASS // BATCH_SIZE
    
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            
            for class_idx in tqdm(range(TOTAL_CLASSES), desc="Sampling Classes"):
                for loop_i in range(loops_per_class):
                    # 准备标签
                    label_tensor = torch.full((BATCH_SIZE,), class_idx, device=device, dtype=torch.long)
                    
                    # 推理
                    # 注意：g_seed 传递给函数，确保生成的多样性通常由函数内部处理，
                    # 但为了保险起见，如果您的 var 版本需要不同的 seed 来生成不同的 batch，
                    # 可以传递 SEED + class_idx * 100 + loop_i
                    recon_B3HW = var.autoregressive_infer_cfg(
                        B=BATCH_SIZE, 
                        label_B=label_tensor, 
                        cfg=CFG, 
                        top_k=TOP_K, 
                        top_p=TOP_P, 
                        g_seed=SEED, # 或者不传，让其随机
                        more_smooth=MORE_SMOOTH
                    )

                    # 后处理并保存图片
                    # recon_B3HW 通常是 Tensor (B, 3, H, W)，范围 [0, 1] (根据 demo 的 mul(255) 推断)
                    for i in range(BATCH_SIZE):
                        img_tensor = recon_B3HW[i]
                        
                        # 转换为 numpy uint8
                        # permute (3, H, W) -> (H, W, 3)
                        img_np = img_tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8)
                        
                        img_pil = PImage.fromarray(img_np)
                        
                        # 文件命名: class_index_sample_index.png
                        global_idx = loop_i * BATCH_SIZE + i
                        save_path = os.path.join(SAMPLE_FOLDER, f"{class_idx:04d}_{global_idx:03d}.png")
                        img_pil.save(save_path, "PNG")

    print(f"采样完成。图片保存在: {SAMPLE_FOLDER}")

    ################## 3. 打包 NPZ ##################
    print("正在打包为 .npz 文件...")
    if osp.exists(SAMPLE_FOLDER):
        create_npz_from_sample_folder(SAMPLE_FOLDER)
        # 注意：create_npz_from_sample_folder 通常会在同级目录生成一个 .npz 文件
        print(f"打包完成。请检查生成的 .npz 文件 (通常名为 {SAMPLE_FOLDER}.npz)。")
    else:
        print("错误: 采样文件夹不存在。")

if __name__ == "__main__":
    main()