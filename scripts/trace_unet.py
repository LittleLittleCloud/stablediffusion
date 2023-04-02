import argparse
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch
from ldm.util import load_model_from_config
from ldm.modules.diffusionmodules.util import extract_into_tensor
import torchvision
from torch import autocast
import os
import numpy as np


def main(opt):
    torch.set_grad_enabled(False)
    seed_everything(42)
    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device=device, verbose=True)
    with torch.no_grad():
        img_in = torch.ones(opt.batch * 2, 4, 96, 96, dtype=torch.float32, device = device)
        t_in = torch.ones(opt.batch * 2, dtype=torch.int64, device = device)
        context = torch.ones(opt.batch * 2, 77, 1024, dtype=torch.float32, device = device)
        samples_ddim = torch.randn(opt.batch, 4, 96, 96, device=device)
        input = {
            'forward': (img_in, t_in, context),
            # 'decode_first_stage': (samples_ddim),
        }
        scripted_diffusion_model = torch.jit.trace(model.model.diffusion_model, (img_in, t_in, context))
        # scripted_unet.save(f"{opt.output}/unet.pt")

        # get Decoder for first stage model scripted
        scripted_decoder = torch.jit.trace_module(model, {'decode_first_stage': samples_ddim})
        # scripted_decoder.save(f"{opt.output}/decoder.pt")

        scale = 9.0
        n_step = 50
        timesteps = 1000
        ddim_timesteps = np.asarray(list(range(0, timesteps, timesteps // n_step)))
        ddim_timesteps = ddim_timesteps + 1

        betas = torch.linspace(0.00085, 0.012, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, 0)
        # betas = model.betas
        # alpha_bars = model.alphas_cumprod
        sqrt_alphas_bar = torch.sqrt(alpha_bars)
        sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alpha_bars)

        alpha_bars_prev = torch.cat((torch.full((1,),alpha_bars[0], device=device), alpha_bars[ddim_timesteps[:-1]] ), 0)
        alpha_bars = alpha_bars[ddim_timesteps]
        img = torch.randn(opt.batch, 4, 96, 96, device=device)
        t = torch.ones(opt.batch, dtype=torch.int32, device=device)
        prompts = opt.batch * ["cute black cat"]
        condition = model.get_learned_conditioning(prompts)
        unconditional_condition = model.get_learned_conditioning(opt.batch * [""])
        for index, i in enumerate(reversed(ddim_timesteps)):
            index = n_step - index - 1
            t = torch.full((opt.batch,), i, dtype=torch.int64, device=device)
            x_in = torch.cat([img] * 2)
            t_in = torch.cat([t] * 2)
            condition_in = torch.cat([unconditional_condition, condition])
            e_t_uncond, e_t = scripted_diffusion_model(x_in, t_in, condition_in).chunk(2)
            model_output = e_t_uncond + scale * (e_t - e_t_uncond)
            e_t = extract_into_tensor(sqrt_alphas_bar, t, img.shape) * model_output + extract_into_tensor(sqrt_one_minus_alphas_bar, t, img.shape) * img
            # e_t = model.predict_eps_from_z_and_v(img, t, model_output)
            # alpha_bar_t = torch.full((opt.batch, 1, 1, 1), alpha_bars[index], dtype=torch.float16, device=device)
            alpha_bar_t_prev = torch.full((opt.batch, 1, 1, 1), alpha_bars_prev[index], dtype=torch.float16, device=device)
            # pred_x0 = (img - torch.sqrt(1-alpha_bar_t) * e_t) / torch.sqrt(alpha_bar_t)
            pred_x0 = extract_into_tensor(sqrt_alphas_bar, t, img.shape) * img - extract_into_tensor(sqrt_one_minus_alphas_bar, t, img.shape) * model_output
            # pred_x0 = model.predict_start_from_z_and_v(img, t, model_output)
            img = torch.sqrt(alpha_bar_t_prev) * pred_x0 + torch.sqrt(1-alpha_bar_t_prev) * e_t

        # img = 1. / model.scale_factor * img
        decoded_images = scripted_decoder.decode_first_stage(img)
        decoded_images = torch.clamp((decoded_images + 1.0) / 2.0, 0.0, 1.0)
        outpath = opt.output
        base_count=0
        sample_path = os.path.join(outpath, "samples")
        for x_sample in decoded_images:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
            base_count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default="outputs")
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cuda"
    )
    opt = parser.parse_args()
    main(opt)