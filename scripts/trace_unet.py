import argparse
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import load_model_from_config
from ldm.modules.diffusionmodules.util import extract_into_tensor
import torchvision
from torch import autocast
import os
import numpy as np
import open_clip

class DDPM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @autocast('cuda')
    @torch.no_grad()
    def diffusion_model_1(self, img, condition, unconditional_condition, t, t_prev):
        x_in = torch.cat([img] * 2)
        t_in = torch.cat([t] * 2)
        condition_in = torch.cat([unconditional_condition, condition])
        e_t_uncond, e_t = self.model.model.diffusion_model(x_in, t_in, condition_in).chunk(2)
        model_output = e_t_uncond + self.scale * (e_t - e_t_uncond)
        e_t = self.model.predict_eps_from_z_and_v(img, t, model_output)
        pred_x0 = self.model.predict_start_from_z_and_v(img, t, model_output)
        img = self.model.predict_start_from_z_and_v(pred_x0, t_prev, e_t)
        return img
    
    @autocast('cuda')
    @torch.no_grad()
    def diffusion_model(self, x_in, t_in, condition_in):
        return self.model.model.diffusion_model(x_in, t_in, condition_in)
    
    @autocast('cuda')
    @torch.no_grad()
    def forward(self, x):
        return x
    
    @autocast('cuda')
    @torch.no_grad()
    def predict_eps_from_z_and_v(self, z, t, v):
        return self.model.predict_eps_from_z_and_v(z, t, v)
    
    @autocast('cuda')
    @torch.no_grad()
    def predict_start_from_z_and_v(self, z, t, v):
        return self.model.predict_start_from_z_and_v(z, t, v)
    
    @autocast('cuda')
    @torch.no_grad()
    def q_sample(self, z, t, v):
        return self.model.q_sample(z, t, v)

    @torch.no_grad()
    def decode_image(self, img):
        return self.model.decode_first_stage(img)
    
    @torch.no_grad()
    def clip_encoder(self, tokens):
        encode = self.model.cond_stage_model.encode_with_transformer(tokens)
        return encode

class AutoencoderKL(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @autocast('cuda')
    @torch.no_grad()
    def forward(self, img):
        return self.model.decode(img)

class ClipEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @autocast('cuda')
    @torch.no_grad()
    def forward(self, tokens):
        return self.model.encode_with_transformer(tokens)
    
def main(opt):
    torch.set_grad_enabled(False)
    seed_everything(42)
    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device=device, verbose=True)
    autoklencoder = AutoencoderKL(model.first_stage_model)
    clipEncoder = ClipEncoder(model.cond_stage_model)
    model.first_stage_model = None
    model.cond_stage_model = None
    scale = 9.0
    sampler = DDPM(model)
    with torch.no_grad():
        img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32, device = device)
        t_in = torch.ones(2, dtype=torch.int64, device = device)
        t_prev = torch.ones(opt.batch, dtype=torch.int64, device = device)
        context = torch.ones(2, 77, 1024, dtype=torch.float32, device = device)
        tokens = torch.ones((opt.batch, 77), dtype=torch.int64, device = device)
        uncondition_context = torch.ones(opt.batch, 77, 1024, dtype=torch.float32, device = device)
        input = {
            'diffusion_model': (img_in,t_in, context),
            #'decode_image': (img_in, ),
            #'clip_encoder': (tokens, ),
            'predict_eps_from_z_and_v': (img_in, t_in, img_in),
            'predict_start_from_z_and_v': (img_in, t_in, img_in),
            'q_sample': (img_in, t_in, img_in),
        }
        # scripted_sampler = sampler
        scripted_sampler = torch.jit.trace_module(sampler, input)
        samples_ddim = torch.randn(opt.batch, 4, 96, 96, device=device)
        scripted_clip_encoder = torch.jit.trace(clipEncoder, (tokens))
        scripted_decoder = torch.jit.trace(autoklencoder, (samples_ddim))
        # scripted_sampler = torch.jit.optimize_for_inference(scripted_sampler, ['diffusion_model', 'predict_eps_from_z_and_v', 'predict_start_from_z_and_v', 'q_sample'])
        # scripted_clip_encoder = torch.jit.optimize_for_inference(scripted_clip_encoder)
        # scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
        scripted_sampler.save(f"{opt.output}/ddim_v_sampler.pt")
        scripted_decoder.save(f"{opt.output}/decoder.pt")
        scripted_clip_encoder.save(f"{opt.output}/clip_encoder.pt")

        n_step = 50
        timesteps = 1000
        ddim_timesteps = np.asarray(list(range(0, timesteps, timesteps // n_step)))
        ddim_timesteps = ddim_timesteps + 1
        # betas = model.betas
        # alpha_bars = model.alphas_cumprod
        img = torch.randn(opt.batch, 4, 96, 96, device=device)
        t = torch.ones(opt.batch, dtype=torch.int32, device=device)
        prompts = opt.batch * ["cute green dog"]
        tokens = open_clip.tokenize(prompts)
        tokens = tokens.to(device)
        condition = scripted_clip_encoder(tokens)
        unconditional_prompt = opt.batch * [""]
        unconditional_tokens = open_clip.tokenize(unconditional_prompt)
        unconditional_tokens = unconditional_tokens.to(device)
        unconditional_condition = scripted_clip_encoder(unconditional_tokens)
        for index, i in enumerate(reversed(ddim_timesteps)):
            index = n_step - index - 1
            t = torch.full((opt.batch,), i, dtype=torch.int64, device=device)
            t_prev = torch.full((opt.batch,), i - 20 if i - 20 >= 0 else 0, dtype=torch.int64, device=device)
            x_in = torch.cat([img]*2)
            t_in = torch.cat([t]*2)
            condition_in = torch.cat([unconditional_condition, condition])
            e_t_uncond, e_t = scripted_sampler.diffusion_model(x_in, t_in, condition_in).chunk(2)
            model_output = e_t_uncond + 9.0 * (e_t - e_t_uncond)
            e_t = scripted_sampler.predict_eps_from_z_and_v(img, t, model_output)
            pred_x0 = scripted_sampler.predict_start_from_z_and_v(img, t, model_output)
            img = scripted_sampler.q_sample(pred_x0, t_prev, e_t)
            print(img.dtype)

        decoded_images = scripted_decoder(img)
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