import torch

torch.num_threads=16

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
from datetime import datetime, timedelta
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, FluxControlNetModel, FluxControlNetPipeline
from tqdm import tqdm
from pathlib import Path

atkbold = ImageFont.truetype("OCRB.ttf", 720)
atkbold = ImageFont.truetype("Atkinson-Hyperlegible-Bold-102.otf", 925)

image_size = (1920, 1024)
screen_size = image_size

def mask_image(mask_text):
    time_img = Image.new("L", image_size, (0,))
    draw = ImageDraw.Draw(time_img)
    draw.multiline_text(
        # xy=(-30,120),
        xy=(0, 120),
        text=mask_text,
        fill=(255,),
        font=atkbold,
        align="center",
        spacing=-10,
    )
    # return time_img
    (i_left, i_top, i_right, i_bottom) = time_img.getbbox()
    # pad the image horizonally to the full size
    i_left = 0
    i_right = image_size[0]
    time_img = time_img.crop((i_left, i_top, i_right, i_bottom))
    return ImageOps.pad(time_img, image_size).convert("RGB").filter(ImageFilter.GaussianBlur(50))

preferred_dtype = torch.float32
# preferred_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
preferred_device = "cpu"

# ctlnetmodelname = "monster-labs/control_v1p_sd15_qrcode_monster"
# ctlnetmodelsubfolder = "v2"
# sdmodelname = "SimianLuo/LCM_Dreamshaper_v7"
# sdconstructor = StableDiffusionControlNetPipeline
# infsteps = 4
# ctlnetmodelname = "monster-labs/control_v1p_sdxl_qrcode_monster"
# ctlnetmodelsubfolder = None
# sdmodelname = "stabilityai/stable-diffusion-xl-base-1.0"
# sdconstructor = StableDiffusionXLControlNetPipeline
infsteps = 20
ctlnetmodelname = "Xlabs-AI/flux-controlnet-depth-diffusers"
# ctlnetmodelname = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"
ctlnetmodelsubfolder = None
sdmodelname = "black-forest-labs/FLUX.1-dev"
sdconstructor = FluxControlNetPipeline

controlnet = FluxControlNetModel.from_pretrained(
    ctlnetmodelname,
    subfolder=ctlnetmodelsubfolder,
    torch_dtype=preferred_dtype,
).to(preferred_device)

pipe = sdconstructor.from_pretrained(
    sdmodelname,
    controlnet=controlnet,
    torch_dtype=preferred_dtype,
    safety_checker=None,
).to(preferred_device)

current_denoising_steps = infsteps
target_filename = "beauty.png"
mask_image("LSB").save(target_filename)
print("full")
for iter in tqdm(range(10)):
  for scale in tqdm([0.21, 0.23, 0.26, 0.28, 0.31, 0.34, 0.37, 0.41, 0.44, 0.47, 0.51, 0.54, 0.57, 0.61]):
    pipe(
       prompt="a pen and watercolor illustration of C-3PO with a red and white Santa hat using a polishing cloth on a shiny metallic box in a sunny workshop",
       control_image=mask_image("LSB"),
       num_inference_steps=current_denoising_steps,
       guidance_scale=3.5,
    #    control_guidance_start=0.2,
    #    control_guidance_end=0.8,
       controlnet_conditioning_scale=scale,
       height=image_size[1],
       width=image_size[0],
    #    generator=torch.manual_seed(85337+iter),
       num_images_per_prompt=1,
    ).images[0].save(f"flux-c3po-lsb.depth.{scale}.{iter}.png")
