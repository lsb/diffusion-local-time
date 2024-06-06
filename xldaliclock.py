import torch

torch.num_threads=16

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
from datetime import datetime, timedelta
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderTiny
from diffusers.models.attention_processor import SlicedAttnProcessor
from tqdm import tqdm
from pathlib import Path

atkbold = ImageFont.truetype("OCRB.ttf", 500)

image_size = (1600, 900)
screen_size = image_size

def mask_image(timestamp):
    mask_text = timestamp.strftime(f"%H\u2009%M\u2009%S")
    #mask_text = timestamp.strftime(f"%H\u2009%M")
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
    return ImageOps.pad(time_img, image_size)

preferred_dtype = torch.float32
preferred_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
#preferred_device = "cpu"

ctlnetmodelname = "monster-labs/control_v1p_sd15_qrcode_monster"
ctlnetmodelsubfolder = "v2"
sdmodelname = "SimianLuo/LCM_Dreamshaper_v7"
sdconstructor = StableDiffusionControlNetPipeline
infsteps = 4
ctlnetmodelname = "monster-labs/control_v1p_sdxl_qrcode_monster"
ctlnetmodelsubfolder = None
sdmodelname = "stabilityai/stable-diffusion-xl-base-1.0"
sdconstructor = StableDiffusionXLControlNetPipeline
infsteps = 20

controlnet = ControlNetModel.from_pretrained(
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

#pipe.vae.set_attn_processor(SlicedAttnProcessor(4))
#pipe.unet.set_attn_processor(SlicedAttnProcessor(4))

pipe.unet = torch.compile(pipe.unet)

current_denoising_steps = infsteps
target_filename = "beauty.png"
mask_image(timestamp=datetime.now()).save(target_filename)
print("just latents")
print(pipe(prompt="a clock", image=mask_image(timestamp=datetime.now()), num_inference_steps=4, guidance_scale=7.0, controlnet_conditioning_scale=1.5, output_type="latent").images[0])
print("full")
print(pipe(prompt="a clock", image=mask_image(timestamp=datetime.now()), num_inference_steps=4, guidance_scale=7.0, controlnet_conditioning_scale=1.5).images[0])


cali1 = "desert landscape with tall mountains and cactus and boulders at sunrise with the sun on the horizon"
cali2 = "stony river in a sunny redwood forest with salmon and deer and bears and mushrooms"
cali3 = "beach, tall cliffs, sun at the horizon, albatross eating fish, no one on the beach, boulders and tide pools in the shallow water"
cali4 = "nighttime photo of a desert landscape with the milky way in the sky and boulders on a shallow lake bed surrounded by tall mountains"

prompts = [
#    cali1,
#    cali2,
#    cali3,
    cali4,
]
conditioning_scales = {
    cali1: 0.45,
    cali2: 0.5,
    cali3: 0.45,
    cali4: 1.0,
}
negative_prompt = "low quality, ugly, wrong"

fps = 4
iteration_range = range(0, 86400 * fps)

def ease(x):
    return (x * x) * (3 - 2 * x) # (x * x * x * x * x) * (126 - 420 * x + 540 * x * x - 315 * x * x * x + 70 * x * x * x * x)

pipe.set_progress_bar_config(disable=True)

for iteration in tqdm(iteration_range):
    synthetic_time = datetime(year=2000,month=1,day=1,hour=0,minute=0,second=0) + timedelta(microseconds=iteration*(1000000/fps))
    target_filename = f"dali-{int(iteration/fps*100):08}.png"

    if Path(target_filename).exists():
        continue

    this_second = synthetic_time
    next_second = synthetic_time + timedelta(seconds=60)
    this_mask = mask_image(this_second).filter(ImageFilter.GaussianBlur(3))
    next_mask = mask_image(next_second).filter(ImageFilter.GaussianBlur(3))
    easing_step = ease((iteration % (fps * 60)) / (fps * 60))
    current_mask_image = Image.fromarray(np.array(this_mask) * (1-easing_step) + np.array(next_mask) * (easing_step))

#    image = current_mask_image.convert("RGB")
    image = pipe(
        prompt=prompts[iteration % len(prompts)],
        negative_prompt=negative_prompt,
        image=current_mask_image,
        num_inference_steps=current_denoising_steps,
        guidance_scale=7.0,
        controlnet_conditioning_scale=conditioning_scales[prompts[iteration % len(prompts)]],
        generator=torch.manual_seed(int(synthetic_time.timestamp()) // 60 // 60),
        height=image_size[1],
        width=image_size[0],
    ).images[0]
    if False:
        draw = ImageDraw.Draw(image)
        draw.text((60, screen_size[1]-60), f"leebutterman.com", fill=(255,255,255), font=atkbold_smol)
    image.save(target_filename)
