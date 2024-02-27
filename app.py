import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from datetime import datetime, timedelta
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderTiny
from diffusers.models.attention_processor import SlicedAttnProcessor
from tqdm import tqdm

def adjust_gamma(img, gamma=0.4):
    npim = np.array(img)
    npim_gamma = 255.0 * (npim / 255.0) ** gamma
    return Image.fromarray(np.uint8(npim_gamma))

atkbold = ImageFont.truetype("Atkinson-Hyperlegible-Bold-102.otf", 600)
atkbold_smol = ImageFont.truetype("Atkinson-Hyperlegible-Bold-102.otf", 40)

image_size = (1440, 810)
screen_size = image_size

def mask_image(timestamp):
    is_two_line = False # our images are 4:3 instead of 1:1, so we have space for all on one line
    linesep = "\n" if is_two_line else ""
    mask_text = timestamp.strftime(f"%-I{linesep}\u2009%M")
    time_img = Image.new("L", image_size, (0,))
    draw = ImageDraw.Draw(time_img)
    draw.multiline_text(
        xy=(-30,120),
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

controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster",
    subfolder="v2",
    torch_dtype=preferred_dtype,
).to(preferred_device)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    controlnet=controlnet,
    torch_dtype=preferred_dtype,
    safety_checker=None,
).to(preferred_device)

#pipe.vae.set_attn_processor(SlicedAttnProcessor(4))
#pipe.unet.set_attn_processor(SlicedAttnProcessor(4))

current_denoising_steps = 4
current_latency = 0
rounding_minutes = 1
target_filename = "beauty.png"
mask_image(timestamp=datetime.now()).save(target_filename)

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
    cali4: 0.7,
}
negative_prompt = "low quality, ugly, wrong"

demo_mode = True
if demo_mode:
    iteration_range = range(len(prompts) * (60 // rounding_minutes) * 48)
else:
    iteration_range = range(86400 * 365 * 80)

for iteration in tqdm(iteration_range):
    if demo_mode:
        synthetic_time = datetime(year=2000,month=1,day=1,hour=0,minute=0,second=0) + timedelta(seconds=iteration*rounding_minutes*60-1)
        target_filename = f"face-render-{iteration:06}.png"
        current_latency = 0
        pre_render_time = synthetic_time
    else:
        pre_render_time = datetime.now()

    target_time_plus_midpoint = pre_render_time + timedelta(seconds=(current_latency + rounding_minutes * 60 / 2))
    rounded_target_time = target_time_plus_midpoint - timedelta(minutes=target_time_plus_midpoint.minute - target_time_plus_midpoint.minute // rounding_minutes * rounding_minutes)
    current_mask_image = mask_image(timestamp=rounded_target_time)
    print(f"current_latency: {current_latency}, pre_render_time: {pre_render_time}, rounded_target_time: {rounded_target_time}, current_denoising_steps: {current_denoising_steps}\n")

    image = pipe(
        prompt=prompts[iteration % len(prompts)],
        negative_prompt=negative_prompt,
        image=current_mask_image,
        num_inference_steps=current_denoising_steps,
        guidance_scale=7.0,
        controlnet_conditioning_scale=conditioning_scales[prompts[iteration % len(prompts)]],
        generator=torch.manual_seed(int(rounded_target_time.timestamp()) // 3600),
        height=image_size[1],
        width=image_size[0],
    ).images[0]
    image = adjust_gamma(image, gamma=0.5)
    image = ImageEnhance.Sharpness(image).enhance(5)
    image = image.resize(screen_size)
    if True:
        draw = ImageDraw.Draw(image)
        draw.text((60, screen_size[1]-60), f"leebutterman.com", fill=(255,255,255), font=atkbold_smol)
    image.save(target_filename)
    post_render_time = datetime.now()
    current_latency = post_render_time.timestamp() - pre_render_time.timestamp()
