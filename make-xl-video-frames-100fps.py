# generate 100 fps for the 0419-0421 animation at 128 steps per transition
# generate 100 fps for the 24 hour animation at 8 steps per transition

from pathlib import Path
from tqdm import tqdm
from diffusers.utils import make_image_grid, load_image
from PIL import Image

def ease(t):
    return 3*t**2 - 2*t**3

def ease6(t):
    return 924 * (t ** 13) - 6006 * (t ** 12) + 16380 * (t ** 11) - 24024 * (t ** 10) + 20020 * (t ** 9) - 9009 * (t ** 8) + 1716 * (t ** 7)


framerate = 5
dest_size = (600,337)
top_image = Image.open(Path("xl-morph/xl-morph-0600-0.0000.png")).resize(dest_size)
bottom_image = Image.open(Path("xl-morph/xl-morph-1800-0.0000.png")).resize(dest_size)

for frameno in tqdm(range(framerate * 60 * 2)):
    hour = 4
    minute = frameno // (framerate * 60) + 19
    fractional_minute = (frameno % (framerate * 60)) / (framerate * 60 * 1.0)
    eased_fractional_minute = ease6(fractional_minute)
    quantized_eased_fractional_minute = round(eased_fractional_minute * 128) / 128.0
    if quantized_eased_fractional_minute == 1.0:
        minute += 1
        quantized_eased_fractional_minute = 0.0
    frame_to_render = Path(f"xl-morph/xl-morph-{hour:02d}{minute:02d}-{quantized_eased_fractional_minute:.4f}.png")
    frame_to_write = Path(f"04190421-{framerate}fps-{frameno:06d}.png")
    center_image = Image.open(frame_to_render).resize(dest_size)
    make_image_grid(images=[i for i in [top_image, center_image, bottom_image]], cols=1, rows=3).save(frame_to_write)
    # frame_to_write.write_bytes(frame_to_render.read_bytes())
    # print(quantized_eased_fractional_minute * 128)

for frameno in tqdm(range(5 * 60 * 60 * 24)):
    hour = frameno // (5 * 60 * 60)
    minute = (frameno % (5 * 60 * 60)) // (5 * 60)
    fractional_minute = (minute * 5 * 60 - frameno) / (5 * 60)
    eased_fractional_minute = ease(fractional_minute)
    quantized_eased_fractional_minute = round(eased_fractional_minute * 8) / 8.0
    frame_to_render = Path(f"xl-morph-{hour:02d}{minute:02d}-{quantized_eased_fractional_minute:.4f}.png")
    frame_to_write = Path(f"24hour-5fps-{frameno:06d}.png")
    # frame_to_write.write_bytes(frame_to_render.read_bytes())
