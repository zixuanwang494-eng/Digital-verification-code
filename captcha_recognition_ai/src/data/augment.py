import json
import random
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class AugmentParams:
    blur_radius: float
    rotation: float
    brightness: float
    contrast: float
    noise_sigma: float
    line_count: int
    perspective_strength: float


def random_params(rng: random.Random) -> AugmentParams:
    return AugmentParams(
        blur_radius=rng.uniform(0.1, 0.9),
        rotation=rng.uniform(-4.0, 4.0),
        brightness=rng.uniform(0.85, 1.15),
        contrast=rng.uniform(0.85, 1.25),
        noise_sigma=rng.uniform(2.0, 9.0),
        line_count=rng.randint(0, 3),
        perspective_strength=rng.uniform(0.0, 0.035),
    )


def params_to_json(params: AugmentParams) -> str:
    return json.dumps(asdict(params), ensure_ascii=False)


def _add_noise(image: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    arr = np.array(image).astype(np.float32)
    noise = rng.normalvariate
    sampled = np.array([noise(0, sigma) for _ in range(arr.size)]).reshape(arr.shape)
    arr = np.clip(arr + sampled, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _draw_interference_lines(image: Image.Image, count: int, rng: random.Random) -> Image.Image:
    if count <= 0:
        return image

    from PIL import ImageDraw

    img = image.convert("L")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for _ in range(count):
        y1 = rng.randint(4, height - 5)
        y2 = rng.randint(4, height - 5)
        shade = rng.randint(70, 150)
        draw.line((0, y1, width, y2), fill=shade, width=rng.randint(1, 2))
    return img


def _mild_perspective(image: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    if strength <= 0:
        return image
    width, height = image.size
    max_dx = width * strength
    max_dy = height * strength
    coeffs = (
        1,
        rng.uniform(-strength, strength),
        rng.uniform(-max_dx, max_dx),
        rng.uniform(-strength, strength),
        1,
        rng.uniform(-max_dy, max_dy),
        rng.uniform(-strength / 100, strength / 100),
        rng.uniform(-strength / 100, strength / 100),
    )
    return image.transform(image.size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def augment_captcha(image: Image.Image, params: AugmentParams, rng: random.Random) -> Image.Image:
    """Apply mild captcha-like distortions while keeping digits human-readable."""
    img = ImageOps.grayscale(image)
    img = img.rotate(params.rotation, resample=Image.Resampling.BICUBIC, fillcolor=255)
    img = _mild_perspective(img, params.perspective_strength, rng)
    img = ImageEnhance.Brightness(img).enhance(params.brightness)
    img = ImageEnhance.Contrast(img).enhance(params.contrast)
    img = _draw_interference_lines(img, params.line_count, rng)
    img = _add_noise(img, params.noise_sigma, rng)
    img = img.filter(ImageFilter.GaussianBlur(radius=params.blur_radius))
    return img
