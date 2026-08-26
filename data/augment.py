"""
AdaptSign — Weather Augmentation
IKS Principle: Panchang — Seasonal adaptation
Simulates Indian weather: Monsoon, Fog, Night, Sun
"""

import cv2
import numpy as np
from PIL import Image
import random


class WeatherAugmenter:
    """
    IKS: Panchang — The Indian almanac that tracks all seasons.
    Our model must work in all seasons, just as Panchang accounts for all.

    Methods:
        add_rain()   — Varsha Ritu (Monsoon)
        add_fog()    — Shishir Ritu (Winter fog)
        add_night()  — Ratri (Nighttime)
        add_shadow() — Chhaya (Shadow on road)
        add_dust()   — Grishma Ritu (Summer dust haze)
    """

    def __init__(self, severity='medium'):
        """
        severity: 'light' | 'medium' | 'heavy'
        """
        self.severity = severity
        self.severity_map = {
            'light': 0.3,
            'medium': 0.6,
            'heavy': 0.9
        }
        self.s = self.severity_map.get(severity, 0.6)

    # ── RAIN (Varsha Ritu — Monsoon) ──────────────────────────────────
    def add_rain(self, img, intensity=None):
        """
        IKS: Varsha Ritu — Monsoon season
        Adds rain streaks to image simulating Indian monsoon
        """
        if intensity is None:
            intensity = self.s

        img_np = self._to_numpy(img)
        h, w = img_np.shape[:2]

        # Create rain layer
        rain_layer = np.zeros_like(img_np)

        # Number of rain drops based on intensity
        num_drops = int(intensity * 300)

        for _ in range(num_drops):
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)
            # Rain falls at slight angle (like Indian monsoon)
            length = random.randint(5, 15)
            angle = random.randint(-10, 10)
            x2 = int(x1 + length * np.sin(np.radians(angle)))
            y2 = int(y1 + length * np.cos(np.radians(90 + angle)))
            x2 = np.clip(x2, 0, w - 1)
            y2 = np.clip(y2, 0, h - 1)

            cv2.line(rain_layer, (x1, y1), (x2, y2),
                     (200, 200, 220), 1)

        # Slight blur for realism
        rain_layer = cv2.blur(rain_layer, (1, 2))

        # Darken image slightly (rain = less light)
        dark_factor = 1.0 - intensity * 0.25
        img_np = (img_np * dark_factor).astype(np.uint8)

        # Blend rain
        alpha = intensity * 0.5
        result = cv2.addWeighted(img_np, 1.0, rain_layer, alpha, 0)

        return self._to_pil(result)

    # ── FOG (Shishir Ritu — Winter fog) ───────────────────────────────
    def add_fog(self, img, intensity=None):
        """
        IKS: Shishir Ritu — Winter season fog
        Common in North India (Delhi, Punjab) and hill stations
        """
        if intensity is None:
            intensity = self.s

        img_np = self._to_numpy(img)
        h, w = img_np.shape[:2]

        # Fog = blend with white layer
        fog_layer = np.full_like(img_np, 255)

        # Non-uniform fog (denser at edges/bottom like real fog)
        fog_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(fog_mask, (w // 2, h // 2), max(h, w),
                   intensity * 0.7, -1)
        fog_mask = cv2.GaussianBlur(fog_mask, (21, 21), 0)
        fog_mask = np.clip(fog_mask, 0, 1)

        # Apply per-channel
        fog_3ch = np.stack([fog_mask] * 3, axis=-1)
        result = (img_np * (1 - fog_3ch * intensity) +
                  fog_layer * fog_3ch * intensity)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return self._to_pil(result)

    # ── NIGHT (Ratri) ─────────────────────────────────────────────────
    def add_night(self, img, intensity=None):
        """
        IKS: Ratri — Nighttime
        Darkens image, adds yellow tint (Indian street lamp sodium vapor)
        """
        if intensity is None:
            intensity = self.s

        img_np = self._to_numpy(img).astype(np.float32)

        # Darken
        dark_factor = 1.0 - intensity * 0.7
        img_np = img_np * dark_factor

        # Add warm sodium-vapor lamp tint (yellow cast)
        img_np[:, :, 0] *= (1.0 + intensity * 0.15)  # slight red boost
        img_np[:, :, 1] *= (1.0 + intensity * 0.08)  # slight green boost
        img_np[:, :, 2] *= (1.0 - intensity * 0.15)  # reduce blue

        # Add headlight spotlight from bottom-center (car headlights)
        h, w = img_np.shape[:2]
        spotlight = np.zeros((h, w), dtype=np.float32)
        cv2.circle(spotlight, (w // 2, h),
                   int(min(h, w) * 0.8),
                   intensity * 0.4, -1)
        spotlight = cv2.GaussianBlur(spotlight, (31, 31), 0)
        spotlight_3ch = np.stack([spotlight] * 3, axis=-1)
        img_np = img_np + spotlight_3ch * 60

        result = np.clip(img_np, 0, 255).astype(np.uint8)
        return self._to_pil(result)

    # ── SHADOW (Chhaya) ───────────────────────────────────────────────
    def add_shadow(self, img, intensity=None):
        """
        IKS: Chhaya — Shadow
        Adds tree/building shadow common on Indian roads
        """
        if intensity is None:
            intensity = self.s * 0.7

        img_np = self._to_numpy(img)
        h, w = img_np.shape[:2]

        # Random shadow strip across image
        shadow_mask = np.ones((h, w), dtype=np.float32)
        x1 = random.randint(0, w // 2)
        x2 = random.randint(w // 2, w)
        pts = np.array([[x1, 0], [x2, 0], [w, h], [0, h]], dtype=np.int32)
        cv2.fillPoly(shadow_mask.reshape(h, w, 1)
                     if False else shadow_mask, [pts],
                     1.0 - intensity * 0.5)

        shadow_3ch = np.stack([shadow_mask] * 3, axis=-1)
        result = (img_np * shadow_3ch).astype(np.uint8)
        return self._to_pil(result)

    # ── DUST HAZE (Grishma Ritu — Summer) ────────────────────────────
    def add_dust(self, img, intensity=None):
        """
        IKS: Grishma Ritu — Summer heat haze
        Brown/yellow dust haze common in Rajasthan, Tamil Nadu summer
        """
        if intensity is None:
            intensity = self.s * 0.7

        img_np = self._to_numpy(img).astype(np.float32)

        # Warm yellow-brown haze
        haze = np.full_like(img_np, [200, 170, 100], dtype=np.float32)

        result = img_np * (1 - intensity * 0.4) + haze * intensity * 0.4
        result = np.clip(result, 0, 255).astype(np.uint8)
        return self._to_pil(result)

    # ── APPLY RANDOM WEATHER ──────────────────────────────────────────
    def apply_random(self, img):
        """Apply a random weather effect"""
        choice = random.choice([
            self.add_rain,
            self.add_fog,
            self.add_night,
            self.add_shadow,
            self.add_dust
        ])
        return choice(img)

    # ── HELPERS ───────────────────────────────────────────────────────
    def _to_numpy(self, img):
        """Convert PIL Image or numpy to numpy uint8"""
        if isinstance(img, Image.Image):
            return np.array(img.convert('RGB'))
        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return img
        else:
            raise TypeError(f"Expected PIL or numpy, got {type(img)}")

    def _to_pil(self, img_np):
        """Convert numpy to PIL Image"""
        return Image.fromarray(img_np.astype(np.uint8))

    def augment_dataset_folder(self, input_dir, output_dir, weather='rain'):
        """
        Augment an entire folder of images.
        Usage: augment all sunny GTSRB images to create rainy variants.
        """
        import os
        from tqdm import tqdm

        os.makedirs(output_dir, exist_ok=True)
        fn_map = {
            'rain': self.add_rain,
            'fog': self.add_fog,
            'night': self.add_night,
            'dust': self.add_dust
        }
        fn = fn_map.get(weather, self.add_rain)

        count = 0
        for root, _, files in os.walk(input_dir):
            for fname in tqdm(files, desc=f'Augmenting {weather}'):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                rel = os.path.relpath(root, input_dir)
                out_sub = os.path.join(output_dir, rel)
                os.makedirs(out_sub, exist_ok=True)

                img = Image.open(os.path.join(root, fname)).convert('RGB')
                aug = fn(img)
                aug.save(os.path.join(out_sub, fname))
                count += 1

        print(f"[AUGMENT] {count} images saved to {output_dir}")


# ─────────────────────────────────────────
# QUICK TEST / DEMO
# ─────────────────────────────────────────

if __name__ == '__main__':
    import os

    aug = WeatherAugmenter(severity='medium')

    # Create a simple test image
    test_img = Image.fromarray(
        np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)
    )

    effects = {
        'rain': aug.add_rain(test_img),
        'fog': aug.add_fog(test_img),
        'night': aug.add_night(test_img),
        'dust': aug.add_dust(test_img),
    }

    os.makedirs('./results', exist_ok=True)
    for name, img in effects.items():
        img.save(f'./results/test_{name}.png')
        print(f"Saved test_{name}.png")

    print("Weather augmentation OK ✓")
