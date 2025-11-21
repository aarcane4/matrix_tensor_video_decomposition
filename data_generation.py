# data generation
import numpy as np

def generate_background(H=128, W=128):
    y, x = np.ogrid[:H, :W]
    center = (H/2, W/2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    return 255 * np.exp(-((r - 40)**2)/(2*20**2))

def add_noise(img, sigma=0.02):
    img = img / 255
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 1) * 255

def build_tensor(bg, frames=10):
    return np.stack([add_noise(bg) for _ in range(frames)], axis=2)

