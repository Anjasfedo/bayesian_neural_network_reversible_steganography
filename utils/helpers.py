import numpy as np
import math

def calculate_metrics(cover, stego):
    mse = np.mean((cover.astype(np.float64) - stego.astype(np.float64)) ** 2)
    if mse == 0:
        return 0.0, float('inf')
    psnr = 10 * math.log10((255.0 ** 2) / mse)
    return mse, psnr