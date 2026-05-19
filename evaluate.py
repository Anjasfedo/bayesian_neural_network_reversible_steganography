import os
import torch
import numpy as np
from PIL import Image

from src.models.resden import ResDen
from src.reversible_steganography import ReversibleSteganography
from utils.helpers import calculate_metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResDen().to(device)
    model_path = "./result/model_bnn.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Model file not found at {model_path}. Please run train.py first.")
        exit()

    stego_system = ReversibleSteganography(model, device)
    
    bpp = 0.3
    total_pixels = 256 * 256
    message_length = int(bpp * total_pixels)
    secret_message = np.random.randint(0, 2, size=message_length).tolist()
    
    data_directory = "./data"
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    for file_name in os.listdir(data_directory):
        if file_name.lower().endswith(valid_extensions):
            file_path = os.path.join(data_directory, file_name)
            img = Image.open(file_path).convert('L')
            img = img.resize((256, 256))
            cover_image = np.array(img)
            
            print(f"\n--- Evaluating: {file_name} ---")
            
            try:
                stego_image = stego_system.encode(cover_image, secret_message)
                extracted_message, recovered_image = stego_system.decode(stego_image, message_length)
                
                is_message_valid = (secret_message == extracted_message)
                is_image_valid = np.array_equal(cover_image, np.round(recovered_image).astype(np.uint8))
                mse, psnr = calculate_metrics(cover_image, stego_image)
                
                print(f"Message Intact   : {is_message_valid}")
                print(f"Image Recovered  : {is_image_valid}")
                print(f"Stego PSNR       : {psnr:.2f} dB")
                print(f"Stego MSE        : {mse:.4f}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")