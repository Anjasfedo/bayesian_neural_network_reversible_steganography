import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

from src.models.resden import ResDen
from src.models.bnnloss import BNNLoss

class LocalStegoDataset(Dataset):
    def __init__(self, data_dir):
        self.images = []
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith(valid_extensions):
                file_path = os.path.join(data_dir, file_name)
                img = Image.open(file_path).convert('L')
                img = img.resize((256, 256))
                arr = np.array(img, dtype=np.float32) / 255.0
                
                h, w = arr.shape
                i, j = np.indices((h, w))
                mask_context = (i + j) % 2 == 0
                mask_query = (i + j) % 2 == 1
                
                img_context = arr * mask_context
                tensor_target = torch.from_numpy(arr).unsqueeze(0)
                tensor_input = torch.from_numpy(img_context).unsqueeze(0)
                tensor_mask_query = torch.from_numpy(mask_query).unsqueeze(0)
                
                self.images.append((tensor_input, tensor_target, tensor_mask_query))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

def train_model(model, dataloader, criterion, optimizer, device, epochs):
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_mse = 0.0
        
        for x_context, y_target, mask_query in dataloader:
            x_context = x_context.to(device)
            y_target = y_target.to(device)
            mask_query = mask_query.to(device)
            
            optimizer.zero_grad()
            y_pred, sigma_sq = model(x_context)
            
            loss = criterion(y_target, y_pred, sigma_sq, mask_query)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                y_true_q = y_target[mask_query]
                y_pred_q = y_pred[mask_query]
                pure_mse = torch.mean((y_true_q - y_pred_q) ** 2)
                total_mse += pure_mse.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1} | BNN Loss: {avg_loss:.4f} | Pure MSE: {avg_mse:.6f}")
        
    return loss_history

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_directory = "./data"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Please add training images to the '{data_directory}' folder and restart.")
        exit()

    dataset = LocalStegoDataset(data_directory)
    if len(dataset) == 0:
        print(f"No images found in '{data_directory}'. Please add some images.")
        exit()

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = ResDen().to(device)
    criterion = BNNLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 200
    loss_history = train_model(model, dataloader, criterion, optimizer, device, epochs)
    
    save_dir = "./result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    torch.save(model.state_dict(), f"{save_dir}/model_bnn.pth")
    print(f"Model successfully saved to {save_dir}/model_bnn.pth")