import torch
import torch.nn as nn

class BNNLoss(nn.Module):
    def __init__(self, lambda_reg=1.0, eps=1e-3):
        super(BNNLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.eps = eps

    def forward(self, y_true, y_pred, sigma_squared, mask_query):
        sigma_sq = torch.clamp(sigma_squared, min=self.eps)
        
        y_true_q = y_true[mask_query]
        y_pred_q = y_pred[mask_query]
        sigma_sq_q = sigma_sq[mask_query]
        
        squared_error = (y_true_q - y_pred_q) ** 2
        sum_sigma_sq = torch.sum(sigma_sq_q)
        normalized_variance = sigma_sq_q / (sum_sigma_sq + self.eps)
        
        inverse_norm_var = 1.0 / (normalized_variance + self.eps)
        distance_loss = torch.mean(squared_error * inverse_norm_var)
        regularizer_loss = torch.mean(torch.log(sigma_sq_q))
        
        total_loss = distance_loss + (self.lambda_reg * regularizer_loss)
        return total_loss