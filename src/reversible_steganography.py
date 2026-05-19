import torch
import numpy as np

class ReversibleSteganography:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def _get_mask(self, shape):
        i, j = np.indices(shape)
        mask_context = (i + j) % 2 == 0
        mask_query = (i + j) % 2 == 1
        return mask_context, mask_query

    def preliminary_phase(self, input_image):
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        h, w = input_image.shape
        mask_context, mask_query = self._get_mask((h, w))

        context_image = np.zeros_like(input_image, dtype=np.float32)
        context_image[mask_context] = input_image[mask_context]
        tensor_context = torch.from_numpy(context_image / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)

        num_samples_t = 100
        all_y_hat = []
        all_sigma_sq_aleatorik = []

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        with torch.no_grad():
            for _ in range(num_samples_t):
                y_hat, sigma_sq = self.model(tensor_context)
                all_y_hat.append(y_hat.squeeze().cpu().numpy())
                all_sigma_sq_aleatoric.append(sigma_sq.squeeze().cpu().numpy())

        all_y_hat = np.array(all_y_hat)
        all_sigma_sq_aleatoric = np.array(all_sigma_sq_aleatoric)

        y_hat_final = np.mean(all_y_hat, axis=0) * 255.0
        epistemic_variance = np.var(all_y_hat, axis=0)
        aleatoric_variance = np.mean(all_sigma_sq_aleatoric, axis=0)
        
        norm_epistemic = epistemic_variance / (np.sum(epistemic_variance) + 1e-8)
        norm_aleatoric = aleatoric_variance / (np.sum(aleatoric_variance) + 1e-8)
        sigma_sq_final = norm_epistemic + norm_aleatoric

        return y_hat_final, sigma_sq_final, mask_query, mask_context

    def encode(self, cover_image, binary_message):
        y_hat, variance_map, mask_query, _ = self.preliminary_phase(cover_image)

        pred_int = np.round(y_hat).astype(np.float32)
        cover_int = np.round(cover_image).astype(np.float32)

        residual_query = (cover_int - pred_int)[mask_query]
        variance_query = variance_map[mask_query]
        pred_query = pred_int[mask_query]
        sort_indices = np.argsort(variance_query)

        safe_indices = []
        for idx in sort_indices:
            if 15 <= pred_query[idx] <= 240:
                safe_indices.append(idx)
        safe_indices = np.array(safe_indices)

        message_length = len(binary_message)
        if message_length > len(safe_indices):
            raise ValueError(f"Image capacity insufficient! Max safe capacity: {len(safe_indices)} bits.")

        original_pos_residual = np.copy(residual_query)
        
        for i in range(message_length):
            target_idx = safe_indices[i]
            epsilon = original_pos_residual[target_idx]
            bit_message = binary_message[i]
            original_pos_residual[target_idx] = (2 * epsilon) + bit_message

        stego_image = np.copy(cover_image).astype(np.float32)
        stego_image[mask_query] = np.mod(pred_int[mask_query] + original_pos_residual, 256)
        stego_image = np.round(stego_image)

        return stego_image

    def decode(self, stego_image, message_length):
        y_hat, variance_map, mask_query, _ = self.preliminary_phase(stego_image)

        pred_int = np.round(y_hat).astype(np.float32)
        stego_int = np.round(stego_image).astype(np.float32)

        diff = np.mod(stego_int - pred_int, 256)
        diff = np.where(diff > 127, diff - 256, diff)
        
        stego_residual_query = diff[mask_query]
        variance_query = variance_map[mask_query]
        pred_query = pred_int[mask_query]
        sort_indices = np.argsort(variance_query)

        safe_indices = []
        for idx in sort_indices:
            if 15 <= pred_query[idx] <= 240:
                safe_indices.append(idx)
        safe_indices = np.array(safe_indices)

        extracted_message = []
        original_pos_residual = np.copy(stego_residual_query)

        for i in range(message_length):
            target_idx = safe_indices[i]
            epsilon_prime = original_pos_residual[target_idx]
            
            bit = int(epsilon_prime % 2)
            extracted_message.append(bit)
            
            epsilon = np.floor(epsilon_prime / 2)
            original_pos_residual[target_idx] = epsilon

        recovered_image = np.copy(stego_image).astype(np.float32)
        recovered_image[mask_query] = np.mod(pred_int[mask_query] + original_pos_residual, 256)
        recovered_image = np.round(recovered_image)

        return extracted_message, recovered_image