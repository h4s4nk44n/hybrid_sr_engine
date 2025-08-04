# training/losses.py
import torch
import torch.nn as nn
import lpips

class CombinedLoss(nn.Module):
    """
    L1 and LPIPS losses, combined and weighted.
    """
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        # Basic loss function
        self.l1_loss = nn.L1Loss()
        
        # --- THIS IS THE UPGRADE ---
        # Load the LPIPS model. 'vgg' network is higher quality than 'alex'.
        # This will only impact training time, NOT final model inference time.
        # The model will be automatically downloaded on first use.
        print("Loading VGG network for LPIPS loss...")
        self.perceptual_loss = lpips.LPIPS(net='vgg')
        print("LPIPS model loaded.")

    def forward(self, predicted, target):
        """
        Calculates the total loss between the predicted and target images.

        Args:
            predicted (torch.Tensor): The high-resolution image produced by the model.
            target (torch.Tensor): The ground truth high-resolution image.

        Returns:
            torch.Tensor: The total, weighted loss value.
        """
        # Calculate L1 loss
        loss_l1 = self.l1_loss(predicted, target)
        
        # Calculate LPIPS loss
        # LPIPS expects images normalized to the [-1, 1] range, while we use [0, 1].
        # We apply the 2*x - 1 transform to convert them.
        loss_lpips = self.perceptual_loss(predicted * 2 - 1, target * 2 - 1).mean()
        
        # Combine the losses with their weights
        total_loss = (self.l1_weight * loss_l1) + (self.perceptual_weight * loss_lpips)
        
        return total_loss