import torch
import torch.nn as nn

class TemporalSegmentationLoss(nn.Module):
    def __init__(self, K1, K2, K3, K4):
        super(TemporalSegmentationLoss, self).__init__()
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
    
    def forward(self, p, s):
        # s <= K1
        case1_mask = s <= self.K1
        loss1 = -torch.log(1 - p + 1e-6)
        
        # K1 < s <= K2
        case2_mask = (s > self.K1) & (s <= self.K2)
        scale = (self.K2 - s) / (self.K2 - self.K1)
        loss2 = -torch.log(1 - torch.clamp(scale * p, min=0, max=1) + 1e-6)
        
        # K2 < s < 0
        case3_mask = (s > self.K2) & (s < 0)
        loss3 = torch.zeros_like(p)  # No loss
        
        # 0 <= s < K3
        case4_mask = (s >= 0) & (s < self.K3)
        scale = s / self.K3
        loss4 = -torch.log(torch.clamp(scale + (1 - scale) * p, min=1e-6, max=1))
        
        # K3 <= s < K4
        case5_mask = (s >= self.K3) & (s < self.K4)
        scale = (s - self.K3) / (self.K4 - self.K3)
        loss5 = -torch.log(1 - torch.clamp(scale * p, min=0, max=1) + 1e-6)
        
        # s >= K4
        case6_mask = s >= self.K4
        loss6 = -torch.log(1 - p + 1e-6)
        
        # Combine losses
        loss = (
            case1_mask * loss1 +
            case2_mask * loss2 +
            case3_mask * loss3 +
            case4_mask * loss4 +
            case5_mask * loss5 +
            case6_mask * loss6
        )

        # Compute mean loss
        return loss.mean()
