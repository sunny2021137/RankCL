import torch
import torch.nn.functional as F

def ranking_aware_contrastive_loss(self, outputs, labels, n_classes):
    # (B, D) -> (N, B//N, D)
    x_repeated = outputs.view(n_classes, outputs.shape[0]//n_classes, outputs.shape[-1])
    # (B) -> (N, B//N)
    y_repeated = labels.view(n_classes, labels.shape[0]//n_classes)
    # (N, B//N, D) -> (N, B, D)
    x_repeated = x_repeated.repeat(1, n_classes, 1)
    # (N, B//N) -> (N, B)
    y_repeated = y_repeated.repeat(1, n_classes)
    perms = torch.stack([torch.randperm(x_repeated.shape[1]) for _ in range(n_classes)]).to(self.device)  # shape: (N, B)
    # (N, B, D)  
    x_repeated = torch.gather(x_repeated, dim=1, index=perms.unsqueeze(-1).expand(-1, -1, x_repeated.shape[-1]))
    # (N, B)
    y_repeated = torch.gather(y_repeated, dim=1, index=perms)
    # (N, B, D) -> (B, N, D)
    x_repeated = x_repeated.permute(1, 0, 2)
    # (N, B) -> (B, N)
    y_repeated = y_repeated.permute(1, 0)

    # === cosine similarity ===
    # Normalize
    x_norm = F.normalize(x_repeated, dim=-1)  # (B, N, D)
    out_norm = F.normalize(outputs.unsqueeze(1), dim=-1)  # (B, 1, D)
    
    # Cosine similarity: (B, N)
    cos_sim = torch.sum(x_norm * out_norm, dim=-1)
    
    
    # (B, N, 1) - (B, 1, N) -> (B, N, N)
    diff =  cos_sim.unsqueeze(1) - cos_sim.unsqueeze(2)
    
    # y
    # (B) -> (B, N, N)
    y_anchor = labels.unsqueeze(1).unsqueeze(2).expand(-1, n_classes, n_classes)
    # (B, N) -> (B, N, 1) -> (B, N, N)
    y_first = y_repeated.unsqueeze(2).expand(-1, n_classes, n_classes)
    y_second = y_repeated.unsqueeze(1).expand(-1, n_classes, n_classes)
    # 條件 1: tensor2 > tensor1 > tensor0 → 設為 1
    cond1 = (y_second > y_first) & (y_first > y_anchor)
    # 條件 2: tensor0 > tensor1 > tensor2 → 設為 1
    cond2 = (y_anchor > y_first) & (y_first > y_second)
    # (B, N, N)
    mask = torch.where(cond1 | cond2, 1, 0)
    # (B, N, N)
    diff = torch.where(diff>0, diff+1, torch.exp(torch.clamp(diff, max=0))) * mask
    # (B, N, N) -> (B, N) -> (B) -> 1
    loss = torch.sum(diff, dim=-1).sum(dim=-1).mean()
    return loss