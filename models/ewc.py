"""
AdaptSign — Elastic Weight Consolidation (EWC)
IKS Principle: Gurukul — Retain old knowledge while learning new

Based on: Kirkpatrick et al. 2017, PNAS
"Overcoming catastrophic forgetting in neural networks"

Our extension: Combined with DriftDetector to apply EWC
only when a real domain shift is detected (Viveka — discernment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm


class EWC:
    """
    Elastic Weight Consolidation.

    IKS: Gurukul System
    ───────────────────
    In the Gurukul system, a student learns new subjects each year
    but NEVER forgets what the Guru taught in previous years.
    The most important lessons are permanently etched in memory.

    EWC does the same for neural networks:
    - After each task, compute which weights are MOST IMPORTANT
    - When learning the next task, penalize changes to those weights
    - Result: The model can learn new tasks without forgetting old ones

    How it works:
    ─────────────
    Standard loss:    L = L_new(θ)
    EWC loss:         L = L_new(θ) + λ * Σ F_i * (θ_i - θ*_i)²

    Where:
        θ        = current parameters
        θ*_i     = optimal parameters from previous task
        F_i      = Fisher Information (importance of weight i)
        λ        = regularization strength (how much to protect)
    """

    def __init__(self, model, device='cpu', lambda_ewc=5000):
        """
        Args:
            model:      The neural network
            device:     'cuda' or 'cpu'
            lambda_ewc: Regularization strength.
                        Higher = stronger protection of old knowledge
                        Recommended: 1000-10000
        """
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc

        # Storage for each completed task
        # Each entry: {'fisher': dict, 'params': dict, 'task_name': str}
        self.task_memory = []

        print(f"[EWC] Initialized | λ = {lambda_ewc}")
        print("[EWC] IKS: Gurukul — continuous learning without forgetting")

    # ─────────────────────────────────────────
    # STEP 1: COMPUTE FISHER INFORMATION MATRIX
    # ─────────────────────────────────────────

    def compute_fisher(self, dataloader, task_name='Task', n_samples=500):
        """
        Compute Fisher Information Matrix for current task.
        Called AFTER training on a task, BEFORE training on the next.

        Fisher Information F_i measures how much the loss changes
        when we change weight θ_i.
        High F_i = Changing this weight hurts performance a LOT → PROTECT
        Low F_i  = This weight is not critical → Free to change

        IKS: Viveka (discernment) — identify what is truly important
        """
        print(f"\n[EWC] Computing Fisher Information for: {task_name}")
        print("[EWC] (Identifying which weights are most important...)")

        self.model.eval()

        # Initialize fisher dict with zeros
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        # Save current optimal parameters θ*
        optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optimal_params[name] = param.data.clone()

        # Compute Fisher via expected squared gradients
        count = 0
        for inputs, labels in tqdm(dataloader, desc='Fisher computation'):
            if count >= n_samples:
                break

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            # Log-likelihood loss
            log_probs = F.log_softmax(outputs, dim=1)

            # Sample from model's distribution
            # E[(∂ log p(y|x,θ) / ∂θ)²]
            loss = F.nll_loss(log_probs, labels)
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.clone().pow(2)

            count += inputs.size(0)

        # Normalize
        for name in fisher:
            fisher[name] /= count

        # Save to task memory
        self.task_memory.append({
            'fisher': fisher,
            'params': optimal_params,
            'task_name': task_name
        })

        # Summary stats
        all_fisher = torch.cat([f.flatten() for f in fisher.values()])
        print(f"[EWC] Fisher computed for {count} samples")
        print(f"[EWC] Mean importance: {all_fisher.mean():.6f}")
        print(f"[EWC] Max importance:  {all_fisher.max():.6f}")
        print(f"[EWC] Tasks in memory: {len(self.task_memory)}")

        return fisher

    # ─────────────────────────────────────────
    # STEP 2: COMPUTE EWC PENALTY
    # ─────────────────────────────────────────

    def penalty(self):
        """
        Compute EWC regularization penalty.

        L_ewc = λ * Σ_tasks Σ_params F_i * (θ_i - θ*_i)²

        This is added to the standard cross-entropy loss during training.
        The penalty grows larger when:
        1. F_i is large (weight was important for old task)
        2. (θ_i - θ*_i)² is large (weight changed a lot from old value)

        IKS: Gurukul — penalize forgetting important lessons
        """
        if not self.task_memory:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)

        for memory in self.task_memory:
            for name, param in self.model.named_parameters():
                if name not in memory['fisher']:
                    continue

                fisher = memory['fisher'][name].to(self.device)
                old_param = memory['params'][name].to(self.device)

                # Core EWC formula: F_i * (θ_i - θ*_i)²
                penalty += (fisher * (param - old_param).pow(2)).sum()

        return self.lambda_ewc * penalty

    # ─────────────────────────────────────────
    # UTILITY: GET FISHER HEATMAP DATA
    # ─────────────────────────────────────────

    def get_fisher_heatmap(self, layer_name=None, top_k=160):
        """
        Returns Fisher values as a flat array for visualization.
        Used in the Streamlit demo to show Fisher Memory Map.
        """
        if not self.task_memory:
            return None

        # Use most recent task memory
        fisher = self.task_memory[-1]['fisher']

        all_values = []
        for name, f in fisher.items():
            if layer_name and layer_name not in name:
                continue
            all_values.append(f.flatten().cpu().numpy())

        if not all_values:
            return None

        import numpy as np
        flat = np.concatenate(all_values)

        # Normalize 0-1
        flat = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)

        # Return top_k values
        if len(flat) > top_k:
            indices = np.linspace(0, len(flat) - 1, top_k).astype(int)
            flat = flat[indices]

        return flat

    def get_memory_summary(self):
        """Return summary of all remembered tasks"""
        summary = []
        for i, mem in enumerate(self.task_memory):
            all_f = torch.cat([f.flatten() for f in mem['fisher'].values()])
            summary.append({
                'task': i + 1,
                'name': mem['task_name'],
                'mean_fisher': float(all_f.mean()),
                'max_fisher': float(all_f.max()),
                'protected_params': int((all_f > all_f.mean()).sum())
            })
        return summary

    def save(self, path):
        """Save EWC state"""
        torch.save({
            'task_memory': self.task_memory,
            'lambda_ewc': self.lambda_ewc
        }, path)
        print(f"[EWC] Saved to {path}")

    def load(self, path):
        """Load EWC state"""
        state = torch.load(path, map_location=self.device)
        self.task_memory = state['task_memory']
        self.lambda_ewc = state['lambda_ewc']
        print(f"[EWC] Loaded {len(self.task_memory)} task memories from {path}")


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')

    from models.backbone import get_model
    from torch.utils.data import DataLoader, TensorDataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(num_classes=43, device=device)

    # Fake dataloader for testing
    X = torch.randn(100, 3, 64, 64)
    y = torch.randint(0, 43, (100,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    ewc = EWC(model, device=device, lambda_ewc=5000)

    # Simulate Task 1 completion
    ewc.compute_fisher(loader, task_name='Task1_Sunny', n_samples=64)

    # Compute penalty
    pen = ewc.penalty()
    print(f"EWC Penalty: {pen.item():.4f}")

    # Fisher heatmap
    heatmap = ewc.get_fisher_heatmap()
    print(f"Heatmap shape: {heatmap.shape if heatmap is not None else None}")

    print("EWC OK ✓")
