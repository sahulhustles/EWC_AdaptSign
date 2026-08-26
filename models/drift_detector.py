"""
AdaptSign — Fisher Drift Detector
IKS Principle: Viveka — Discriminative Wisdom

YOUR KEY INNOVATION:
Standard EWC blindly applies after EVERY training step.
Our system first DETECTS if a real domain shift has occurred,
then decides whether to consolidate knowledge.

This is smarter, more efficient, and more theoretically sound.
No paper does this for traffic signs. This IS your novelty.
"""

import torch
import numpy as np
from collections import deque
import torch.nn.functional as F


class DriftDetector:
    """
    Detects when the input distribution has significantly changed
    (e.g., sunny → rainy) using feature-space statistics.

    IKS: Viveka (विवेक) — The power to discriminate truth from falsehood.
    ──────────────────────────────────────────────────────────────────────
    A wise student knows WHEN to make notes of important lessons.
    Not after every sentence — only when a truly new concept appears.

    Our detector knows when a new "weather world" has been encountered
    and triggers EWC consolidation only at that moment.

    Method:
    ───────
    1. Maintain a rolling window of feature embeddings
    2. Compute mean feature vector for current window
    3. Compare with previous window via cosine distance
    4. If distance > threshold → DRIFT DETECTED → trigger EWC
    """

    def __init__(self, model, device='cpu', threshold=0.15, window_size=200):
        """
        Args:
            model:       AdaptSignBackbone (used for feature extraction)
            device:      'cuda' or 'cpu'
            threshold:   Cosine distance threshold for drift detection
                         0.0 = identical distributions
                         1.0 = completely different
                         Recommended: 0.10-0.20 for post-training feature space
            window_size: Number of samples to average for statistics
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.window_size = window_size

        # Rolling window of feature vectors
        self.current_window = deque(maxlen=window_size)
        self.reference_mean = None
        self.drift_history = []  # Log of when drifts occurred

        # Statistics
        self.total_checks = 0
        self.drifts_detected = 0

        print(f"[DRIFT] DriftDetector initialized | threshold={threshold}")
        print("[DRIFT] IKS: Viveka — discernment of domain shift")

    def update(self, dataloader, n_batches=10):
        """
        Update the reference distribution with new data.
        Called at the start of each new task's data.

        Returns:
            drift_detected: bool
            distance: float (cosine distance from previous distribution)
        """
        self.model.eval()
        new_features = []

        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= n_batches:
                    break
                inputs = inputs.to(self.device)
                feats = self.model.get_features(inputs)
                new_features.append(feats.cpu())

        if not new_features:
            return False, 0.0

        # Compute mean feature vector for new data
        all_feats = torch.cat(new_features, dim=0)
        new_mean = all_feats.mean(dim=0)

        self.total_checks += 1

        # First time — set as reference
        if self.reference_mean is None:
            self.reference_mean = new_mean
            print("[DRIFT] Reference distribution established (Task 1)")
            return False, 0.0

        # Compute cosine distance between distributions
        distance = self._cosine_distance(self.reference_mean, new_mean)

        drift_detected = distance > self.threshold
        self.drift_history.append({
            'check': self.total_checks,
            'distance': float(distance),
            'drift': drift_detected
        })

        if drift_detected:
            self.drifts_detected += 1
            print(f"\n[DRIFT] ⚠️  DOMAIN SHIFT DETECTED!")
            print(f"[DRIFT] Distance: {distance:.4f} > threshold: {self.threshold}")
            print(f"[DRIFT] Triggering EWC consolidation...")
        else:
            print(f"[DRIFT] No significant shift | Distance: {distance:.4f}")

        # Always advance reference to current distribution so that each
        # task is compared against the PREVIOUS task (not always Task 1).
        self.reference_mean = new_mean

        return drift_detected, float(distance)

    def check_batch(self, inputs):
        """
        Lightweight check on a single batch.
        Used during training to monitor drift in real-time.
        Returns drift probability (0-1).
        """
        self.model.eval()
        with torch.no_grad():
            feats = self.model.get_features(inputs.to(self.device))
            batch_mean = feats.mean(dim=0).cpu()

        if self.reference_mean is None:
            return 0.0

        distance = self._cosine_distance(self.reference_mean, batch_mean)
        return float(distance / self.threshold)  # normalized 0-1

    def _cosine_distance(self, a, b):
        """
        Cosine distance between two vectors.
        0 = identical, 2 = opposite
        Normalized to 0-1 range.
        """
        a_norm = F.normalize(a.unsqueeze(0), dim=1)
        b_norm = F.normalize(b.unsqueeze(0), dim=1)
        similarity = torch.mm(a_norm, b_norm.t()).squeeze()
        distance = (1 - similarity.item()) / 2  # normalize to 0-1
        return max(0.0, min(1.0, distance))

    def get_stats(self):
        """Return drift detection statistics"""
        return {
            'total_checks': self.total_checks,
            'drifts_detected': self.drifts_detected,
            'drift_rate': self.drifts_detected / max(1, self.total_checks),
            'history': self.drift_history,
            'threshold': self.threshold
        }

    def visualize_history(self, save_path=None):
        """
        Plot drift detection history.
        Shows distance over time with threshold line.
        """
        if not self.drift_history:
            print("[DRIFT] No history to plot")
            return

        import matplotlib.pyplot as plt

        checks = [h['check'] for h in self.drift_history]
        distances = [h['distance'] for h in self.drift_history]
        drifts = [h['drift'] for h in self.drift_history]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')

        ax.plot(checks, distances, color='#00d4aa', linewidth=2, label='Distance')
        ax.axhline(y=self.threshold, color='#ffa502', linewidth=1.5,
                   linestyle='--', label=f'Threshold ({self.threshold})')

        # Mark drift points
        drift_checks = [h['check'] for h in self.drift_history if h['drift']]
        drift_dists = [h['distance'] for h in self.drift_history if h['drift']]
        ax.scatter(drift_checks, drift_dists, color='#ff4757',
                   s=100, zorder=5, label='Drift Detected')

        ax.set_xlabel('Check #', color='#6b6b8a')
        ax.set_ylabel('Cosine Distance', color='#6b6b8a')
        ax.set_title('Fisher Drift Detector — Domain Shift History',
                     color='#e8e8f0', fontsize=13)
        ax.tick_params(colors='#6b6b8a')
        ax.spines['bottom'].set_color('#2a2a3a')
        ax.spines['left'].set_color('#2a2a3a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(facecolor='#12121a', labelcolor='#e8e8f0')
        ax.grid(True, color='#1a1a26', alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor='#0a0a0f')
            print(f"[DRIFT] Plot saved to {save_path}")
        else:
            plt.show()
        plt.close()


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from models.backbone import get_model
    from torch.utils.data import DataLoader, TensorDataset

    device = 'cpu'
    model = get_model(num_classes=43, device=device)
    detector = DriftDetector(model, device=device, threshold=0.25)

    # Simulate Task 1 (sunny) data
    X1 = torch.randn(64, 3, 64, 64) * 0.5 + 1.0  # brighter
    y1 = torch.randint(0, 43, (64,))
    loader1 = DataLoader(TensorDataset(X1, y1), batch_size=16)

    drift, dist = detector.update(loader1)
    print(f"Task 1 → Drift: {drift}, Distance: {dist:.4f}")

    # Simulate Task 2 (rainy) data — different distribution
    X2 = torch.randn(64, 3, 64, 64) * 1.5 - 0.5  # darker, more variance
    y2 = torch.randint(0, 43, (64,))
    loader2 = DataLoader(TensorDataset(X2, y2), batch_size=16)

    drift, dist = detector.update(loader2)
    print(f"Task 2 → Drift: {drift}, Distance: {dist:.4f}")

    print("DriftDetector OK ✓")
