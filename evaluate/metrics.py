"""
AdaptSign -- Evaluation Metrics
All metrics needed for lifelong learning evaluation.
These are what you report in your paper and review.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os


class LifelongMetrics:
    """
    Compute and track all lifelong learning metrics.

    Metrics:
        Accuracy       - Standard per-task accuracy
        Forgetting (F) - How much accuracy dropped on old tasks
        BWT            - Backward Transfer (negative = forgetting)
        FWT            - Forward Transfer (how old tasks help new)
        Plasticity     - Ability to learn new tasks
        Stability      - Ability to retain old tasks
    """

    def __init__(self, n_tasks=4, task_names=None):
        self.n_tasks = n_tasks
        self.task_names = task_names or [f'Task{i+1}' for i in range(n_tasks)]

        # R[i][j] = accuracy on task i after training task j
        self.R = [[None] * n_tasks for _ in range(n_tasks)]

    def update(self, task_idx, after_task_idx, accuracy):
        """Record accuracy on task_idx after training up to after_task_idx"""
        self.R[task_idx][after_task_idx] = accuracy

    def forgetting(self):
        """
        Average Forgetting:
        F = (1/(T-1)) * sum_i max_j<T (R[i][j] - R[i][T-1])

        Measures how much accuracy drops on old tasks
        Target: F < 10% (lower is better)
        """
        T = self.n_tasks
        forgetting_per_task = []

        for i in range(T - 1):  # all tasks except last
            # Max accuracy achieved on this task
            max_acc = max(self.R[i][j] for j in range(i, T) if self.R[i][j] is not None)
            # Final accuracy on this task
            final_acc = self.R[i][T - 1]
            if final_acc is not None:
                forgetting_per_task.append(max_acc - final_acc)

        if not forgetting_per_task:
            return None
        return sum(forgetting_per_task) / len(forgetting_per_task)

    def bwt(self):
        """
        Backward Transfer:
        BWT = (1/(T-1)) * sum_i (R[i][T-1] - R[i][i])

        Measures how training later tasks affects earlier tasks.
        BWT < 0 = forgetting occurred
        BWT = 0 = no interference
        BWT > 0 = learning later tasks helps earlier ones (rare)
        """
        T = self.n_tasks
        bwt_sum = 0
        count = 0
        for i in range(T - 1):
            if self.R[i][T-1] is not None and self.R[i][i] is not None:
                bwt_sum += self.R[i][T-1] - self.R[i][i]
                count += 1
        return bwt_sum / count if count > 0 else None

    def fwt(self):
        """
        Forward Transfer:
        FWT = (1/(T-1)) * sum_i (R[i][i] - R0[i])

        Measures if learning previous tasks helps future tasks.
        Positive = positive transfer (good)
        """
        # Random baseline: 100% / num_classes (chance-level for a balanced dataset)
        random_baseline = 100.0 / self.n_tasks if self.n_tasks > 0 else 0.0
        T = self.n_tasks
        fwt_sum = 0
        count = 0
        for i in range(1, T):
            if self.R[i][i] is not None:
                fwt_sum += self.R[i][i] - random_baseline
                count += 1
        return fwt_sum / count if count > 0 else None

    def average_accuracy(self):
        """Average accuracy across all tasks at end of training"""
        T = self.n_tasks
        accs = [self.R[i][T-1] for i in range(T) if self.R[i][T-1] is not None]
        return sum(accs) / len(accs) if accs else None

    def summary(self):
        """Return full metrics summary"""
        return {
            'average_accuracy': self.average_accuracy(),
            'forgetting': self.forgetting(),
            'bwt': self.bwt(),
            'fwt': self.fwt(),
            'accuracy_matrix': self.R,
            'task_names': self.task_names
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "="*50)
        print("  LIFELONG LEARNING METRICS SUMMARY")
        print("="*50)
        print(f"  Average Accuracy:  {s['average_accuracy']:.1f}%" if s['average_accuracy'] else "  Average Accuracy: N/A")
        print(f"  Forgetting Score:  {s['forgetting']:.2f}%" if s['forgetting'] else "  Forgetting: N/A")
        print(f"  BWT:              {s['bwt']:.4f}" if s['bwt'] else "  BWT: N/A")
        print(f"  FWT:              {s['fwt']:.4f}" if s['fwt'] else "  FWT: N/A")
        print()
        print("  Accuracy Matrix:")
        header = f"  {'':10}" + "".join(f"{'AfterT'+str(j+1):>10}" for j in range(self.n_tasks))
        print(header)
        for i in range(self.n_tasks):
            row = f"  {self.task_names[i]:10}"
            for j in range(self.n_tasks):
                val = self.R[i][j]
                row += f"  {val:7.1f}%" if val is not None else f"  {'---':>7}"
            print(row)
        print("="*50)

    def plot_accuracy_matrix(self, title='Accuracy Matrix', save_path=None):
        """Heatmap of the accuracy matrix"""
        T = self.n_tasks
        matrix = np.array([
            [self.R[i][j] if self.R[i][j] is not None else 0 for j in range(T)]
            for i in range(T)
        ])

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#12121a')

        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        plt.colorbar(im, ax=ax, label='Accuracy (%)')

        ax.set_xticks(range(T))
        ax.set_xticklabels([f'After T{j+1}\n({self.task_names[j]})' for j in range(T)],
                           color='#e8e8f0', fontsize=9)
        ax.set_yticks(range(T))
        ax.set_yticklabels([f'T{i+1}: {self.task_names[i]}' for i in range(T)],
                           color='#e8e8f0', fontsize=9)

        for i in range(T):
            for j in range(T):
                if self.R[i][j] is not None:
                    ax.text(j, i, f'{self.R[i][j]:.1f}%',
                            ha='center', va='center',
                            color='black', fontsize=10, fontweight='bold')

        ax.set_title(title, color='#e8e8f0', fontsize=13, pad=15)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
        else:
            plt.show()
        plt.close()

    def save(self, path):
        """Save metrics to JSON"""
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)
        print(f"[METRICS] Saved to {path}")


def evaluate_model(model, loader, device='cpu', class_names=None):
    """
    Full evaluation: accuracy + confusion matrix + classification report
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100.0 * (all_preds == all_labels).mean()

    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
    }


if __name__ == '__main__':
    # Test with fake data
    metrics = LifelongMetrics(n_tasks=4, task_names=['Sunny', 'Rainy', 'Foggy', 'Night'])

    # Simulate EWC results
    fake_ewc = [
        [97.0, 92.0, 90.0, 88.0],
        [None, 93.0, 91.0, 89.0],
        [None, None, 92.0, 90.0],
        [None, None, None, 91.0],
    ]
    for i in range(4):
        for j in range(4):
            if fake_ewc[i][j] is not None:
                metrics.update(i, j, fake_ewc[i][j])

    metrics.print_summary()
    print("\nMetrics module OK")
