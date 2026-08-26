"""
AdaptSign -- Dharma Confidence Gate
IKS Principle: Dharma -- Right action, ethical responsibility
"""

import torch
import torch.nn.functional as F


class DharmaGate:
    """
    IKS: Dharma - Right Action
    If model is uncertain, it should NOT guess.
    Better to say "I don't know" than to misclassify a Stop sign.
    """

    def __init__(self, high_threshold=0.92, low_threshold=0.75):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.total_predictions = 0
        self.high_conf = 0
        self.medium_conf = 0
        self.abstentions = 0
        print(f"[DHARMA] Gate initialized | High>{high_threshold:.0%} | Abstain<{low_threshold:.0%}")

    def predict(self, model, inputs, class_names=None, device='cpu'):
        model.eval()
        inputs = inputs.to(device)

        with torch.no_grad():
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = probs.max(dim=1)

        results = []
        for i in range(len(inputs)):
            conf = confidences[i].item()
            pred = predictions[i].item()
            self.total_predictions += 1

            name = class_names[pred] if (class_names and pred < len(class_names)) else f'Class {pred}'

            if conf >= self.high_threshold:
                self.high_conf += 1
                status = 'SAFE'
                message = f'OK {name} -- {conf:.1%} confident'
                dharma_msg = 'Right action taken with certainty'
            elif conf >= self.low_threshold:
                self.medium_conf += 1
                status = 'CAUTION'
                message = f'WARN {name} -- {conf:.1%} (verify recommended)'
                dharma_msg = 'Acting with caution -- awareness of uncertainty'
            else:
                self.abstentions += 1
                status = 'ABSTAIN'
                pred = None
                name = 'UNCERTAIN'
                message = f'ABSTAIN ({conf:.1%}) -- human verification needed'
                dharma_msg = 'Dharma: Better to abstain than to cause harm'

            top3_probs, top3_ids = probs[i].topk(3)
            results.append({
                'class_id': pred,
                'class_name': name,
                'confidence': conf,
                'status': status,
                'message': message,
                'dharma': dharma_msg,
                'top3': [
                    {
                        'class_id': top3_ids[j].item(),
                        'class_name': class_names[top3_ids[j].item()] if class_names else f'Class {top3_ids[j].item()}',
                        'confidence': top3_probs[j].item()
                    } for j in range(3)
                ]
            })

        return results

    def get_stats(self):
        total = max(1, self.total_predictions)
        return {
            'total': self.total_predictions,
            'safe_rate': self.high_conf / total,
            'caution_rate': self.medium_conf / total,
            'abstention_rate': self.abstentions / total,
        }
