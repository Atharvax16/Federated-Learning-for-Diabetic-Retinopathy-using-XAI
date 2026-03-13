"""Hospital client for Federated Learning simulation.

Each hospital has its own local dataset (simulated), trains a local model
on its data, and sends updated weights back to the central server.
"""

import copy
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from federated.common.model import build_resnet50, FocalLoss
from federated.common.config import NUM_CLASSES, IMG_SIZE


class HospitalClient:
    """Simulates a hospital participating in federated learning.

    Since we don't have the actual APTOS images at each hospital,
    this generates synthetic training data that mimics the statistical
    properties of retinal images for realistic simulation.
    """

    def __init__(
        self,
        hospital_id: str,
        hospital_name: str,
        num_samples: int,
        class_distribution: Optional[List[int]] = None,
    ):
        self.hospital_id = hospital_id
        self.hospital_name = hospital_name
        self.num_samples = num_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Generate class distribution if not provided
        if class_distribution is None:
            # Default skewed distribution mimicking real-world DR prevalence
            probs = np.array([0.49, 0.10, 0.28, 0.05, 0.08])
            class_distribution = np.random.multinomial(num_samples, probs).tolist()

        self.class_distribution = class_distribution
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate synthetic data that mimics retinal image features.

        We create synthetic feature tensors (not raw images) to simulate
        what the model would see. In production, each hospital would
        use their own real patient images.
        """
        np.random.seed(hash(self.hospital_id) % (2**31))
        torch.manual_seed(hash(self.hospital_id) % (2**31))

        all_images = []
        all_labels = []

        for cls_idx, count in enumerate(self.class_distribution):
            if count <= 0:
                continue
            # Synthetic 224x224x3 images with class-specific patterns
            # Different severity levels have different statistical properties
            base_brightness = 0.5 - cls_idx * 0.05
            noise_level = 0.1 + cls_idx * 0.03

            images = torch.randn(count, 3, IMG_SIZE, IMG_SIZE) * noise_level + base_brightness
            # Add class-specific spatial patterns (simulating lesions)
            if cls_idx >= 2:
                # More severe = more/larger "lesion" spots
                num_spots = cls_idx * 3
                for i in range(count):
                    for _ in range(num_spots):
                        cx = np.random.randint(20, IMG_SIZE - 20)
                        cy = np.random.randint(20, IMG_SIZE - 20)
                        r = np.random.randint(3, 8 + cls_idx * 2)
                        y_coords, x_coords = torch.meshgrid(
                            torch.arange(IMG_SIZE), torch.arange(IMG_SIZE), indexing="ij"
                        )
                        mask = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) < r**2
                        images[i, 0, mask] -= 0.3  # dark spots in red channel

            # Normalize to ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = (images - mean) / std

            labels = torch.full((count,), cls_idx, dtype=torch.long)
            all_images.append(images)
            all_labels.append(labels)

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Shuffle
        perm = torch.randperm(len(all_labels))
        all_images = all_images[perm]
        all_labels = all_labels[perm]

        # 80/20 train/val split
        split = int(0.8 * len(all_labels))
        self.train_images = all_images[:split]
        self.train_labels = all_labels[:split]
        self.val_images = all_images[split:]
        self.val_labels = all_labels[split:]

    def local_train(
        self,
        global_weights: OrderedDict,
        local_epochs: int = 2,
        local_lr: float = 1e-4,
        batch_size: int = 16,
        loss_type: str = "focal",
        focal_gamma: float = 2.0,
    ) -> Dict:
        """Train the model locally using this hospital's data.

        Args:
            global_weights: Current global model weights from the server.
            local_epochs: Number of local training epochs.
            local_lr: Local learning rate.
            batch_size: Local batch size.
            loss_type: Loss function type ('ce', 'weighted_ce', 'focal').
            focal_gamma: Gamma parameter for focal loss.

        Returns:
            Dict with 'weights', 'train_loss', 'val_acc', 'val_auc',
            'val_f1', 'per_class_recall'.
        """
        # Build local model and load global weights
        model = build_resnet50()
        model.load_state_dict(copy.deepcopy(global_weights))
        model.to(self.device)

        # Compute class weights for this hospital's distribution
        counts = np.array(self.class_distribution, dtype=np.float32)
        total = counts.sum()
        weights = total / (NUM_CLASSES * np.maximum(counts, 1))
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Set up loss
        if loss_type == "focal":
            criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        elif loss_type == "weighted_ce":
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=local_lr, weight_decay=1e-4)

        # Data loaders
        train_ds = TensorDataset(self.train_images, self.train_labels)
        val_ds = TensorDataset(self.val_images, self.val_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

        # Local training
        total_loss = 0.0
        num_batches = 0
        model.train()
        for epoch in range(local_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.append(probs)
                all_preds.append(preds)
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        acc = float(np.mean(all_preds == all_labels))

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(all_labels, all_probs, multi_class="ovr"))
        except Exception:
            auc = None

        # F1
        from sklearn.metrics import f1_score, recall_score
        f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        recalls = recall_score(
            all_labels, all_preds, average=None,
            labels=list(range(NUM_CLASSES)), zero_division=0
        ).tolist()

        # Return updated weights and metrics
        updated_weights = copy.deepcopy(model.state_dict())
        # Move weights to CPU for aggregation
        for k in updated_weights:
            updated_weights[k] = updated_weights[k].cpu()

        return {
            "weights": updated_weights,
            "train_loss": avg_train_loss,
            "val_acc": acc,
            "val_auc": auc,
            "val_f1": f1,
            "per_class_recall": recalls,
        }
