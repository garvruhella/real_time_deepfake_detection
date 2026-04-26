import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms

from src.model_discriminator import DCGANDiscriminator128


IMAGE_SIZE = 128
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def build_detector_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1),
    )
    return model


def build_discriminator_model() -> nn.Module:
    return DCGANDiscriminator128()


def _load_state(model: nn.Module, state_path: str, device: torch.device) -> nn.Module:
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_hybrid_models(
    detector_path: str,
    discriminator_path: str,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    detector = _load_state(build_detector_model(), detector_path, device)
    discriminator = _load_state(build_discriminator_model(), discriminator_path, device)
    return detector, discriminator


def compute_hybrid_scores(
    images: torch.Tensor,
    detector: nn.Module,
    discriminator: nn.Module,
) -> Dict[str, torch.Tensor]:
    detector_logit = detector(images).view(-1)
    detector_prob = torch.sigmoid(detector_logit)

    # D outputs "realness", so invert to get fake probability.
    disc_real_prob = discriminator(images).view(-1)
    disc_fake_prob = 1.0 - disc_real_prob

    return {
        "detector_logit": detector_logit,
        "detector_fake_prob": detector_prob,
        "disc_fake_prob": disc_fake_prob,
    }


def save_hybrid_config(
    output_path: str,
    detector_weight: float,
    discriminator_weight: float,
    threshold: float,
) -> None:
    payload = {
        "version": 1,
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
        "fusion": {
            "detector_weight": detector_weight,
            "discriminator_weight": discriminator_weight,
            "threshold": threshold,
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
