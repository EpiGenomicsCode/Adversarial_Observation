"""Smoke tests confirming APSO works as used in the Poison26 singularity container.
Mirrors the workflow in manuscripts/Poison26/bin/attack/poison_MNIST.py.
"""
import pytest
import torch
import torch.nn as nn
from Adversarial_Observation.Swarm import PSO

IMG_DIM = 1 * 28 * 28
N_PARTICLES = 10
TARGET_CLASS = 3


@pytest.fixture
def mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model.eval()
    return model


def adversarial_cost(model, position):
    """Maximize softmax confidence for TARGET_CLASS — mirrors poison_MNIST.py cost."""
    img = position.float().view(1, 1, 28, 28)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)
    return probs[0, TARGET_CLASS].item()


@pytest.fixture
def apso(mnist_model):
    torch.manual_seed(42)
    starting_positions = torch.rand(N_PARTICLES, IMG_DIM)
    return PSO(
        starting_positions=starting_positions,
        cost_func=adversarial_cost,
        model=mnist_model,
        w=0.8,
        c1=0.2,
        c2=1.5,
        minclamp=0.0,
        maxclamp=1.0,
    )


def test_apso_initialization(apso):
    assert len(apso.swarm) == N_PARTICLES
    assert apso.epoch == 0
    assert 0.0 <= apso.cos_best_g.item() <= 1.0
    assert apso.pos_best_g.shape == torch.Size([IMG_DIM])


def test_apso_step_advances_epoch(apso):
    apso.step()
    assert apso.epoch == 1


def test_apso_global_best_nondecreasing(apso):
    """Global best cost must never decrease — a core PSO invariant."""
    score_before = apso.cos_best_g.clone()
    apso.step()
    assert apso.cos_best_g >= score_before


def test_apso_run_epochs(apso):
    apso.run(epochs=3)
    assert apso.epoch == 3


def test_apso_best_position_clamped(apso):
    """Best position must stay within [0, 1] after optimization."""
    apso.run(epochs=3)
    best = apso.getBest()
    assert best.shape == torch.Size([IMG_DIM])
    assert best.min() >= 0.0
    assert best.max() <= 1.0


def test_apso_points_shape(apso):
    apso.step()
    points = apso.getPoints()
    assert points.shape == torch.Size([N_PARTICLES, IMG_DIM])


def test_captum_importable():
    """Captum is a required dep in apso_poison.def; confirm it's available for XAI."""
    captum = pytest.importorskip("captum")
    from captum.attr import Saliency, IntegratedGradients  # noqa: F401
