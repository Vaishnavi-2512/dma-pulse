import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dma_pulse.model import PulseTransformer

def test_binary_forward_shape():
    model = PulseTransformer(num_classes=2); y = model(torch.randn(4, 5, 11)); assert y.shape == (4, 2)

def test_multiclass_forward_shape():
    model = PulseTransformer(num_classes=4); y = model(torch.randn(3, 5, 11)); assert y.shape == (3, 4)
