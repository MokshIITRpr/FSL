"""
Models package for Few-Shot Sketch Recognition Framework.

This package contains implementations of:
- Backbone encoders (ResNet, CNN)
- Self-supervised learning methods (SimCLR, BYOL)
- Few-shot learning algorithms (Prototypical Networks, Matching Networks)
- Supervised baselines
"""

from .backbone import SketchEncoder, ResNetEncoder
from .contrastive import SimCLR, BYOL
from .few_shot import PrototypicalNetwork, MatchingNetwork
from .supervised import SupervisedBaseline

__all__ = [
    'SketchEncoder',
    'ResNetEncoder',
    'SimCLR',
    'BYOL',
    'PrototypicalNetwork',
    'MatchingNetwork',
    'SupervisedBaseline'
]

