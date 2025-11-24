"""
Data loading and preprocessing for sketch datasets.

Supports:
- TU-Berlin sketch dataset
- QuickDraw dataset
- Custom sketch datasets
- Few-shot episode sampling
"""

from .datasets import TUBerlinDataset, QuickDrawDataset, SketchDataset
from .samplers import FewShotSampler, EpisodeSampler
from .transforms import get_sketch_transforms
from .download import download_tuberlin, download_quickdraw

__all__ = [
    'TUBerlinDataset',
    'QuickDrawDataset',
    'SketchDataset',
    'FewShotSampler',
    'EpisodeSampler',
    'get_sketch_transforms',
    'download_tuberlin',
    'download_quickdraw'
]

