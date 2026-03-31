# Copyright (c) 2026. All rights reserved.
#
# This source code is licensed under the Restricted Academic License.
# Commercial use, redistribution, or unauthorized execution is strictly prohibited.
# The core algorithms are patent-pending.

from .mdf_net import MDFNet
from .modules import DualStreamEncoder, BDNetDecoder
from .evidential import EvidentialFusion

__all__ = [
    'MDFNet',
    'DualStreamEncoder',
    'BDNetDecoder',
    'EvidentialFusion'
]

__version__ = '2.1.0-restricted'