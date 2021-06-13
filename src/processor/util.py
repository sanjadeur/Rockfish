from collections import namedtuple
from dataclasses import dataclass
import numpy as np

from typing import Tuple, Set, List, Dict

Interval = namedtuple('Interval', ['start', 'end'])  # Start - inclusive, end - exclusive

BEDPos = Dict[str, Tuple[Set[int], Set[int]]]  # Included positions - {chromosome: (fwd_pos, rev_pos)}


@dataclass
class ResegmentationData:
    position: int
    event_intervals: List[Interval]  # Intervals of signal points
    event_lens: np.ndarray  # Lengths of intervals of signal points
    bases: str
