from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional


@dataclass
class Match:
    id: int
    length: int
    occurrences: List[int]
    similarity: float
    bars: List[int]


@dataclass
class LargeMatch:
    id: int
    start_bar_a: int
    start_bar_b: int
    length_bars: int
    avg_similarity: float


@dataclass
class TimeSignature:
    numerator: int
    denominator: int
    ticks_per_bar: int
