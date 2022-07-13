import dataclasses
import itertools
from typing import Optional, List, Dict

from utils.dataset_utils import ROI


@dataclasses.dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclasses.dataclass
class GlyphItem:
    value: str
    box: Rect


@dataclasses.dataclass
class GroundTruthPathsItem:
    """
    Ground truth value
    """
    str_value: str

    """
    Unique image name
    """
    img_name: str

    """
    Full path to rendered value
    """
    img_path: str

    """
    Region of interest
    """
    roi: Optional[ROI] = None


@dataclasses.dataclass
class GroundTruthItem:
    str_value: str
    img_path: str
    box_values: List[GlyphItem] = None


@dataclasses.dataclass
class Word:
    word_id: str
    text: str
    glyphs: List[Rect]

    def offset_x(self, x):
        for g in self.glyphs:
            g.x += x

    def get_xy(self):
        min_y = list(itertools.accumulate(
            self.glyphs, lambda s, g: update_min(s, g.y),
            initial=None
        ))[-1]
        return self.glyphs[0].x, min_y

    def get_width(self):
        return list(itertools.accumulate(self.glyphs, lambda s, g: s + g.width))[-1]

    def get_height(self):
        x, y = self.get_xy()
        max_y = list(itertools.accumulate(
            self.glyphs, lambda s, g: update_max(s, g.y),
            initial=None
        ))[-1]
        return max_y - y

    def get_rect(self):
        return Rect(*self.get_xy(), self.get_width(), self.get_height())


@dataclasses.dataclass
class Line:
    text: str
    words: Dict[str, Word]


def update_min(existing, candidate):
    if existing is None:
        return candidate
    return min(existing, candidate)


def update_max(existing, candidate):
    if existing is None:
        return candidate
    return max(existing, candidate)
