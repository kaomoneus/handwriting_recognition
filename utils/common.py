import dataclasses
import itertools
from typing import Optional, List, Dict, Tuple


@dataclasses.dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int

    def to_vec(self):
        return self.x, self.y, self.width, self.height


@dataclasses.dataclass
class Point:
    x: int
    y: int

    def to_vec(self):
        return self.x, self.y


@dataclasses.dataclass
class GlyphItem:
    value: str
    box: Rect


ROI = Tuple[int, int, int, int]


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

    def get_top(self):
        return min(self.glyphs, key=lambda g: g.y).y

    def get_left(self):
        return min(self.glyphs, key=lambda g: g.x).x

    def get_right(self):
        max_x_r = max(self.glyphs, key=lambda g: g.x + g.width)
        return max_x_r.x + max_x_r.width

    def get_bottom(self):
        max_y_r = max(self.glyphs, key=lambda g: g.y + g.height)
        return max_y_r.y + max_y_r.height

    def get_rect(self):
        left, top, right, bottom = \
            self.get_left(), self.get_top(), self.get_right(), self.get_bottom()

        return Rect(left, top, right-left, bottom-top)


@dataclasses.dataclass
class Line:
    text: str
    words: List[Word]


Dataset = List[GroundTruthPathsItem]
GroundTruth = Dict[str, GroundTruthPathsItem]