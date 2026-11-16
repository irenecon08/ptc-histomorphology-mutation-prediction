"""Tests for the tiling helpers that do not need OpenSlide."""

import numpy as np

from ptc_mil.preprocessing.tiling import is_background, level0_location


def test_white_tile_is_background() -> None:
    assert is_background(np.full((256, 256, 3), 240, dtype=np.uint8))


def test_tissue_tile_is_not_background() -> None:
    tile = np.full((256, 256, 3), 240, dtype=np.uint8)
    tile[:, :64] = 120  # 25% tissue -> 75% white, below the 85% cut-off
    assert not is_background(tile)


def test_level0_uses_actual_downsample_not_power_of_two() -> None:
    assert level0_location(256, 512, 4.0002) == (1024, 2048)
    assert level0_location(1000, 0, 4.0002) == (4000, 0)
    assert level0_location(10000, 0, 4.0002) == (40002, 0)
