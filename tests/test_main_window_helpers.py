from __future__ import annotations

import pytest

pytest.importorskip("PySide6")  # this module needs a real Qt install; CI's lightweight venv skips it

from pose_craft.gui.main_window import _nearest_person_index  # noqa: E402


class _StubPerson:
    def __init__(self, x: float, y: float):
        self._xy = (x, y)

    def centroid_xy(self) -> tuple[float, float]:
        return self._xy


def test_nearest_person_index_picks_closest_centroid():
    people = [_StubPerson(0.1, 0.5), _StubPerson(0.9, 0.5), _StubPerson(0.5, 0.1)]

    assert _nearest_person_index(people, 0.12, 0.48) == 0
    assert _nearest_person_index(people, 0.85, 0.5) == 1
    assert _nearest_person_index(people, 0.5, 0.05) == 2
