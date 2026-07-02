"""Tests for anytrack.roi ordering helpers."""
from __future__ import annotations

from anytrack.roi import _reading_order


def test_reading_order_2x2_with_jitter():
    # A 2x2 grid where the two arenas in each row have slightly different cy.
    # A naive (cy, cx) sort would order by cy first and swap left/right within
    # a row; reading order must group rows then go left->right.
    circles = [
        (300.0, 90.0, 40.0),   # top-right   (cy 90)
        (100.0, 100.0, 40.0),  # top-left    (cy 100)
        (300.0, 290.0, 40.0),  # bottom-right (cy 290)
        (100.0, 300.0, 40.0),  # bottom-left  (cy 300)
    ]
    ordered = _reading_order(circles)
    cxcy = [(c[0], c[1]) for c in ordered]
    assert cxcy == [(100.0, 100.0), (300.0, 90.0), (100.0, 300.0), (300.0, 290.0)]


def test_reading_order_single_row():
    circles = [(300.0, 100.0, 30.0), (100.0, 100.0, 30.0), (200.0, 100.0, 30.0)]
    ordered = _reading_order(circles)
    assert [c[0] for c in ordered] == [100.0, 200.0, 300.0]


def test_reading_order_empty():
    assert _reading_order([]) == []
