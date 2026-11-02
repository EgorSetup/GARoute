"""Проверки взаимоисключающих режимов и границ учебного сегмента."""

from interface.controls import UIState
from interface.map_view import handle_map_click, handle_segment_click, set_map_mode


def test_modes_do_not_overlap() -> None:
    state = UIState(demo_mode=True)
    assert set_map_mode(state, "A")
    assert handle_map_click(state, (0.0, -4.0))
    assert state.demo_point_a == (0.0, -4.0)
    assert set_map_mode(state, "traffic")
    assert not handle_map_click(state, (0.0, 4.0))
    assert handle_segment_click(state, "demo", "increase")
    assert state.demo_point_b is None
    assert state.demo_point_a == (0.0, -4.0)


def test_segment_boundaries_and_single_step() -> None:
    state = UIState(demo_mode=True)
    assert set_map_mode(state, "traffic")
    assert handle_segment_click(state, "demo", "decrease")
    assert state.demo_segment_level == 1
    assert state.demo_segment_clicks == 1
    assert handle_segment_click(state, "demo", "increase")
    assert state.demo_segment_level == 2
    for _ in range(20):
        assert handle_segment_click(state, "demo", "increase")
    assert state.demo_segment_level == 11
    assert handle_segment_click(state, "demo", "decrease")
    assert state.demo_segment_level == 10
    assert not handle_segment_click(state, "other", "increase")


def test_normal_map_has_no_demo_segment() -> None:
    state = UIState()
    assert not handle_segment_click(state, "demo", "increase")
    assert state.demo_segment_clicks == 0
