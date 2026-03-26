"""
Tests for the transitional paste candidate session.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.paste_candidate import PasteCandidateSession  # noqa: E402


def test_paste_candidate_session_clear_resets_everything() -> None:
    session = PasteCandidateSession(
        asset_path="/tmp/asset.png",
        pasteimg="pasteimg",
        origin_pasteimg="origin",
        paste_canvas="thumb",
        pasteimg_width=12,
        pasteimg_height=34,
        pasteimg_channel=4,
        resizeimg="resize",
        rotated="rotated",
        bc_image="bc",
        norm_pimg=["norm"],
        bbox_pimg=[1, 2, 3, 4, 10, 10],
        real_bbox_pimg=[10, 20, 30, 40],
        pasteimg_canvas="preview",
        anchor_x=100,
        anchor_y=200,
    )

    session.clear()

    assert session.asset_path == ""
    assert session.pasteimg is None
    assert session.origin_pasteimg is None
    assert session.paste_canvas is None
    assert session.pasteimg_width == 0
    assert session.pasteimg_height == 0
    assert session.pasteimg_channel == 0
    assert session.resizeimg is None
    assert session.rotated is None
    assert session.bc_image is None
    assert session.norm_pimg is None
    assert session.bbox_pimg is None
    assert session.real_bbox_pimg is None
    assert session.pasteimg_canvas is None
    assert session.anchor_x is None
    assert session.anchor_y is None


def test_paste_candidate_session_clear_candidate_preserves_asset_state() -> None:
    session = PasteCandidateSession(
        asset_path="/tmp/asset.png",
        pasteimg="pasteimg",
        origin_pasteimg="origin",
        paste_canvas="thumb",
        pasteimg_width=12,
        pasteimg_height=34,
        pasteimg_channel=4,
        norm_pimg=["norm"],
        bbox_pimg=[1, 2, 3, 4, 10, 10],
        real_bbox_pimg=[10, 20, 30, 40],
        pasteimg_canvas="preview",
        anchor_x=100,
        anchor_y=200,
    )

    session.clear_candidate()

    assert session.asset_path == "/tmp/asset.png"
    assert session.pasteimg == "pasteimg"
    assert session.origin_pasteimg == "origin"
    assert session.paste_canvas == "thumb"
    assert session.pasteimg_width == 12
    assert session.pasteimg_height == 34
    assert session.pasteimg_channel == 4
    assert session.norm_pimg is None
    assert session.bbox_pimg is None
    assert session.real_bbox_pimg is None
    assert session.pasteimg_canvas is None
    assert session.anchor_x is None
    assert session.anchor_y is None


def test_paste_candidate_session_tracks_anchor_presence() -> None:
    session = PasteCandidateSession()

    assert session.has_anchor is False

    session.set_anchor(12, 34)

    assert session.has_anchor is True
    assert session.anchor_x == 12
    assert session.anchor_y == 34
