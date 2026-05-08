"""Tests for folder-based image navigation in the main window."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets  # noqa: E402
from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QPixmap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.main_window import MyWidget  # noqa: E402
from gui.dialogs.savelab_window import SavelabWindow  # noqa: E402
from sdde.prediction import PredictionRecord  # noqa: E402
from sdde.dataset_scan import FolderAnnotationScanResult  # noqa: E402
from sdde.error_analysis import ErrorCase, ERROR_FP  # noqa: E402
from sdde.error_analysis_scan import FolderErrorAnalysisResult  # noqa: E402
from sdde.class_catalog import ClassCatalog, ClassInfo  # noqa: E402
from sdde.classes_yaml import save_classes_yaml_path  # noqa: E402
from sdde.project_config import ProjectConfig  # noqa: E402
from sdde.prediction_review import initial_prediction_review_state, update_prediction_review_state  # noqa: E402
from sdde.prediction_review_report import PredictionReviewReport, PredictionReviewReportEntry  # noqa: E402
from sdde.prediction_review_store import has_prediction_review_session, save_prediction_review_session  # noqa: E402
from sdde.validation import DatasetValidationResult, ValidationIssue  # noqa: E402


APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeMouseEvent:
    def __init__(self, x: float, y: float) -> None:
        self._pos = QPointF(x, y)

    def position(self) -> QPointF:
        return self._pos

    def button(self):
        return Qt.MouseButton.LeftButton


def test_folder_navigation_ui_tracks_current_index() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    widget._set_folder_images(
        ["/tmp/a.jpg", "/tmp/b.png", "/tmp/c.tif"],
        current_path="/tmp/b.png",
    )

    assert widget._folder_image_index == 1
    assert widget.lbl_image_nav.text() == "Folder image 2 / 3"
    assert widget.action_prev_image.isEnabled() is True
    assert widget.action_next_image.isEnabled() is True

    widget.close()


def test_open_next_image_loads_adjacent_folder_item_without_prompt() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget._set_folder_images(
        ["/tmp/a.jpg", "/tmp/b.png", "/tmp/c.tif"],
        current_path="/tmp/a.jpg",
    )

    with patch.object(widget, "_load_image_file", return_value=True) as load_image:
        widget.open_next_image()

    load_image.assert_called_once_with("/tmp/b.png", ask_confirm=False)
    assert widget._folder_image_index == 1
    assert widget.lbl_image_nav.text() == "Folder image 2 / 3"

    widget.close()


def test_load_image_file_accepts_four_channel_input() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    rgba = np.zeros((12, 10, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255

    with patch("gui.main_window.cv2.imread", return_value=rgba):
        with patch.object(widget, "_check_autosave_recovery") as autosave:
            loaded = widget._load_image_file("/tmp/sample.tiff", ask_confirm=False)

    assert loaded is True
    assert widget.origin_width == 10
    assert widget.origin_height == 12
    assert widget.canvas is not None
    autosave.assert_called_once_with("/tmp/sample.tiff")

    widget.close()


def test_tile_direction_navigation_moves_spatially() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    origin = QPixmap(200, 200)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.origin_width = 200
    widget.origin_height = 200
    widget.ratio_value = 50
    widget._tile_panel.spin_size.setValue(100)
    widget._tile_panel.spin_stride.setValue(100)
    widget._tile_panel.chk_enabled.setChecked(True)
    APP.processEvents()

    assert widget._tile_panel.current_index() == 0
    widget._on_tile_step_requested(1, 0)
    assert widget._tile_panel.current_index() == 1
    widget._on_tile_step_requested(0, 1)
    assert widget._tile_panel.current_index() == 3
    widget._on_tile_step_requested(-1, 0)
    assert widget._tile_panel.current_index() == 2

    widget.close()


def test_tile_overview_click_selects_tile_and_exits_overview() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    origin = QPixmap(200, 200)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.origin_width = 200
    widget.origin_height = 200
    widget.ratio_value = 50
    widget._tile_panel.spin_size.setValue(100)
    widget._tile_panel.spin_stride.setValue(100)
    widget._tile_panel.chk_enabled.setChecked(True)
    APP.processEvents()

    widget._tile_panel.set_overview_enabled(True)
    APP.processEvents()
    assert widget._tile_overview_mode is True
    assert widget._tile_index_from_canvas_point(150, 50) == 1

    handled = widget._handle_tile_overview_press(_FakeMouseEvent(150, 50))

    assert handled is True
    assert widget._tile_panel.current_index() == 1
    assert widget._tile_panel.overview_enabled() is False
    assert widget._tile_overview_mode is False

    widget.close()


def test_load_label_routes_json_files_to_unified_json_importer() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.origin_width = 640
    widget.origin_height = 480
    widget.imgfilePath = "/tmp/sample.jpg"

    with patch("gui.main_window.QtWidgets.QFileDialog.getOpenFileName", return_value=("/tmp/labels.json", "JSON (*.json)")):
        with patch.object(widget.mbox, "question", return_value=widget.mbox.StandardButton.Ok):
            with patch.object(widget, "_prompt_load_label_mode", return_value="append"):
                with patch("gui.main_window.import_json_label_file", return_value=[]) as import_json:
                    with patch("gui.main_window.legacy_blocks_from_annotations", return_value=[]) as legacy_blocks:
                        widget.loadLabel()

    import_json.assert_called_once()
    kwargs = import_json.call_args.kwargs
    assert kwargs["image_w"] == 640
    assert kwargs["image_h"] == 480
    assert kwargs["image_path"] == "/tmp/sample.jpg"
    assert kwargs["class_mapping"].names == widget.object_list
    legacy_blocks.assert_called_once()

    widget.close()


def test_load_label_replace_uses_replace_controller_path() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.origin_width = 640
    widget.origin_height = 480
    widget.imgfilePath = "/tmp/sample.jpg"

    blocks = [(["naval", 1, 2, 3, 4, 640, 480], ["naval", 10, 20, 30, 40], "naval")]

    with patch("gui.main_window.QtWidgets.QFileDialog.getOpenFileName", return_value=("/tmp/labels.txt", "TXT (*.txt)")):
        with patch.object(widget.mbox, "question", return_value=widget.mbox.StandardButton.Ok):
            with patch.object(widget, "_prompt_load_label_mode", return_value="replace"):
                with patch("gui.main_window.import_yolo_hbb_label_file", return_value=["ann"]) as import_yolo:
                    with patch("gui.main_window.legacy_blocks_from_annotations", return_value=blocks):
                        with patch.object(widget._gt_actions, "replace_with_blocks") as replace_blocks:
                            widget.loadLabel()

    import_yolo.assert_called_once()
    replace_blocks.assert_called_once_with(blocks)

    widget.close()


def test_show_statistics_uses_combined_gt_and_paste_records() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.origin_width = 960
    widget.origin_height = 960
    widget.imgfilePath = "/tmp/sample.jpg"
    widget.real_data = [["naval", 0, 0, 10, 10]]
    widget.box_attributes = [{"size_tag": "small", "crowded": "true"}]
    widget.real_pimg_data = [["merchant", 20, 20, 40, 40]]

    with patch.object(widget, "_prompt_statistics_scope", return_value="image"):
        with patch("gui.main_window.StatisticsDialog") as dialog_cls:
            widget.show_statistics()

    records = dialog_cls.call_args.kwargs["records"]
    assert dialog_cls.call_args.kwargs["scope_label"] == "Current image"
    assert len(records) == 2
    assert records[0]["annotation_source"] == "gt"
    assert records[1]["annotation_source"] == "paste"
    dialog_cls.return_value.exec.assert_called_once()

    widget.close()


def test_run_error_analysis_uses_combined_gt_and_paste_annotations() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.origin_width = 960
    widget.origin_height = 960
    widget.imgfilePath = "/tmp/sample.jpg"
    widget.real_data = [["naval", 0, 0, 10, 10]]
    widget.box_attributes = [{"size_tag": "small", "crowded": "true"}]
    widget.real_pimg_data = [["merchant", 20, 20, 40, 40]]
    widget.predictions = [
        PredictionRecord(
            pred_id="pred-1",
            class_id=0,
            class_name="naval",
            x1=0,
            y1=0,
            x2=10,
            y2=10,
            confidence=0.9,
        )
    ]

    with patch.object(widget, "_prompt_error_analysis_scope", return_value="image"):
        with patch("gui.main_window.ErrorAnalysisDialog") as dialog_cls:
            widget.run_error_analysis()

    kwargs = dialog_cls.call_args.kwargs
    assert len(kwargs["gt_boxes"]) == 2
    assert kwargs["gt_boxes"][0][0] == "naval"
    assert kwargs["gt_boxes"][1][0] == "merchant"
    assert len(kwargs["gt_attributes"]) == 2
    assert kwargs["gt_attributes"][0]["crowded"] == "true"
    assert kwargs["gt_attributes"][1]["crowded"] == "false"
    assert kwargs["scope_label"] == "Current image"
    dialog_cls.return_value.exec.assert_called_once()

    widget.close()


def test_run_error_analysis_folder_scope_uses_scan_result() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = "/tmp/folder/sample.jpg"

    scan_result = FolderErrorAnalysisResult(
        folder_path="/tmp/folder",
        prediction_root="/tmp/preds",
        image_paths=("/tmp/folder/a.jpg", "/tmp/folder/b.jpg"),
        analyzed_image_paths=("/tmp/folder/a.jpg",),
        labeled_image_paths=("/tmp/folder/a.jpg", "/tmp/folder/b.jpg"),
        prediction_image_paths=("/tmp/folder/a.jpg",),
        cases=(ErrorCase(image_id="/tmp/folder/a.jpg", error_type=ERROR_FP),),
    )

    with patch.object(widget, "_prompt_error_analysis_scope", return_value="folder"):
        with patch.object(widget, "_prompt_prediction_folder", return_value=Path("/tmp/preds")):
            with patch.object(widget, "_scan_current_folder_error_cases", return_value=scan_result):
                with patch("gui.main_window.ErrorAnalysisDialog") as dialog_cls:
                    with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Ok) as question:
                        widget.run_error_analysis()

    kwargs = dialog_cls.call_args.kwargs
    assert "Labels matched: 2" in question.call_args.args[2]
    assert "Predictions matched: 1" in question.call_args.args[2]
    assert kwargs["scope_label"] == "Current folder"
    assert len(kwargs["cases"]) == 1
    assert "Images analyzed: 1 / 2" in kwargs["detail_label"]
    assert "Predictions matched: 1" in kwargs["detail_label"]
    dialog_cls.return_value.exec.assert_called_once()

    widget.close()


def test_run_error_analysis_folder_scope_can_cancel_after_summary() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = "/tmp/folder/sample.jpg"

    scan_result = FolderErrorAnalysisResult(
        folder_path="/tmp/folder",
        prediction_root="/tmp/preds",
        image_paths=("/tmp/folder/a.jpg",),
        analyzed_image_paths=("/tmp/folder/a.jpg",),
        labeled_image_paths=("/tmp/folder/a.jpg",),
        prediction_image_paths=("/tmp/folder/a.jpg",),
        cases=(ErrorCase(image_id="/tmp/folder/a.jpg", error_type=ERROR_FP),),
    )

    with patch.object(widget, "_prompt_error_analysis_scope", return_value="folder"):
        with patch.object(widget, "_prompt_prediction_folder", return_value=Path("/tmp/preds")):
            with patch.object(widget, "_scan_current_folder_error_cases", return_value=scan_result):
                with patch("gui.main_window.ErrorAnalysisDialog") as dialog_cls:
                    with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Cancel):
                        widget.run_error_analysis()

    dialog_cls.assert_not_called()
    widget.close()


def test_start_fp_to_label_review_builds_folder_queue_and_opens_first_case() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = "/tmp/folder/current.jpg"

    fp_case = ErrorCase(
        image_id="/tmp/folder/a.jpg",
        error_type=ERROR_FP,
        pred_index=0,
        pred_class="naval",
        confidence=0.9,
        pred_box=(10.0, 20.0, 40.0, 50.0),
    )
    scan_result = FolderErrorAnalysisResult(
        folder_path="/tmp/folder",
        prediction_root="/tmp/preds",
        image_paths=("/tmp/folder/a.jpg",),
        analyzed_image_paths=("/tmp/folder/a.jpg",),
        labeled_image_paths=(),
        prediction_image_paths=("/tmp/folder/a.jpg",),
        cases=(fp_case,),
    )

    with patch.object(widget, "_prompt_fp_review_scope", return_value="folder"):
        with patch.object(widget, "_prompt_prediction_folder", return_value=Path("/tmp/preds")):
            with patch.object(widget, "_scan_current_folder_error_cases", return_value=scan_result):
                with patch.object(widget, "_confirm_folder_error_analysis_run", return_value=True):
                    with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Ok):
                        with patch.object(widget, "_load_image_file", return_value=True) as load_image:
                            with patch.object(widget, "_sync_folder_images_from_path"):
                                with patch.object(widget, "_load_fp_review_predictions_for_current_image"):
                                    with patch.object(widget, "_select_fp_review_prediction", return_value=True):
                                        with patch.object(widget, "set_img_ratio"):
                                            widget.start_fp_to_label_review()

    assert widget._fp_review_queue == [fp_case]
    assert widget._fp_review_index == 0
    assert widget._fp_review_prediction_root == Path("/tmp/preds")
    load_image.assert_called_once_with("/tmp/folder/a.jpg", ask_confirm=True)
    assert widget.action_next_fp_review.isEnabled() is True

    widget.close()


def test_fp_review_prediction_selection_matches_by_bbox_not_stale_index() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.predictions = [
        PredictionRecord("low", 0, "naval", 1, 1, 5, 5, 0.1),
        PredictionRecord("target", 0, "naval", 10, 20, 40, 50, 0.9),
        PredictionRecord("other", 1, "merchant", 60, 60, 80, 80, 0.95),
    ]
    widget._prediction_conf_threshold = 0.5
    widget._refresh_prediction_threshold_label()
    widget._refresh_pred_listwidget()

    case = ErrorCase(
        image_id="/tmp/folder/a.jpg",
        error_type=ERROR_FP,
        pred_index=0,
        pred_class="naval",
        confidence=0.9,
        pred_box=(10.0, 20.0, 40.0, 50.0),
    )

    with patch.object(widget, "_preview_prediction_row"):
        assert widget._select_fp_review_prediction(case) is True

    assert widget._selected_prediction_index() == 1
    assert widget.pred_listwidget.currentRow() == 0

    widget.close()


def test_next_fp_review_keeps_same_image_loaded_to_preserve_unsaved_labels() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = "/tmp/folder/a.jpg"
    widget._fp_review_queue = [
        ErrorCase(image_id="/tmp/folder/a.jpg", error_type=ERROR_FP, pred_index=0),
        ErrorCase(image_id="/tmp/folder/a.jpg", error_type=ERROR_FP, pred_index=1),
    ]
    widget._fp_review_index = 0
    widget._fp_review_prediction_root = Path("/tmp/preds")

    with patch.object(widget, "_load_image_file") as load_image:
        with patch.object(widget, "_select_fp_review_prediction", return_value=True):
            with patch.object(widget, "set_img_ratio"):
                widget.open_next_fp_review_case()

    load_image.assert_not_called()
    assert widget._fp_review_index == 1

    widget.close()


def test_scan_current_folder_error_cases_uses_disk_sidecars_when_live_state_empty() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    image_path = "/tmp/folder/sample.jpg"
    widget.imgfilePath = image_path
    widget._set_folder_images([image_path], current_path=image_path)
    widget.predictions = []

    scan_result = FolderErrorAnalysisResult(
        folder_path="/tmp/folder",
        prediction_root="/tmp/preds",
        image_paths=(image_path,),
        analyzed_image_paths=(),
        labeled_image_paths=(),
        prediction_image_paths=(),
        cases=(),
    )

    with patch("gui.main_window.scan_folder_error_cases", return_value=scan_result) as scan:
        widget._scan_current_folder_error_cases(Path("/tmp/preds"))

    kwargs = scan.call_args.kwargs
    assert kwargs["current_image_path"] is None
    assert kwargs["current_image_gt_bundle"] is None
    assert kwargs["current_image_predictions"] is None
    widget.close()


def test_scan_current_folder_error_cases_uses_empty_live_predictions_after_review() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    image_path = "/tmp/folder/sample.jpg"
    prediction_root = Path("/tmp/preds")
    widget.imgfilePath = image_path
    widget._set_folder_images([image_path], current_path=image_path)
    widget._prediction_folder_path = prediction_root
    pred = PredictionRecord("p1", 0, "naval", 10, 10, 40, 40, 0.9)
    widget._prediction_review_states[widget._review_key(image_path)] = update_prediction_review_state(
        initial_prediction_review_state([pred]),
        rejected_delta=1,
        remaining_predictions=[],
    )
    widget.predictions = []

    scan_result = FolderErrorAnalysisResult(
        folder_path="/tmp/folder",
        prediction_root=str(prediction_root),
        image_paths=(image_path,),
        analyzed_image_paths=(),
        labeled_image_paths=(),
        prediction_image_paths=(),
        cases=(),
    )

    with patch("gui.main_window.scan_folder_error_cases", return_value=scan_result) as scan:
        widget._scan_current_folder_error_cases(prediction_root)

    kwargs = scan.call_args.kwargs
    assert kwargs["current_image_path"] == image_path
    assert kwargs["current_image_gt_bundle"] is None
    assert kwargs["current_image_predictions"] == []
    widget.close()


def test_scan_current_folder_annotations_uses_disk_sidecars_when_live_state_empty() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    image_path = "/tmp/folder/sample.jpg"
    widget.imgfilePath = image_path
    widget._set_folder_images([image_path], current_path=image_path)

    scan_result = FolderAnnotationScanResult(
        folder_path="/tmp/folder",
        image_paths=(image_path,),
        labeled_image_paths=(),
        records=(),
    )

    with patch("gui.main_window.scan_folder_annotation_records", return_value=scan_result) as scan:
        widget._scan_current_folder_annotations()

    kwargs = scan.call_args.kwargs
    assert kwargs["current_image_path"] is None
    assert kwargs["current_image_records"] is None
    widget.close()


def test_scan_current_folder_annotations_uses_live_records_when_present() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    image_path = "/tmp/folder/sample.jpg"
    widget.imgfilePath = image_path
    widget._set_folder_images([image_path], current_path=image_path)
    widget.origin_width = 100
    widget.origin_height = 100
    widget.object_list = ["naval", "merchant", "dock", "other_vessel"]
    widget.real_data = [["naval", 10, 10, 40, 40]]
    widget.box_attributes = [{}]

    scan_result = FolderAnnotationScanResult(
        folder_path="/tmp/folder",
        image_paths=(image_path,),
        labeled_image_paths=(image_path,),
        records=(),
    )

    with patch("gui.main_window.scan_folder_annotation_records", return_value=scan_result) as scan:
        widget._scan_current_folder_annotations()

    kwargs = scan.call_args.kwargs
    assert kwargs["current_image_path"] == image_path
    assert kwargs["current_image_records"][0]["class_name"] == "naval"
    widget.close()


def test_show_statistics_folder_scope_uses_scan_result() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = "/tmp/folder/sample.jpg"

    scan_result = FolderAnnotationScanResult(
        folder_path="/tmp/folder",
        image_paths=("/tmp/folder/a.jpg", "/tmp/folder/b.jpg"),
        labeled_image_paths=("/tmp/folder/a.jpg",),
        records=(
            {
                "image_path": "/tmp/folder/a.jpg",
                "class_name": "naval",
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "size_tag": "small",
                "scene_tag": "unknown",
            },
        ),
    )

    with patch.object(widget, "_prompt_statistics_scope", return_value="folder"):
        with patch.object(widget, "_scan_current_folder_annotations", return_value=scan_result):
            with patch("gui.main_window.StatisticsDialog") as dialog_cls:
                with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Ok) as question:
                    widget.show_statistics()

    kwargs = dialog_cls.call_args.kwargs
    assert "Labels matched: 1" in question.call_args.args[2]
    assert "Annotations found: 1" in question.call_args.args[2]
    assert kwargs["scope_label"] == "Current folder"
    assert "Label root:" in kwargs["detail_label"]
    assert kwargs["total_images_override"] == 2
    assert kwargs["labeled_images_override"] == 1
    assert len(kwargs["records"]) == 1
    dialog_cls.return_value.exec.assert_called_once()

    widget.close()


def test_show_statistics_folder_scope_can_cancel_after_summary() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = "/tmp/folder/sample.jpg"

    scan_result = FolderAnnotationScanResult(
        folder_path="/tmp/folder",
        image_paths=("/tmp/folder/a.jpg",),
        labeled_image_paths=("/tmp/folder/a.jpg",),
        records=(
            {
                "image_path": "/tmp/folder/a.jpg",
                "class_name": "naval",
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "size_tag": "small",
                "scene_tag": "unknown",
            },
        ),
    )

    with patch.object(widget, "_prompt_statistics_scope", return_value="folder"):
        with patch.object(widget, "_scan_current_folder_annotations", return_value=scan_result):
            with patch("gui.main_window.StatisticsDialog") as dialog_cls:
                with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Cancel):
                    widget.show_statistics()

    dialog_cls.assert_not_called()
    widget.close()


def test_show_statistics_project_scope_uses_scan_result() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    project_dir = Path("/tmp/proj")
    widget.imgfilePath = str(project_dir / "images" / "train" / "sample.jpg")
    widget._project_config = ProjectConfig(project_root=".", image_root="images", label_root="labels")
    widget._project_config_path = project_dir / "project_config.yaml"

    scan_result = FolderAnnotationScanResult(
        folder_path=str(project_dir / "images"),
        image_paths=(str(project_dir / "images" / "train" / "a.jpg"), str(project_dir / "images" / "val" / "b.jpg")),
        labeled_image_paths=(str(project_dir / "images" / "train" / "a.jpg"),),
        records=(
            {
                "image_path": str(project_dir / "images" / "train" / "a.jpg"),
                "class_name": "naval",
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "size_tag": "small",
                "scene_tag": "unknown",
            },
        ),
    )

    with patch.object(widget, "_prompt_statistics_scope", return_value="project"):
        with patch.object(widget, "_scan_current_project_annotations", return_value=scan_result):
            with patch("gui.main_window.StatisticsDialog") as dialog_cls:
                with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Ok) as question:
                    widget.show_statistics()

    kwargs = dialog_cls.call_args.kwargs
    assert "Labels matched: 1" in question.call_args.args[2]
    assert kwargs["scope_label"] == "Current project"
    assert "Label root:" in kwargs["detail_label"]
    assert kwargs["total_images_override"] == 2
    dialog_cls.return_value.exec.assert_called_once()
    widget.close()


def test_run_error_analysis_project_scope_uses_scan_result() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    project_dir = Path("/tmp/proj")
    widget.imgfilePath = str(project_dir / "images" / "train" / "sample.jpg")
    widget._project_config = ProjectConfig(project_root=".", image_root="images", label_root="labels")
    widget._project_config_path = project_dir / "project_config.yaml"

    scan_result = FolderErrorAnalysisResult(
        folder_path=str(project_dir / "images"),
        prediction_root=str(project_dir / "predictions"),
        image_paths=(str(project_dir / "images" / "train" / "a.jpg"), str(project_dir / "images" / "val" / "b.jpg")),
        analyzed_image_paths=(str(project_dir / "images" / "train" / "a.jpg"),),
        labeled_image_paths=(str(project_dir / "images" / "train" / "a.jpg"),),
        prediction_image_paths=(str(project_dir / "images" / "train" / "a.jpg"),),
        cases=(ErrorCase(image_id=str(project_dir / "images" / "train" / "a.jpg"), error_type=ERROR_FP),),
    )

    with patch.object(widget, "_prompt_error_analysis_scope", return_value="project"):
        with patch.object(widget, "_prompt_prediction_folder", return_value=project_dir / "predictions"):
            with patch.object(widget, "_scan_current_project_error_cases", return_value=scan_result):
                with patch("gui.main_window.ErrorAnalysisDialog") as dialog_cls:
                    with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Ok) as question:
                        widget.run_error_analysis()

    kwargs = dialog_cls.call_args.kwargs
    assert "Predictions matched: 1" in question.call_args.args[2]
    assert kwargs["scope_label"] == "Current project"
    assert "Prediction root:" in kwargs["detail_label"]
    dialog_cls.return_value.exec.assert_called_once()
    widget.close()


def test_run_dataset_qc_folder_scope_uses_scan_result() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = "/tmp/folder/sample.jpg"

    result = DatasetValidationResult(
        scope_path="/tmp/folder",
        image_paths=("/tmp/folder/a.jpg", "/tmp/folder/b.jpg"),
        matched_label_paths=("/tmp/labels/a.txt",),
        matched_prediction_paths=("/tmp/preds/a.txt",),
        issues=(
            ValidationIssue(
                image_path="/tmp/folder/b.jpg",
                file_path="/tmp/labels/b.txt",
                source="label",
                issue_type="missing_label",
                detail="No matching label sidecar found.",
            ),
        ),
    )

    with patch.object(widget, "_prompt_validation_scope", return_value="folder"):
        with patch.object(widget, "_prompt_optional_validation_prediction_folder", return_value=Path("/tmp/preds")):
            with patch.object(widget, "_scan_current_folder_validation", return_value=result):
                with patch("gui.main_window.ValidationDialog") as dialog_cls:
                    with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Ok) as question:
                        widget.run_dataset_qc()

    kwargs = dialog_cls.call_args.kwargs
    assert "Labels matched: 1" in question.call_args.args[2]
    assert "Predictions matched: 1" in question.call_args.args[2]
    assert kwargs["scope_label"] == "Current folder"
    assert kwargs["result"] == result
    assert "Prediction folder: /tmp/preds" in kwargs["detail_label"]
    dialog_cls.return_value.exec.assert_called_once()
    widget.close()


def test_run_dataset_qc_project_scope_can_skip_prediction_checks() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    project_dir = Path("/tmp/proj")
    widget.imgfilePath = str(project_dir / "images" / "train" / "sample.jpg")
    widget._project_config = ProjectConfig(project_root=".", image_root="images", label_root="labels")
    widget._project_config_path = project_dir / "project_config.yaml"

    result = DatasetValidationResult(
        scope_path=str(project_dir / "images"),
        image_paths=(str(project_dir / "images" / "train" / "a.jpg"),),
        matched_label_paths=(str(project_dir / "labels" / "train" / "a.txt"),),
        matched_prediction_paths=(),
        issues=(),
    )

    with patch.object(widget, "_prompt_validation_scope", return_value="project"):
        with patch.object(widget, "_prompt_optional_validation_prediction_folder", return_value=None):
            with patch.object(widget, "_scan_current_project_validation", return_value=result):
                with patch("gui.main_window.ValidationDialog") as dialog_cls:
                    with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Ok) as question:
                        widget.run_dataset_qc()

    kwargs = dialog_cls.call_args.kwargs
    assert "Predictions matched: 0" in question.call_args.args[2]
    assert kwargs["scope_label"] == "Current project"
    assert "Prediction root: (not checked)" in kwargs["detail_label"]
    dialog_cls.return_value.exec.assert_called_once()
    widget.close()


def test_show_prediction_review_summary_folder_scope_uses_scan_result() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = "/tmp/folder/sample.jpg"
    widget._prediction_folder_path = Path("/tmp/preds")

    report = PredictionReviewReport(
        scope_path="/tmp/folder",
        prediction_root="/tmp/preds",
        image_paths=("/tmp/folder/a.jpg", "/tmp/folder/b.jpg"),
        entries=(
            PredictionReviewReportEntry("/tmp/folder/a.jpg", "/tmp/preds/a.txt", True, "reviewed", 2, 2, 0, 0),
            PredictionReviewReportEntry("/tmp/folder/b.jpg", "/tmp/preds/b.txt", True, "pending", 1, 0, 0, 1),
        ),
    )

    with patch.object(widget, "_prompt_review_summary_scope", return_value="folder"):
        with patch.object(widget, "_scan_current_folder_review_report", return_value=report):
            with patch("gui.main_window.PredictionReviewReportDialog") as dialog_cls:
                with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Ok) as question:
                    widget.show_prediction_review_summary()

    kwargs = dialog_cls.call_args.kwargs
    assert "Images with predictions: 2" in question.call_args.args[2]
    assert "Reviewed images: 1" in question.call_args.args[2]
    assert kwargs["scope_label"] == "Current folder"
    assert kwargs["report"] == report
    dialog_cls.return_value.exec.assert_called_once()
    widget.close()


def test_show_prediction_review_summary_project_scope_prompts_for_prediction_root_when_missing() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    project_dir = Path("/tmp/proj")
    widget.imgfilePath = str(project_dir / "images" / "train" / "sample.jpg")
    widget._project_config = ProjectConfig(project_root=".", image_root="images", label_root="labels")
    widget._project_config_path = project_dir / "project_config.yaml"

    report = PredictionReviewReport(
        scope_path=str(project_dir / "images"),
        prediction_root=str(project_dir / "predictions"),
        image_paths=(str(project_dir / "images" / "train" / "a.jpg"),),
        entries=(PredictionReviewReportEntry(str(project_dir / "images" / "train" / "a.jpg"), str(project_dir / "predictions" / "train" / "a.txt"), True, "pending", 1, 0, 0, 1),),
    )

    with patch.object(widget, "_prompt_review_summary_scope", return_value="project"):
        with patch.object(widget, "_prompt_prediction_folder", return_value=project_dir / "predictions") as prompt_pred:
            with patch.object(widget, "_scan_current_project_review_report", return_value=report):
                with patch("gui.main_window.PredictionReviewReportDialog") as dialog_cls:
                    with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Ok):
                        widget.show_prediction_review_summary()

    prompt_pred.assert_called_once()
    kwargs = dialog_cls.call_args.kwargs
    assert kwargs["scope_label"] == "Current project"
    assert "Prediction root:" in kwargs["detail_label"]
    dialog_cls.return_value.exec.assert_called_once()
    widget.close()


def test_apply_project_config_loads_classes_yaml_from_config_path(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    classes_path = project_dir / "custom_classes.yaml"
    save_classes_yaml_path(
        ClassCatalog.from_list(
            "custom_project",
            [
                ClassInfo(0, "alpha_ship", "vessel"),
                ClassInfo(1, "beta_dock", "facility"),
            ],
        ),
        classes_path,
    )

    widget._project_config = ProjectConfig(
        project_root=".",
        classes_yaml="custom_classes.yaml",
    )
    widget._project_config_path = project_dir / "project_config.yaml"

    widget._apply_project_config()

    assert widget.object_list == ["alpha_ship", "beta_dock"]

    widget.close()


def test_project_config_default_directories_follow_config_path(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    project_dir = tmp_path / "proj"
    image_root = project_dir / "images"
    label_root = project_dir / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)

    widget._project_config = ProjectConfig(
        project_root=".",
        image_root="images",
        label_root="labels",
    )
    widget._project_config_path = project_dir / "project_config.yaml"

    assert widget._default_image_directory() == image_root.resolve()
    assert widget._default_label_directory() == label_root.resolve()

    with patch("gui.main_window.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")) as picker:
        widget.newFile()
    assert picker.call_args.kwargs["directory"] == str(image_root.resolve())

    with patch("gui.main_window.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")) as picker:
        widget.loadLabel()
    assert picker.call_args.kwargs["directory"] == str(label_root.resolve())

    widget.close()


def test_input_obj_uses_project_classes_yaml_path(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    widget._project_config = ProjectConfig(
        project_root=".",
        classes_yaml="configs/classes.yaml",
    )
    widget._project_config_path = project_dir / "project_config.yaml"

    with patch("gui.main_window.ClassMappingDialog") as dialog_cls:
        dialog_cls.return_value.exec.return_value = 0
        widget.inputObj()

    assert dialog_cls.call_args.kwargs["default_yaml_path"] == (project_dir / "configs" / "classes.yaml").resolve()
    dialog_cls.return_value.exec.assert_called_once()

    widget.close()


def test_savelab_window_uses_project_default_format_and_label_root(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    project_dir = tmp_path / "proj"
    image_root = project_dir / "images" / "train"
    label_root = project_dir / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    widget._project_config = ProjectConfig(
        project_root=".",
        image_root="images",
        label_root="labels",
        default_export_format="coco_json",
    )
    widget._project_config_path = project_dir / "project_config.yaml"
    widget.imgfilePath = str(image_root / "sample.jpg")
    widget.origin_width = 960
    widget.origin_height = 960
    widget.real_data = [["naval", 0, 0, 10, 10]]

    dlg = SavelabWindow(widget)

    assert dlg.box_format.currentText() == "COCO JSON"

    with patch("gui.dialogs.savelab_window.QtWidgets.QFileDialog.getSaveFileName", return_value=("", "")) as picker:
        dlg.saveLabel()

    assert picker.call_args.kwargs["directory"] == str((label_root / "train" / "sample.json").resolve())
    dlg.close()
    widget.close()


def test_run_model_predictions_uses_service_and_populates_prediction_list() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    origin = QPixmap(200, 200)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.origin_width = 200
    widget.origin_height = 200
    widget.imgfilePath = "/tmp/sample.jpg"
    widget.ratio_value = 50
    widget._yolo_model_handle = object()  # type: ignore[assignment]

    preds = [
        PredictionRecord(
            pred_id="pred-1",
            class_id=1,
            class_name="merchant",
            x1=10,
            y1=20,
            x2=50,
            y2=80,
            confidence=0.88,
        )
    ]

    with patch("gui.main_window.run_yolo_model_inference", return_value=preds) as infer:
        widget.run_model_predictions()

    infer.assert_called_once()
    kwargs = infer.call_args.kwargs
    assert kwargs["image_path"] == "/tmp/sample.jpg"
    assert kwargs["object_list"] == widget.object_list
    assert widget.predictions == preds
    assert widget.pred_listwidget.count() == 1
    assert widget.pred_listwidget.currentRow() == 0

    widget.close()


def test_prediction_geometry_update_marks_prediction_as_edited() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    origin = QPixmap(200, 200)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.origin_width = 200
    widget.origin_height = 200
    widget.ratio_value = 50
    widget._image_canvas.set_canvas(origin.copy())
    widget.set_img_ratio()

    widget.predictions = [
        PredictionRecord(
            pred_id="pred-1",
            class_id=0,
            class_name="ship",
            x1=10,
            y1=20,
            x2=50,
            y2=60,
            confidence=0.9,
        )
    ]
    widget._refresh_pred_listwidget()
    widget.pred_listwidget.setCurrentRow(0)
    APP.processEvents()

    ok = widget._request_prediction_update(0, 20, 30, 80, 90)

    assert ok is True
    assert widget.predictions[0].x1 == 20
    assert widget.predictions[0].y1 == 30
    assert widget.predictions[0].x2 == 80
    assert widget.predictions[0].y2 == 90
    assert widget.predictions[0].pred_status == "edited"
    assert "[edited]" in widget.pred_listwidget.item(0).text()

    widget.close()


def test_rename_selected_prediction_updates_class_and_clears_gt_focus() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    origin = QPixmap(200, 200)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.origin_width = 200
    widget.origin_height = 200
    widget.ratio_value = 50
    widget._image_canvas.set_canvas(origin.copy())
    widget.set_img_ratio()

    widget._gt_actions.add_box(
        ["naval", 5, 5, 25, 25, 200, 200],
        ["naval", 5, 5, 25, 25],
        "naval",
        False,
        select_new_row=True,
    )
    assert widget.listwidget.currentRow() == 0

    widget.predictions = [
        PredictionRecord(
            pred_id="pred-1",
            class_id=0,
            class_name="naval",
            x1=10,
            y1=20,
            x2=50,
            y2=60,
            confidence=0.9,
        )
    ]
    widget._refresh_pred_listwidget()

    with patch("gui.main_window.QtWidgets.QInputDialog.getItem", return_value=("merchant", True)):
        widget.pred_listwidget.setCurrentRow(0)
        APP.processEvents()
        widget._rename_selected_prediction()

    assert widget.predictions[0].class_name == "merchant"
    assert widget.predictions[0].class_id == widget.object_list.index("merchant")
    assert widget.predictions[0].pred_status == "edited"
    assert widget.listwidget.currentRow() == -1
    assert widget.pred_listwidget.item(0).text().startswith("merchant ")

    widget.close()


def test_toggle_error_overlay_forces_gt_and_predictions_visible() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    origin = QPixmap(200, 200)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.origin_width = 200
    widget.origin_height = 200
    widget.ratio_value = 50
    widget._image_canvas.set_canvas(origin.copy())

    widget._gt_actions.add_box(
        ["naval", 10, 10, 70, 70, 200, 200],
        ["naval", 10, 10, 70, 70],
        "naval",
        False,
    )
    widget.predictions = [
        PredictionRecord(
            pred_id="pred-1",
            class_id=0,
            class_name="naval",
            x1=12,
            y1=12,
            x2=68,
            y2=68,
            confidence=0.95,
        )
    ]
    widget.hideBox.setChecked(True)
    widget.chk_show_preds.setChecked(False)

    widget._toggle_error_overlay(True)

    assert widget._error_overlay_enabled is True
    assert widget.hideBox.isChecked() is False
    assert widget.chk_show_preds.isChecked() is True
    assert len(widget._current_error_overlay_cases()) == 1
    assert widget._current_error_overlay_cases()[0].error_type == "TP"

    widget.close()


def test_default_prediction_directory_prefers_sibling_predictions_folder(tmp_path: Path) -> None:
    image_dir = tmp_path / "dataset" / "test" / "images"
    prediction_dir = tmp_path / "dataset" / "test" / "predictions"
    image_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)

    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = str(image_dir / "t_0005.bmp")
    widget._set_folder_images([widget.imgfilePath], current_path=widget.imgfilePath)

    assert widget._default_prediction_directory() == prediction_dir
    widget.close()


def test_default_prediction_directory_maps_nested_images_folder_to_predictions(tmp_path: Path) -> None:
    image_dir = tmp_path / "dataset" / "images" / "test"
    prediction_dir = tmp_path / "dataset" / "predictions" / "test"
    image_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)

    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = str(image_dir / "t_0005.bmp")
    widget._set_folder_images([widget.imgfilePath], current_path=widget.imgfilePath)

    assert widget._default_prediction_directory() == prediction_dir
    widget.close()


def test_load_predictions_dialog_starts_in_prediction_directory(tmp_path: Path) -> None:
    image_dir = tmp_path / "dataset" / "test" / "images"
    prediction_dir = tmp_path / "dataset" / "test" / "predictions"
    image_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)

    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = str(image_dir / "t_0005.bmp")
    widget.origin_width = 100
    widget.origin_height = 100
    widget._set_folder_images([widget.imgfilePath], current_path=widget.imgfilePath)

    with patch("gui.main_window.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")) as dialog:
        widget.load_predictions()

    assert dialog.call_args.kwargs["directory"] == str(prediction_dir)
    widget.close()


def test_load_prediction_folder_auto_loads_current_image_sidecar(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget.imgfilePath = str(tmp_path / "sample.jpg")
    widget.origin_width = 100
    widget.origin_height = 100
    pred_root = tmp_path / "preds"
    pred_root.mkdir()
    (pred_root / "sample.txt").write_text("0 0.5 0.5 0.4 0.4 0.8\n", encoding="utf-8")

    with patch("gui.main_window.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(pred_root)):
        widget.load_prediction_folder()

    assert widget._prediction_folder_path == pred_root
    assert len(widget.predictions) == 1
    assert widget.predictions[0].class_name == "naval"
    assert widget.pred_listwidget.count() == 1

    widget.close()


def test_load_prediction_folder_resumes_saved_review_session(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget._project_config_path = tmp_path / "project_config.yaml"
    widget.imgfilePath = str(tmp_path / "images" / "sample.jpg")
    widget.origin_width = 100
    widget.origin_height = 100
    widget._set_folder_images([str(tmp_path / "images" / "sample.jpg")], current_path=str(tmp_path / "images" / "sample.jpg"))
    pred_root = tmp_path / "preds"
    pred_root.mkdir(parents=True)
    (pred_root / "sample.txt").write_text("0 0.5 0.5 0.4 0.4 0.8\n", encoding="utf-8")
    save_prediction_review_session(
        image_folder=tmp_path / "images",
        prediction_folder=pred_root,
        review_root=tmp_path,
        states={
            str((tmp_path / "images" / "sample.jpg").resolve()): update_prediction_review_state(
                initial_prediction_review_state([PredictionRecord("p1", 0, "naval", 10, 10, 40, 40, 0.9)]),
                rejected_delta=1,
                remaining_predictions=[],
            )
        },
    )

    with patch("gui.main_window.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(pred_root)):
        with patch.object(widget, "_prompt_prediction_review_resume_mode", return_value="resume"):
            widget.load_prediction_folder()

    assert widget._prediction_folder_path == pred_root
    assert widget._review_status_for_image(tmp_path / "images" / "sample.jpg") == "reviewed"
    assert widget.predictions == []

    widget.close()


def test_load_prediction_folder_start_fresh_clears_saved_session(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget._project_config_path = tmp_path / "project_config.yaml"
    widget.imgfilePath = str(tmp_path / "images" / "sample.jpg")
    widget.origin_width = 100
    widget.origin_height = 100
    widget._set_folder_images([str(tmp_path / "images" / "sample.jpg")], current_path=str(tmp_path / "images" / "sample.jpg"))
    pred_root = tmp_path / "preds"
    pred_root.mkdir(parents=True)
    (pred_root / "sample.txt").write_text("0 0.5 0.5 0.4 0.4 0.8\n", encoding="utf-8")
    save_prediction_review_session(
        image_folder=tmp_path / "images",
        prediction_folder=pred_root,
        review_root=tmp_path,
        states={
            str((tmp_path / "images" / "sample.jpg").resolve()): update_prediction_review_state(
                initial_prediction_review_state([PredictionRecord("p1", 0, "naval", 10, 10, 40, 40, 0.9)]),
                rejected_delta=1,
                remaining_predictions=[],
            )
        },
    )

    with patch("gui.main_window.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(pred_root)):
        with patch.object(widget, "_prompt_prediction_review_resume_mode", return_value="fresh"):
            widget.load_prediction_folder()

    assert widget._prediction_folder_path == pred_root
    assert len(widget.predictions) == 1
    assert widget._review_status_for_image(tmp_path / "images" / "sample.jpg") == "pending"
    assert has_prediction_review_session(
        image_folder=tmp_path / "images",
        prediction_folder=pred_root,
        review_root=tmp_path,
    ) is True

    widget.close()


def test_open_next_review_image_skips_images_without_prediction_sidecars(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    pred_root = tmp_path / "preds"
    pred_root.mkdir()
    (pred_root / "c.txt").write_text("0 0.5 0.5 0.4 0.4 0.8\n", encoding="utf-8")
    widget._prediction_folder_path = pred_root
    widget._set_folder_images(
        [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg"), str(tmp_path / "c.jpg")],
        current_path=str(tmp_path / "a.jpg"),
    )

    with patch.object(widget, "_load_image_file", return_value=True) as load_image:
        widget.open_next_review_image()

    load_image.assert_called_once_with(str(tmp_path / "c.jpg"), ask_confirm=False)
    assert widget._folder_image_index == 2

    widget.close()


def test_accept_all_visible_predictions_appends_gt_and_keeps_hidden_predictions() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    origin = QPixmap(200, 200)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.origin_width = 200
    widget.origin_height = 200
    widget.ratio_value = 50
    widget._image_canvas.set_canvas(origin.copy())
    widget.set_img_ratio()
    widget._prediction_conf_threshold = 0.5

    widget.predictions = [
        PredictionRecord("p1", 0, "naval", 10, 10, 40, 40, 0.9),
        PredictionRecord("p2", 1, "merchant", 60, 60, 100, 100, 0.8),
        PredictionRecord("p3", 0, "naval", 120, 120, 140, 140, 0.3),
    ]
    widget._refresh_pred_listwidget()

    widget.accept_all_visible_predictions()

    assert len(widget.real_data) == 2
    assert len(widget.predictions) == 1
    assert widget.predictions[0].pred_id == "p3"
    assert widget.pred_listwidget.count() == 0

    widget.close()


def test_clear_saved_prediction_review_state_resets_current_review(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()
    widget._project_config_path = tmp_path / "project_config.yaml"
    widget.imgfilePath = str(tmp_path / "images" / "sample.jpg")
    widget.origin_width = 100
    widget.origin_height = 100
    widget._set_folder_images([str(tmp_path / "images" / "sample.jpg")], current_path=str(tmp_path / "images" / "sample.jpg"))
    pred_root = tmp_path / "preds"
    pred_root.mkdir(parents=True)
    (pred_root / "sample.txt").write_text("0 0.5 0.5 0.4 0.4 0.8\n", encoding="utf-8")
    widget._prediction_folder_path = pred_root
    widget._prediction_review_states = {
        str((tmp_path / "images" / "sample.jpg").resolve()): update_prediction_review_state(
            initial_prediction_review_state([PredictionRecord("p1", 0, "naval", 10, 10, 40, 40, 0.9)]),
            rejected_delta=1,
            remaining_predictions=[],
        )
    }
    widget.predictions = []
    widget._save_prediction_review_session()

    with patch("gui.main_window.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.StandardButton.Yes):
        widget.clear_saved_prediction_review_state()

    assert len(widget.predictions) == 1
    assert widget._review_status_for_image(tmp_path / "images" / "sample.jpg") == "pending"
    assert has_prediction_review_session(
        image_folder=tmp_path / "images",
        prediction_folder=pred_root,
        review_root=tmp_path,
    ) is True

    widget.close()


def test_reject_all_visible_predictions_removes_only_visible_predictions() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    origin = QPixmap(200, 200)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.origin_width = 200
    widget.origin_height = 200
    widget.ratio_value = 50
    widget._image_canvas.set_canvas(origin.copy())
    widget.set_img_ratio()
    widget._prediction_conf_threshold = 0.5

    widget.predictions = [
        PredictionRecord("p1", 0, "naval", 10, 10, 40, 40, 0.9),
        PredictionRecord("p2", 1, "merchant", 60, 60, 100, 100, 0.8),
        PredictionRecord("p3", 0, "naval", 120, 120, 140, 140, 0.3),
    ]
    widget._refresh_pred_listwidget()

    widget.reject_all_visible_predictions()

    assert len(widget.predictions) == 1
    assert widget.predictions[0].pred_id == "p3"
    assert widget.pred_listwidget.count() == 0

    widget.close()


def test_review_status_label_tracks_pending_queue_and_current_summary(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    widget._prediction_folder_path = tmp_path / "preds"
    widget._set_folder_images(
        [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")],
        current_path=str(tmp_path / "a.jpg"),
    )
    widget.imgfilePath = str(tmp_path / "a.jpg")
    widget._prediction_review_states = {
        str((tmp_path / "a.jpg").resolve()): initial_prediction_review_state(
            [PredictionRecord("p1", 0, "naval", 0, 0, 10, 10, 0.9)]
        ),
        str((tmp_path / "b.jpg").resolve()): update_prediction_review_state(
            initial_prediction_review_state([PredictionRecord("p2", 0, "naval", 0, 0, 10, 10, 0.9)]),
            rejected_delta=1,
            remaining_predictions=[],
        ),
    }

    with patch.object(widget, "_has_review_prediction_for_image", return_value=True):
        widget._refresh_prediction_review_actions()

    text = widget.lbl_review_status.text()
    assert "1 pending / 2 with preds" in text
    assert "current pending" in text

    widget.close()


def test_accept_all_visible_predictions_auto_advances_to_next_unreviewed_review_image(tmp_path: Path) -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    origin = QPixmap(200, 200)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.origin_width = 200
    widget.origin_height = 200
    widget.ratio_value = 50
    widget._image_canvas.set_canvas(origin.copy())
    widget.set_img_ratio()
    widget.imgfilePath = str(tmp_path / "a.jpg")
    widget._prediction_folder_path = tmp_path / "preds"
    widget._set_folder_images(
        [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg"), str(tmp_path / "c.jpg")],
        current_path=str(tmp_path / "a.jpg"),
    )
    widget.predictions = [PredictionRecord("p1", 0, "naval", 10, 10, 40, 40, 0.9)]
    widget._prediction_review_states[str((tmp_path / "a.jpg").resolve())] = initial_prediction_review_state(widget.predictions)
    widget._prediction_review_states[str((tmp_path / "b.jpg").resolve())] = update_prediction_review_state(
        initial_prediction_review_state([PredictionRecord("p2", 0, "naval", 10, 10, 40, 40, 0.9)]),
        rejected_delta=1,
        remaining_predictions=[],
    )
    widget._refresh_pred_listwidget()

    with patch.object(widget, "_has_review_prediction_for_image", side_effect=lambda path: Path(path).name in {"a.jpg", "b.jpg", "c.jpg"}):
        with patch.object(widget, "_load_image_file", return_value=True) as load_image:
            widget.accept_all_visible_predictions()

    load_image.assert_called_once_with(str(tmp_path / "c.jpg"), ask_confirm=False)
    assert widget._review_status_for_image(tmp_path / "a.jpg") == "reviewed"

    widget.close()


def test_paste_zone_drag_stores_origin_rect() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    origin = QPixmap(200, 200)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.origin_width = 200
    widget.origin_height = 200
    widget.ratio_value = 50
    widget.set_img_ratio()

    widget._begin_paste_zone_selection()
    widget._handle_paste_zone_press(_FakeMouseEvent(20, 30))
    widget._handle_paste_zone_release(_FakeMouseEvent(120, 130))

    assert widget.paste_candidate.smart_zone_rect == (20, 30, 120, 130)
    assert widget.btn_clear_zone.isEnabled() is True

    widget.close()
