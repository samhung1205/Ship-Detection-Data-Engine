"""
Ship Detection Data Engine (SDDE).

This package contains the data model and service-layer utilities that the GUI
will eventually connect to (stepwise refactor from the legacy PyQt6 code).
"""

from .models import (
    ClassMapping,
    HBBBoxPx,
    HBBBoxYoloNorm,
    HBBAnnotation,
    ImageAnnotation,
    AnnotationAttributes,
)
from .import_export import export_bbox_txt, export_yolo_hbb_txt, import_yolo_hbb_label_file, parse_yolo_hbb_txt
from .class_catalog import ClassCatalog, ClassInfo, default_ship_catalog
from .classes_yaml import load_classes_yaml_path, save_classes_yaml_path
from .prediction import (
    PredictionRecord,
    STATUS_ACCEPTED,
    STATUS_EDITED,
    STATUS_PREDICTED,
    STATUS_REJECTED,
    parse_predictions_yolo_txt,
)
from .error_analysis import (
    ErrorCase,
    export_error_cases_csv,
    iou_xyxy,
    match_gt_pred,
    summarise_error_cases,
)
from .tile import (
    TileConfig,
    TileRect,
    annotations_in_tile,
    compute_tile_grid,
    global_to_tile,
    tile_to_global,
)
from .augmentation import (
    PasteRecord,
    export_paste_records_csv,
    export_paste_records_json,
)
from .statistics import (
    compute_dataset_stats,
    export_stats_csv,
    export_stats_json,
)
from .project_config import ProjectConfig, load_project_config, save_project_config
from .autosave import has_autosave, read_autosave, remove_autosave, write_autosave

__all__ = [
    "ClassMapping",
    "HBBBoxPx",
    "HBBBoxYoloNorm",
    "HBBAnnotation",
    "ImageAnnotation",
    "AnnotationAttributes",
    "parse_yolo_hbb_txt",
    "import_yolo_hbb_label_file",
    "export_yolo_hbb_txt",
    "export_bbox_txt",
    "ClassCatalog",
    "ClassInfo",
    "default_ship_catalog",
    "load_classes_yaml_path",
    "save_classes_yaml_path",
    "PredictionRecord",
    "STATUS_ACCEPTED",
    "STATUS_EDITED",
    "STATUS_PREDICTED",
    "STATUS_REJECTED",
    "parse_predictions_yolo_txt",
    "ErrorCase",
    "export_error_cases_csv",
    "iou_xyxy",
    "match_gt_pred",
    "summarise_error_cases",
    "TileConfig",
    "TileRect",
    "annotations_in_tile",
    "compute_tile_grid",
    "global_to_tile",
    "tile_to_global",
    "PasteRecord",
    "export_paste_records_csv",
    "export_paste_records_json",
    "compute_dataset_stats",
    "export_stats_csv",
    "export_stats_json",
    "ProjectConfig",
    "load_project_config",
    "save_project_config",
    "has_autosave",
    "read_autosave",
    "remove_autosave",
    "write_autosave",
]

