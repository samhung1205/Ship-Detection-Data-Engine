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
from .import_export import (
    import_annotation_metadata_json_file,
    import_coco_bbox_json_file,
    import_json_label_file,
    import_yolo_hbb_label_file,
    parse_annotation_metadata_json,
    parse_coco_bbox_json,
    parse_yolo_hbb_txt,
    export_bbox_txt,
    export_coco_bbox_json,
    export_pascal_voc_xml,
    export_yolo_hbb_txt,
)
from .class_catalog import ClassCatalog, ClassInfo, default_ship_catalog
from .classes_yaml import load_classes_yaml_path, save_classes_yaml_path
from .prediction import (
    PredictionRecord,
    STATUS_ACCEPTED,
    STATUS_EDITED,
    STATUS_PREDICTED,
    STATUS_REJECTED,
    filter_predictions_by_confidence,
    parse_predictions_yolo_txt,
    rename_prediction_class,
    update_prediction_geometry_from_canvas_rect,
)
from .prediction_scan import (
    has_prediction_sidecar,
    load_prediction_sidecar,
    prediction_sidecar_path,
)
from .error_analysis import (
    ALL_ERROR_TYPES,
    ERROR_FILTER_ALL,
    ErrorCase,
    export_error_cases_csv,
    filter_error_cases,
    gt_attributes_for_case,
    iou_xyxy,
    match_gt_pred,
    summarise_error_cases,
)
from .tile import (
    TileConfig,
    TileRect,
    annotations_in_tile,
    boundary_crossing_annotations,
    compute_tile_grid,
    find_neighbor_tile_index,
    find_tile_index_by_point,
    global_to_tile,
    tile_to_global,
)
from .augmentation import (
    PasteAdjustments,
    PasteRecord,
    export_paste_records_csv,
    export_paste_records_json,
)
from .statistics import (
    compute_dataset_stats,
    export_stats_csv,
    export_stats_json,
)
from .project_config import (
    ProjectConfig,
    load_project_config,
    resolve_project_path,
    resolve_project_root,
    save_project_config,
)
from .autosave import has_autosave, read_autosave, remove_autosave, write_autosave
from .document import AnnotationDocument
from .document import AnnotationBoxState, AnnotationDocumentSnapshot
from .dataset_scan import (
    FolderAnnotationScanResult,
    ImageAnnotationBundle,
    load_image_annotation_bundle,
    read_image_size,
    scan_folder_annotation_records,
)
from .error_analysis_scan import FolderErrorAnalysisResult, scan_folder_error_cases
from .paste_candidate import PasteCandidateSession
from .paste_document import PasteDocument, PasteEntryState
from .legacy_rows import (
    annotations_from_legacy_rows,
    class_mapping_from_object_list,
    legacy_blocks_from_annotations,
)
from .image_browser import (
    SUPPORTED_IMAGE_SUFFIXES,
    find_image_index,
    is_supported_image_path,
    list_supported_images,
)
from .model_inference import YoloModelHandle, load_yolo_model, run_yolo_model_inference

__all__ = [
    "ClassMapping",
    "HBBBoxPx",
    "HBBBoxYoloNorm",
    "HBBAnnotation",
    "ImageAnnotation",
    "AnnotationAttributes",
    "parse_yolo_hbb_txt",
    "parse_coco_bbox_json",
    "parse_annotation_metadata_json",
    "import_yolo_hbb_label_file",
    "import_coco_bbox_json_file",
    "import_annotation_metadata_json_file",
    "import_json_label_file",
    "export_yolo_hbb_txt",
    "export_bbox_txt",
    "export_coco_bbox_json",
    "export_pascal_voc_xml",
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
    "filter_predictions_by_confidence",
    "parse_predictions_yolo_txt",
    "rename_prediction_class",
    "update_prediction_geometry_from_canvas_rect",
    "has_prediction_sidecar",
    "load_prediction_sidecar",
    "prediction_sidecar_path",
    "ALL_ERROR_TYPES",
    "ERROR_FILTER_ALL",
    "ErrorCase",
    "export_error_cases_csv",
    "filter_error_cases",
    "gt_attributes_for_case",
    "iou_xyxy",
    "match_gt_pred",
    "summarise_error_cases",
    "TileConfig",
    "TileRect",
    "annotations_in_tile",
    "boundary_crossing_annotations",
    "compute_tile_grid",
    "find_neighbor_tile_index",
    "find_tile_index_by_point",
    "global_to_tile",
    "tile_to_global",
    "PasteAdjustments",
    "PasteRecord",
    "export_paste_records_csv",
    "export_paste_records_json",
    "compute_dataset_stats",
    "export_stats_csv",
    "export_stats_json",
    "ProjectConfig",
    "load_project_config",
    "resolve_project_path",
    "resolve_project_root",
    "save_project_config",
    "has_autosave",
    "read_autosave",
    "remove_autosave",
    "write_autosave",
    "AnnotationBoxState",
    "AnnotationDocument",
    "AnnotationDocumentSnapshot",
    "FolderAnnotationScanResult",
    "FolderErrorAnalysisResult",
    "ImageAnnotationBundle",
    "PasteCandidateSession",
    "PasteDocument",
    "PasteEntryState",
    "annotations_from_legacy_rows",
    "class_mapping_from_object_list",
    "legacy_blocks_from_annotations",
    "SUPPORTED_IMAGE_SUFFIXES",
    "is_supported_image_path",
    "list_supported_images",
    "find_image_index",
    "load_image_annotation_bundle",
    "read_image_size",
    "scan_folder_annotation_records",
    "scan_folder_error_cases",
    "YoloModelHandle",
    "load_yolo_model",
    "run_yolo_model_inference",
]
