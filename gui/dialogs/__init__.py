"""
Dialog windows for ImgLab and ImgBlending.
"""
from .class_mapping_dialog import ClassMappingDialog
from .error_analysis_dialog import ErrorAnalysisDialog
from .prediction_review_report_dialog import PredictionReviewReportDialog
from .statistics_dialog import StatisticsDialog
from .validation_dialog import ValidationDialog
from .paste_effects_dialog import PasteEffectsDialog
from .showlab_window import ShowlabWindow
from .saveimg_window import SaveimgWindow
from .savelab_window import SavelabWindow

__all__ = [
    'ClassMappingDialog',
    'ErrorAnalysisDialog',
    'PredictionReviewReportDialog',
    'StatisticsDialog',
    'ValidationDialog',
    'PasteEffectsDialog',
    'ShowlabWindow',
    'SaveimgWindow',
    'SavelabWindow',
]
