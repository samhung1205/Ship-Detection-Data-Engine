"""
Dialog windows for ImgLab and ImgBlending.
"""
from .class_mapping_dialog import ClassMappingDialog
from .error_analysis_dialog import ErrorAnalysisDialog
from .statistics_dialog import StatisticsDialog
from .input_window import InputWindow
from .showlab_window import ShowlabWindow
from .saveimg_window import SaveimgWindow
from .savelab_window import SavelabWindow

__all__ = [
    'ClassMappingDialog',
    'ErrorAnalysisDialog',
    'StatisticsDialog',
    'InputWindow',
    'ShowlabWindow',
    'SaveimgWindow',
    'SavelabWindow',
]
