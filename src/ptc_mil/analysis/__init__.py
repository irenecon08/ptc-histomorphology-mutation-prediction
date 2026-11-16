"""Shared access to data, models and output folders for the scripts in ``analyses/``.

The analysis scripts are run directly, e.g. ``python analyses/tiic/tiic_correlation.py``.
They find the project configuration through two environment variables, which default
to the files in ``configs/``:

    PTC_MIL_PATHS       paths YAML       (default: configs/paths.yaml)
    PTC_MIL_EXPERIMENT  experiment YAML  (default: configs/experiment.yaml)
"""

from ptc_mil.analysis.context import AnalysisContext, load_context

__all__ = ["AnalysisContext", "load_context"]
