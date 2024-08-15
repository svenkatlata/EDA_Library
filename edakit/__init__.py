# edakit/__init__.py

"""
edakit
======

A library for advanced exploratory data analysis, including univariate, bivariate, multivariate, and time series analysis, outlier detection, feature engineering, distribution comparison, and dimensionality reduction.

Modules:
- analysis: Functions for various types of analysis.
"""

from analysis import (
    univariate_analysis,
    bivariate_analysis,
    multivariate_analysis,
    segmentation_analysis,
    time_series_analysis,
    outlier_detection,
    missing_data_analysis,
    feature_engineering_insights,
    distribution_comparison,
    dimensionality_reduction
)

__version__ = "0.1.0"
