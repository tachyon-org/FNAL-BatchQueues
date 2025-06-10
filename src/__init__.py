"""
fifebatch

This package provides the code necessary to parse and explore the
fifebatch raw data format (parquet). It includes functionality to
load, analyze, and visualize data fields from the fifebatch dataset.
"""
from .fifebatch import FifebatchField

__all__ = [
    'FifebatchField',
]