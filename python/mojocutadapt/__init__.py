"""Mojo implementation of Cutadapt's adapter-alignment subset."""

from .align import Aligner, EndSkip, PrefixComparer, SuffixComparer
from .adapters import AnywhereAdapter, BackAdapter, FrontAdapter, PrefixAdapter, SuffixAdapter

__all__ = ["Aligner", "EndSkip", "PrefixComparer", "SuffixComparer", "FrontAdapter", "BackAdapter", "AnywhereAdapter", "PrefixAdapter", "SuffixAdapter"]

__version__ = "0.1.0"
