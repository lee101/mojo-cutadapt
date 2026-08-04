"""Useful Cutadapt adapter classes implemented on the Mojo aligner."""

from __future__ import annotations

from enum import IntFlag
from typing import Optional

from .align import Aligner, EndSkip, PrefixComparer, SuffixComparer


class Where(IntFlag):
    BACK = EndSkip.QUERY_START | EndSkip.QUERY_STOP | EndSkip.REFERENCE_END
    FRONT = EndSkip.QUERY_START | EndSkip.QUERY_STOP | EndSkip.REFERENCE_START
    PREFIX = EndSkip.QUERY_STOP
    SUFFIX = EndSkip.QUERY_START
    FRONT_NOT_INTERNAL = EndSkip.REFERENCE_START | EndSkip.QUERY_STOP
    BACK_NOT_INTERNAL = EndSkip.QUERY_START | EndSkip.REFERENCE_END
    ANYWHERE = EndSkip.SEMIGLOBAL


class SingleMatch:
    def __init__(self, astart: int, astop: int, rstart: int, rstop: int, score: int,
                 errors: int, adapter: "SingleAdapter", sequence: str):
        self.astart, self.astop = astart, astop
        self.rstart, self.rstop = rstart, rstop
        self.score, self.errors = score, errors
        self.adapter, self.sequence = adapter, sequence
        self.length = astop - astart

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(astart={self.astart}, astop={self.astop}, "
                f"rstart={self.rstart}, rstop={self.rstop}, score={self.score}, errors={self.errors})")

    def match_sequence(self) -> str:
        return self.sequence[self.rstart:self.rstop]

    def wildcards(self, wildcard_char: str = "N") -> str:
        return "".join(self.sequence[self.rstart + i] for i in range(self.length)
                       if self.adapter.sequence[self.astart + i] == wildcard_char
                       and self.rstart + i < len(self.sequence))


class RemoveBeforeMatch(SingleMatch):
    def rest(self) -> str:
        return self.sequence[:self.rstart]

    def remainder_interval(self) -> tuple[int, int]:
        return self.rstop, len(self.sequence)

    def retained_adapter_interval(self) -> tuple[int, int]:
        return self.rstart, len(self.sequence)

    def trim_slice(self) -> slice:
        return slice(self.rstop, None)

    def removed_sequence_length(self) -> int:
        return self.rstop


class RemoveAfterMatch(SingleMatch):
    def rest(self) -> str:
        return self.sequence[self.rstop:]

    def remainder_interval(self) -> tuple[int, int]:
        return 0, self.rstart

    def retained_adapter_interval(self) -> tuple[int, int]:
        return 0, self.rstop

    def trim_slice(self) -> slice:
        return slice(None, self.rstart)

    def adjacent_base(self) -> str:
        return self.sequence[self.rstart - 1:self.rstart]

    def removed_sequence_length(self) -> int:
        return len(self.sequence) - self.rstart


class SingleAdapter:
    description = "adapter with one component"
    allows_partial_matches = True
    _next_name = 1

    def __init__(self, sequence: str, max_errors: float = 0.1, min_overlap: int = 3,
                 read_wildcards: bool = False, adapter_wildcards: bool = True,
                 name: Optional[str] = None, indels: bool = True):
        self.name = str(SingleAdapter._next_name) if name is None else name
        if name is None:
            SingleAdapter._next_name += 1
        self.sequence = sequence.upper().replace("U", "T").replace("I", "N")
        if not self.sequence:
            raise ValueError("Adapter sequence is empty")
        if max_errors >= 1 and self.sequence.count("N") != len(self.sequence):
            max_errors /= len(self.sequence) - self.sequence.count("N")
        self.max_error_rate = max_errors
        self.min_overlap = min(min_overlap, len(self.sequence))
        self.adapter_wildcards = adapter_wildcards and not set(self.sequence) <= set("ACGT")
        self.read_wildcards = read_wildcards
        self.indels = indels
        self.aligner = self._aligner()

    def _make_aligner(self, flags: int) -> Aligner:
        return Aligner(self.sequence, self.max_error_rate, flags, self.adapter_wildcards,
                       self.read_wildcards, 1 if self.indels else 1_000_000, self.min_overlap)

    def _aligner(self) -> Aligner:
        raise NotImplementedError

    @property
    def effective_length(self) -> int:
        return self.aligner.effective_length

    def __len__(self) -> int:
        return len(self.sequence)

    def enable_debug(self) -> None:
        self.aligner.enable_debug()


class FrontAdapter(SingleAdapter):
    description = "regular 5'"

    def __init__(self, *args, **kwargs):
        self._force_anywhere = kwargs.pop("force_anywhere", False)
        super().__init__(*args, **kwargs)

    def _aligner(self) -> Aligner:
        return self._make_aligner(Where.ANYWHERE if self._force_anywhere else Where.FRONT)

    def match_to(self, sequence: str):
        hit = self.aligner.locate(sequence)
        return RemoveBeforeMatch(*hit, adapter=self, sequence=sequence) if hit else None

    def spec(self) -> str:
        return f"{self.sequence}..."


class BackAdapter(SingleAdapter):
    description = "regular 3'"

    def __init__(self, *args, **kwargs):
        self._force_anywhere = kwargs.pop("force_anywhere", False)
        super().__init__(*args, **kwargs)

    def _aligner(self) -> Aligner:
        return self._make_aligner(Where.ANYWHERE if self._force_anywhere else Where.BACK)

    def match_to(self, sequence: str):
        hit = self.aligner.locate(sequence)
        return RemoveAfterMatch(*hit, adapter=self, sequence=sequence) if hit else None

    def spec(self) -> str:
        return self.sequence


class AnywhereAdapter(SingleAdapter):
    description = "variable 5'/3'"

    def _aligner(self) -> Aligner:
        return self._make_aligner(Where.ANYWHERE)

    def match_to(self, sequence: str):
        hit = self.aligner.locate(sequence)
        if not hit:
            return None
        cls = RemoveBeforeMatch if hit[2] == 0 else RemoveAfterMatch
        return cls(*hit, adapter=self, sequence=sequence)

    def spec(self) -> str:
        return f"...{self.sequence}..."


class PrefixAdapter(FrontAdapter):
    description = "anchored 5'"
    allows_partial_matches = False

    def __init__(self, sequence: str, *args, **kwargs):
        kwargs["min_overlap"] = len(sequence)
        super().__init__(sequence, *args, **kwargs)

    def _aligner(self) -> Aligner:
        if not self.indels:
            return PrefixComparer(self.sequence, self.max_error_rate, self.adapter_wildcards,
                                  self.read_wildcards, self.min_overlap)
        return self._make_aligner(Where.PREFIX)

    def spec(self) -> str:
        return f"^{self.sequence}..."


class SuffixAdapter(BackAdapter):
    description = "anchored 3'"
    allows_partial_matches = False

    def __init__(self, sequence: str, *args, **kwargs):
        kwargs["min_overlap"] = len(sequence)
        super().__init__(sequence, *args, **kwargs)

    def _aligner(self) -> Aligner:
        if not self.indels:
            return SuffixComparer(self.sequence, self.max_error_rate, self.adapter_wildcards,
                                  self.read_wildcards, self.min_overlap)
        return self._make_aligner(Where.SUFFIX)

    def spec(self) -> str:
        return f"{self.sequence}$"
