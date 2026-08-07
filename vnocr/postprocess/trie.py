"""Syllable trie — the structural validity check (Design §5.1).

Vietnamese has only ~7k valid syllables, so a "syllable" outside that set is
almost certainly a frame error.  The trie answers two questions in O(len):

* :meth:`contains` — is this a valid syllable?
* :meth:`is_prefix` — could a valid syllable start like this? (used to prune
  the CTC prefix-beam in :mod:`vnocr.postprocess.decode`).

It also enumerates candidates within a small edit radius for the noisy-channel
corrector.  Case-insensitive by design: validity is a property of the lowercase
syllable, while the corrector restores the original casing.

Pure stdlib.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from ..charset.normalize import nfc

__all__ = ["SyllableTrie"]


class _Node:
    __slots__ = ("children", "is_word")

    def __init__(self) -> None:
        self.children: Dict[str, _Node] = {}
        self.is_word: bool = False


class SyllableTrie:
    def __init__(self, syllables: Optional[Iterable[str]] = None) -> None:
        self._root = _Node()
        self._size = 0
        if syllables:
            for s in syllables:
                self.add(s)

    def __len__(self) -> int:
        return self._size

    def __contains__(self, word: str) -> bool:
        return self.contains(word)

    def add(self, word: str) -> None:
        word = nfc(word).lower()
        if not word:
            return
        node = self._root
        for ch in word:
            node = node.children.setdefault(ch, _Node())
        if not node.is_word:
            node.is_word = True
            self._size += 1

    def _walk(self, prefix: str) -> Optional[_Node]:
        node = self._root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def contains(self, word: str) -> bool:
        node = self._walk(nfc(word).lower())
        return bool(node and node.is_word)

    def is_prefix(self, prefix: str) -> bool:
        """True if some valid syllable begins with ``prefix`` (empty ⇒ True)."""
        return self._walk(nfc(prefix).lower()) is not None

    def candidates_within(self, word: str, max_edits: int = 1) -> List[str]:
        """All valid syllables within Levenshtein ``max_edits`` of ``word``.

        Trie-guided edit search (Ukkonen-style rolling DP row): the candidate
        pool the noisy-channel model (Design §5.2) rescores.  Kept to radius
        1–2 on purpose — larger radii both slow down and admit noise.
        """
        word = nfc(word).lower()
        results: Set[str] = set()
        first_row = list(range(len(word) + 1))
        for ch, child in self._root.children.items():
            self._search(child, ch, ch, word, first_row, max_edits, results)
        return sorted(results)

    def _search(self, node: _Node, letter: str, path: str, word: str,
                prev_row: List[int], max_edits: int, out: Set[str]) -> None:
        cols = len(word) + 1
        cur_row = [prev_row[0] + 1]
        for j in range(1, cols):
            cur_row.append(min(
                cur_row[j - 1] + 1,                              # insertion
                prev_row[j] + 1,                                 # deletion
                prev_row[j - 1] + (word[j - 1] != letter),       # match / sub
            ))
        if node.is_word and cur_row[-1] <= max_edits:
            out.add(path)
        # Prune: descend only while some cell can still reach the budget.
        if min(cur_row) <= max_edits:
            for ch, child in node.children.items():
                self._search(child, ch, path + ch, word, cur_row, max_edits, out)
