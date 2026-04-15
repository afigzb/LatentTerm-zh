"""
阶段一：全文词汇发现

VocabBuilderMixin 提供：
  build_index        — 扫描全文，构建词频表、邻字表、词表
  _cohesion          — 计算词的凝固度（PMI 变体）
  _filter_extensions — 清洗伪合成词（散字粘连 / 短语拼合）
  _find_all          — 在全文中找子串所有出现位置
  _vocab_after       — 从指定位置向后匹配词表词
  _vocab_before      — 从指定位置向前匹配词表词

所有阈值参数通过 TermExtractor 的类常量读取（self.VOCAB_COH_STRICT 等），
不在本文件中硬编码。
"""

import math
from collections import Counter, defaultdict
from typing import DefaultDict

from ._utils import _RE_CHINESE, _entropy, _clean_boundary


class VocabBuilderMixin:

    # ------------------------------------------------------------------ #
    # 阶段一：构建词表                                                       #
    # ------------------------------------------------------------------ #

    def build_index(self, text: str):
        self._text = text
        self._text_len = len(text)

        freq: Counter = Counter()
        left_nb: defaultdict = defaultdict(Counter)
        right_nb: defaultdict = defaultdict(Counter)

        for m in _RE_CHINESE.finditer(text):
            seq = m.group()
            n = len(seq)
            # 单字频率必须单独统计，否则凝固度分母默认为 1，
            # 导致"名字+泛用字"凝固度虚高而混入词表
            for ch in seq:
                freq[ch] += 1
            for i in range(n):
                for length in range(self.min_len,
                                    min(self.max_len + 1, n - i + 1)):
                    w = seq[i:i + length]
                    prev = freq[w]
                    freq[w] = prev + 1
                    # 仅对第二次及以后出现的 n-gram 统计邻字，
                    # 跳过 singleton 以减少 Counter 写入量
                    if prev >= 1:
                        if i > 0:
                            left_nb[w][seq[i - 1]] += 1
                        if i + length < n:
                            right_nb[w][seq[i + length]] += 1

        self._freq = freq
        self._left_nb = left_nb
        self._right_nb = right_nb

        coh_strict = self.VOCAB_COH_STRICT
        free_strict = self.VOCAB_FREE_STRICT
        coh_relaxed = self.VOCAB_COH_RELAXED
        free_relaxed = self.VOCAB_FREE_RELAXED

        vocab: dict = {}
        vocab_relaxed: dict = {}
        for w, f in freq.items():
            if f < 2 or len(w) < self.min_len:
                continue
            coh = self._cohesion(w)
            if coh < coh_relaxed:
                continue
            freedom = min(_entropy(left_nb[w]), _entropy(right_nb[w]))
            entry = {
                'freq': f,
                'cohesion': round(coh, 4),
                'freedom': round(freedom, 4),
            }
            if freedom >= free_relaxed:
                vocab_relaxed[w] = entry
            if coh >= coh_strict and freedom >= free_strict:
                vocab[w] = entry

        self._filter_extensions(vocab, freq)
        self._filter_extensions(vocab_relaxed, freq, lenient=True)
        self._vocab = vocab
        self._vocab_relaxed = vocab_relaxed
        self._find_cache: dict[str, list[int]] = {}

        char_idx: DefaultDict[str, set[str]] = defaultdict(set)
        bigram_idx: DefaultDict[str, set[str]] = defaultdict(set)
        for w in vocab:
            for ch in set(w):
                char_idx[ch].add(w)
            for i in range(len(w) - 1):
                bigram_idx[w[i:i + 2]].add(w)
        self._char_index = dict(char_idx)
        self._bigram_index = dict(bigram_idx)

        self._built = True

    def _cohesion(self, word: str) -> float:
        """词的凝固度：所有切分点上 PMI 的最小值"""
        f = self._freq.get(word, 0)
        if f == 0 or len(word) < 2:
            return 0.0
        worst = float('inf')
        for i in range(1, len(word)):
            lf = self._freq.get(word[:i], 1)
            rf = self._freq.get(word[i:], 1)
            worst = min(worst, f / math.sqrt(lf * rf))
        return worst

    def _filter_extensions(self, vocab: dict, freq: Counter,
                           lenient: bool = False):
        """清洗"伪合成词"：散字粘连 / 短语拼合。
        阈值从类常量 FILTER_COH_* / FILTER_DOM_* 读取。
        """
        coh_threshold = self.FILTER_COH_LENIENT if lenient else self.FILTER_COH_STRICT
        dom_threshold = self.FILTER_DOM_LENIENT if lenient else self.FILTER_DOM_STRICT

        to_remove: set = set()
        for w in list(vocab):
            if len(w) < 3:
                continue
            f_w = vocab[w]['freq']

            prefix = w[:-1]
            if prefix in vocab:
                pair = w[-2:]
                if pair not in vocab:
                    coh = f_w / math.sqrt(
                        max(freq.get(prefix, 1) * freq.get(w[-1], 1), 1))
                    if coh < coh_threshold:
                        to_remove.add(w)
                        continue

            suffix = w[1:]
            if suffix in vocab:
                pair = w[:2]
                if pair not in vocab:
                    coh = f_w / math.sqrt(
                        max(freq.get(w[0], 1) * freq.get(suffix, 1), 1))
                    if coh < coh_threshold:
                        to_remove.add(w)
                        continue

            if len(w) >= 4:
                for sp in range(2, len(w) - 1):
                    head, tail = w[:sp], w[sp:]
                    if head in vocab and tail in vocab:
                        h_dom = freq.get(head, 1) / max(f_w, 1)
                        t_dom = freq.get(tail, 1) / max(f_w, 1)
                        if min(h_dom, t_dom) > dom_threshold:
                            to_remove.add(w)
                            break

        for w in to_remove:
            del vocab[w]

    # ------------------------------------------------------------------ #
    # 辅助：文本定位工具                                                     #
    # ------------------------------------------------------------------ #

    def _find_all(self, s: str) -> list[int]:
        """返回子串 s 在全文中所有出现的起始位置（带缓存）"""
        cached = self._find_cache.get(s)
        if cached is not None:
            return cached
        out, start = [], 0
        while True:
            p = self._text.find(s, start)
            if p < 0:
                break
            out.append(p)
            start = p + 1
        self._find_cache[s] = out
        return out

    def _vocab_after(self, pos: int, vocab: dict) -> list[str]:
        """从 pos 位置开始，返回所有能匹配到的词表词"""
        if pos >= self._text_len or not ('\u4e00' <= self._text[pos] <= '\u9fff'):
            return []
        hits = []
        for length in range(self.min_len, self.max_len + 1):
            end = pos + length
            if end > self._text_len:
                break
            w = self._text[pos:end]
            if w in vocab and _clean_boundary(w):
                hits.append(w)
        return hits

    def _vocab_before(self, pos: int, vocab: dict) -> list[str]:
        """以 pos 位置结尾，返回所有能匹配到的词表词"""
        if pos < self.min_len or not ('\u4e00' <= self._text[pos - 1] <= '\u9fff'):
            return []
        hits = []
        for length in range(self.min_len, self.max_len + 1):
            start = pos - length
            if start < 0:
                continue
            w = self._text[start:pos]
            if w in vocab and _clean_boundary(w):
                hits.append(w)
        return hits
