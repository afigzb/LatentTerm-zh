"""
阶段一：全文词汇发现（两步法：PMI+自由度 过滤 → Trie 分词 → 干净频数）

VocabBuilderMixin 提供：
  build_index           — 扫描全文，构建词频表、邻字表、词表、段落索引
  _build_segment_index  — 构建段落级词分布索引（供共主题策略使用）
  _cohesion             — 计算词的凝固度（log-PMI，对词长天然归一化）
  _filter_extensions    — 清洗伪合成词（散字粘连 / 短语拼合）
  _find_all             — 在全文中找子串所有出现位置
  _vocab_after          — 从指定位置向后匹配词表词
  _vocab_before         — 从指定位置向前匹配词表词

核心改进（参考 bojone/word-discovery）：
  传统单步法直接对所有子串计频，高频长词（如"张无忌"）会污染子串
  （如"张无""无忌"）的频数和邻字分布。两步法：
    Pass 1  原始 n-gram 统计，同时收集邻字。
            用 PMI（凝固度）+ 自由度（信息熵）双重过滤出"结实"的 n-gram。
            "物系魂兽"右边永远跟着固定字 → 自由度≈0 → 淘汰。
            "灭绝师"右边永远是"太" → 自由度≈0 → 淘汰。
    Pass 2  用 Trie 长词优先分词对原文切分，从切分结果统计干净频数。
            "植物系魂兽"被整体切走 → "物系魂兽"独立出现次数归零 → 出局。
            "植物系"和"魂兽"在别处也出现 → 保持健康频数 → 留存。

所有阈值参数通过 TermExtractor 的类常量读取（self.VOCAB_COH_STRICT 等），
不在本文件中硬编码。
"""

import math
from collections import Counter, defaultdict
from typing import DefaultDict

from ._utils import (
    _RE_CHINESE, _entropy, _clean_boundary,
    _LONG_PHRASE_INTERIOR, _DIALOGUE_TRAIL,
)
from ._pattern_miner import PatternMiner, can_decompose_by_l1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trie 树 + 正向最长匹配
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _SimpleTrie:
    """Trie 树，正向最长匹配分词（长词优先）。"""

    def __init__(self):
        self._root: dict = {}
        self._END = True

    def add_word(self, word: str):
        node = self._root
        for c in word:
            node = node.setdefault(c, {})
        node[self._END] = True

    def tokenize(self, text: str) -> list[str]:
        """正向最长匹配：每次从当前位置起取最长能匹配的词。"""
        result: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            node = self._root
            last_end = i + 1          # 默认：单字
            j = i
            while j < n and text[j] in node:
                node = node[text[j]]
                j += 1
                if self._END in node:
                    last_end = j      # 记住最长匹配
            result.append(text[i:last_end])
            i = last_end
        return result


class VocabBuilderMixin:

    # ------------------------------------------------------------------ #
    # 阶段一：构建词表（两步法）                                               #
    # ------------------------------------------------------------------ #

    def build_index(self, text: str):
        self._text = text
        self._text_len = len(text)

        # ── Pass 1：原始 n-gram 统计 + 邻字收集 ──
        raw_freq: Counter = Counter()
        raw_left: defaultdict = defaultdict(Counter)
        raw_right: defaultdict = defaultdict(Counter)
        total_chars = 0

        for m in _RE_CHINESE.finditer(text):
            seq = m.group()
            n = len(seq)
            total_chars += n
            for ch in seq:
                raw_freq[ch] += 1
            for i in range(n):
                for length in range(self.min_len,
                                    min(self.max_len + 1, n - i + 1)):
                    w = seq[i:i + length]
                    raw_freq[w] += 1
                    if i > 0:
                        raw_left[w][seq[i - 1]] += 1
                    if i + length < n:
                        raw_right[w][seq[i + length]] += 1

        self._freq = raw_freq
        self._total_chars = total_chars

        # ── PMI + 自由度 双重过滤 → 结实 n-gram 集 ──
        # "手持倚"右边永远是"天" → 自由度=0 → 不进 Trie
        # "灭绝师"右边永远是"太" → 自由度=0 → 不进 Trie
        solid: set[str] = set()
        pmi_values: dict[str, float] = {}
        solid_base = self.SOLID_PMI_MIN
        solid_free = self.SOLID_FREE_MIN

        candidates = [
            (w, f) for w, f in raw_freq.items()
            if f >= 2 and len(w) >= self.min_len
        ]
        candidates.sort(key=lambda x: len(x[0]))

        for w, f in candidates:
            wlen = len(w)
            if not _clean_boundary(w):
                continue
            if wlen >= 4 and any(c in _LONG_PHRASE_INTERIOR
                                for c in w[1:-1]):
                continue

            # 凝固度（log-PMI）
            coh = self._cohesion(w)
            solid_threshold = solid_base + max(wlen - 2, 0)
            if coh < solid_threshold:
                continue

            # 自由度预过滤：如果两侧都有邻字观测，则要求两侧都有起码的多样性
            left_cnt = sum(raw_left[w].values())
            right_cnt = sum(raw_right[w].values())
            if left_cnt > 0 and right_cnt > 0:
                raw_freedom = min(_entropy(raw_left[w]),
                                  _entropy(raw_right[w]))
                if raw_freedom < solid_free:
                    continue

            # 名字+对白动词不应作为分词单元
            if wlen >= 3 and w[-1] in _DIALOGUE_TRAIL:
                prefix = w[:-1]
                if len(prefix) >= self.min_len and prefix in solid:
                    continue

            solid.add(w)
            pmi_values[w] = coh

        # ── Trie 长词优先分词 → 干净频数 ──
        trie = _SimpleTrie()
        for w in solid:
            trie.add_word(w)

        clean_freq: Counter = Counter()
        left_nb: defaultdict = defaultdict(Counter)
        right_nb: defaultdict = defaultdict(Counter)

        for m in _RE_CHINESE.finditer(text):
            seq = m.group()
            slen = len(seq)
            tokens = trie.tokenize(seq)
            pos = 0
            for tok in tokens:
                tlen = len(tok)
                if tlen >= self.min_len:
                    clean_freq[tok] += 1
                    if pos > 0:
                        left_nb[tok][seq[pos - 1]] += 1
                    if pos + tlen < slen:
                        right_nb[tok][seq[pos + tlen]] += 1
                pos += tlen

        self._left_nb = left_nb
        self._right_nb = right_nb

        # ── 构建词表 ──
        coh_strict = self.VOCAB_COH_STRICT
        free_strict = self.VOCAB_FREE_STRICT
        coh_relaxed = self.VOCAB_COH_RELAXED
        free_relaxed = self.VOCAB_FREE_RELAXED
        coh_rescue = coh_strict + 2.0

        vocab: dict = {}
        vocab_relaxed: dict = {}

        for w in solid:
            f = clean_freq.get(w, 0)
            if f < 2:
                continue
            coh = pmi_values[w]
            if coh < coh_relaxed:
                continue

            freedom = min(_entropy(left_nb[w]), _entropy(right_nb[w]))

            if (len(w) >= 3 and w[-1] in _DIALOGUE_TRAIL
                    and w[:-1] in solid and freedom < free_strict):
                continue

            entry = {
                'freq': f,
                'cohesion': round(coh, 4),
                'freedom': round(freedom, 4),
            }
            if freedom >= free_relaxed:
                vocab_relaxed[w] = entry
            if (coh >= coh_strict
                    and freedom >= free_strict):
                vocab[w] = entry
            elif (coh >= coh_rescue
                  and freedom >= free_relaxed):
                vocab[w] = entry
            elif (freedom >= free_strict * 2
                  and coh >= coh_relaxed):
                vocab[w] = entry

        self._filter_extensions(vocab, raw_freq)
        self._filter_extensions(vocab_relaxed, raw_freq, lenient=True)
        self._vocab = vocab
        self._vocab_relaxed = vocab_relaxed
        self._find_cache: dict[str, list[int]] = {}

        # ── 通道 B：模板狙击，产出 L2 候选池 ──
        miner = PatternMiner(min_len=self.min_len, max_len=self.max_len)
        pattern_hits = miner.mine(text)

        # ── 合并 L1 (vocab_relaxed) + L2 (pattern_hits) → candidates ──
        self._candidates = self._merge_candidates(
            vocab_relaxed, pattern_hits, raw_freq)
        self._pattern_miner = miner

        # ── 索引基于候选池重建（L1 + L2 都能被字符/构词策略发现）──
        char_idx: DefaultDict[str, set[str]] = defaultdict(set)
        bigram_idx: DefaultDict[str, set[str]] = defaultdict(set)
        for w in self._candidates:
            for ch in set(w):
                char_idx[ch].add(w)
            for i in range(len(w) - 1):
                bigram_idx[w[i:i + 2]].add(w)
        self._char_index = dict(char_idx)
        self._bigram_index = dict(bigram_idx)

        self._build_segment_index(self._candidates)
        self._built = True

    # ------------------------------------------------------------------ #
    # 候选池合并：L1 (统计通道) + L2 (模板通道)                                #
    # ------------------------------------------------------------------ #

    def _merge_candidates(self, vocab_relaxed: dict,
                          pattern_hits: dict,
                          raw_freq: Counter) -> dict:
        """把统计词表和模板命中合并成统一候选池。

        每个候选的字段：
          freq       — 频数（L1 用 clean_freq，L2 用正则命中数或 raw_freq）
          tier       — 'L1' / 'L2'
          origins    — {'A'} / {'B1', 'B2', ...} / {'A', 'B1', ...}
          templates  — 命中模板 id 集合
          type       — person/place/creature/skill/group/misc/None
          evidence   — [(pos, tmpl_id), ...]
          cohesion   — L1 有，L2 为 0.0
          freedom    — L1 有，L2 为 0.0
        """
        candidates: dict = {}

        # L1 先入池
        for w, info in vocab_relaxed.items():
            candidates[w] = {
                'freq': info['freq'],
                'tier': 'L1',
                'origins': {'A'},
                'templates': set(),
                'type': None,
                'evidence': [],
                'cohesion': info['cohesion'],
                'freedom': info['freedom'],
            }

        # L1 词集（供纯 L2 候选做"短语拼凑"反验）
        l1_words = set(vocab_relaxed.keys())

        # L2 合并
        for w, meta in pattern_hits.items():
            if w in candidates:
                # 已在 L1 → 补类型 + 模板证据
                entry = candidates[w]
                entry['origins'].update(meta['origins'])
                entry['templates'].update(meta['templates'])
                if entry['type'] is None:
                    entry['type'] = meta['type']
                # 合并证据（截断）
                budget = 5 - len(entry['evidence'])
                if budget > 0:
                    entry['evidence'].extend(meta['evidence'][:budget])
            else:
                # 纯 L2 候选 → L1 词表反验：能被完全切分 = 短语拼凑，丢弃
                # （"真空地带" = "真空"+"地带"；"一尊强者" 已被前置规则挡掉）
                if can_decompose_by_l1(w, l1_words):
                    continue
                rf = raw_freq.get(w, 0)
                freq_guess = max(rf, meta['freq'])
                if freq_guess < 1:
                    continue
                candidates[w] = {
                    'freq': freq_guess,
                    'tier': 'L2',
                    'origins': set(meta['origins']),
                    'templates': set(meta['templates']),
                    'type': meta['type'],
                    'evidence': list(meta['evidence'][:5]),
                    'cohesion': 0.0,
                    'freedom': 0.0,
                }

        return candidates

    # ------------------------------------------------------------------ #
    # 段落级索引（供共主题策略使用）                                            #
    # ------------------------------------------------------------------ #

    _SEG_SIZE = 500

    def _build_segment_index(self, vocab: dict):
        """扫描全文，记录每个词表词出现在哪些段落片段中。"""
        text = self._text
        seg_size = self._SEG_SIZE
        n_segs = max(1, (self._text_len + seg_size - 1) // seg_size)

        word_segs: dict[str, set[int]] = defaultdict(set)
        min_l, max_l = self.min_len, self.max_len

        for m in _RE_CHINESE.finditer(text):
            seq = m.group()
            base = m.start()
            n = len(seq)
            for i in range(n):
                sid = (base + i) // seg_size
                for length in range(min_l, min(max_l + 1, n - i + 1)):
                    w = seq[i:i + length]
                    if w in vocab:
                        word_segs[w].add(sid)

        self._word_segs: dict[str, set[int]] = dict(word_segs)
        self._n_segments: int = n_segs

    def _cohesion(self, word: str) -> float:
        """词的凝固度：所有切分点上 log-PMI 的最小值。

        log-PMI = log(N · freq(word) / (freq(L) · freq(R)))
        对词长天然归一化（随机基线 ≈ 0），无需长度折扣。
        """
        f = self._freq.get(word, 0)
        if f == 0 or len(word) < 2:
            return 0.0
        total = self._total_chars
        worst = float('inf')
        for i in range(1, len(word)):
            lf = self._freq.get(word[:i], 1)
            rf = self._freq.get(word[i:], 1)
            pmi = math.log(total * f / max(lf * rf, 1))
            worst = min(worst, pmi)
        return worst

    def _filter_extensions(self, vocab: dict, freq: Counter,
                           lenient: bool = False):
        """清洗"伪合成词"：散字粘连 / 短语拼合。

        边界 log-PMI：用词的干净频数作分子，原始子串频数作分母，
        检验末字/首字是否真正属于该词。
        """
        coh_threshold = (self.FILTER_COH_LENIENT if lenient
                         else self.FILTER_COH_STRICT)
        dom_threshold = (self.FILTER_DOM_LENIENT if lenient
                         else self.FILTER_DOM_STRICT)
        free_protect = self.VOCAB_FREE_STRICT
        total = self._total_chars

        to_remove: set = set()
        for w in list(vocab):
            if len(w) < 3:
                continue
            if len(w) <= 4 and vocab[w]['freedom'] >= free_protect:
                continue
            f_w = vocab[w]['freq']

            prefix = w[:-1]
            if prefix in vocab:
                pair = w[-2:]
                if pair not in vocab:
                    lf = freq.get(prefix, 1)
                    rf = freq.get(w[-1], 1)
                    coh = math.log(total * f_w / max(lf * rf, 1))
                    if coh < coh_threshold:
                        to_remove.add(w)
                        continue

            suffix = w[1:]
            if suffix in vocab:
                pair = w[:2]
                if pair not in vocab:
                    lf = freq.get(w[0], 1)
                    rf = freq.get(suffix, 1)
                    coh = math.log(total * f_w / max(lf * rf, 1))
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
