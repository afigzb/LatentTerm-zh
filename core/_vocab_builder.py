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
    Pass 1  原始 n-gram 统计，得到 PMI/自由度过滤后的"结实"n-gram 集。
            "物系魂兽"右边永远跟着固定字 → 自由度≈0 → 淘汰。
            "灭绝师"右边永远是"太" → 自由度≈0 → 淘汰。
    Pass 2  用 Trie 长词优先分词对原文切分，从切分结果统计干净频数。
            "植物系魂兽"被整体切走 → "物系魂兽"独立出现次数归零 → 出局。
            "植物系"和"魂兽"在别处也出现 → 保持健康频数 → 留存。

Pass 1 内部进一步分两步（A1 + A3 优化，输出与朴素实现完全等价）：
    Pass 1a  Apriori 频率剪枝：按长度增量统计 n-gram。
             长度 k 的 n-gram 仅在其 (k-1)-前缀和 (k-1)-后缀的频数都 >= 2
             时才登记。由 Apriori 性质：freq(k-gram)>=2 ⇒ 两端 (k-1)-gram
             freq 都 >=2，故剪枝不丢任何下游会用到的词
             （候选过滤要求 f>=2；_cohesion 对缺失子串以默认值 1 兜底）。
             unique 数从 30M+ 降到 1–3M。
    PMI 粗过滤  仅依赖频数，先筛出"通过凝固度"的候选 pmi_set。
    Pass 1b  邻字延迟收集：仅为 pmi_set 收集左右邻字，用 Aho-Corasick
             对各中文段扫一次，命中 (end_idx, w) 即取 seq[start-1] /
             seq[end_idx+1]，避免为 30M+ 子串无谓地分配 Counter，也省掉
             O(N·max_len) 的 Python 三重循环。
    自由度 + 对白动词 终筛 → solid。

所有阈值参数通过 TermExtractor 的类常量读取（self.VOCAB_COH_STRICT 等），
不在本文件中硬编码。
"""

import math
from collections import Counter, defaultdict
from typing import DefaultDict

import ahocorasick

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

        # ── Pass 1a：Apriori 频率统计（不收集邻字）──
        # 缓存中文段，供 Pass 1a / Pass 1b 复用，避免重复跑正则
        segments: list[str] = [m.group()
                               for m in _RE_CHINESE.finditer(text)]

        raw_freq: Counter = Counter()
        total_chars = 0

        # 长度 1：所有汉字一次性吞下（Counter.update 为 C 实现）
        for seq in segments:
            total_chars += len(seq)
            raw_freq.update(seq)

        # 长度 min_len..max_len：min_len 直接计数；更长用 Apriori 剪枝
        # Apriori：若 freq(k-gram) >= 2，则两端 (k-1)-gram 的 freq 都 >= 2。
        # 下游候选过滤要求 f >= 2，剪掉 freq<2 的 n-gram 不影响最终结果
        # （_cohesion 对缺失子串以默认值 1 兜底，与原实现等价）。
        for k in range(self.min_len, self.max_len + 1):
            if k == self.min_len:
                for seq in segments:
                    n = len(seq)
                    for i in range(n - k + 1):
                        raw_freq[seq[i:i + k]] += 1
            else:
                for seq in segments:
                    n = len(seq)
                    for i in range(n - k + 1):
                        if raw_freq.get(seq[i:i + k - 1], 0) < 2:
                            continue
                        if raw_freq.get(seq[i + 1:i + k], 0) < 2:
                            continue
                        raw_freq[seq[i:i + k]] += 1

        self._freq = raw_freq
        self._total_chars = total_chars

        # ── PMI 粗过滤（不读邻字，省下 ~30M 个 Counter 实例）──
        # "手持倚"右边永远是"天" → 自由度=0 → 后续会被淘汰
        # "灭绝师"右边永远是"太" → 自由度=0 → 后续会被淘汰
        solid: set[str] = set()
        pmi_values: dict[str, float] = {}
        solid_base = self.SOLID_PMI_MIN
        solid_free = self.SOLID_FREE_MIN

        candidates = [
            (w, f) for w, f in raw_freq.items()
            if f >= 2 and len(w) >= self.min_len
        ]
        candidates.sort(key=lambda x: len(x[0]))

        # 第一轮：仅做 PMI（凝固度）筛选，邻字延后收集
        pmi_passed: list[tuple[str, int]] = []
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

            pmi_passed.append((w, wlen))
            pmi_values[w] = coh

        pmi_set: set[str] = {w for w, _ in pmi_passed}

        # ── Pass 1b：仅为 PMI 候选收集邻字（Aho-Corasick 驱动）──
        # 邻字 Counter 数量仍锁定在 ~|pmi_set|（通常 5万–50万），
        # 对应内存从 ~10GB 降到 < 200MB；同时把原来 O(N·max_len) 的
        # Python 三重循环替换成一次 C 级 AC 多模式匹配：
        #   • 以 pmi_set 建自动机，对每个中文段调 iter(seq)
        #   • 命中 (end_idx, w) → start = end_idx - len(w) + 1
        #     与原 (i, length) 命中一一对应，命中集合完全等价
        #   • 邻字仍按段内取（start>0 / end_idx+1<n），跨段不相邻，
        #     与原三重循环边界语义一致
        raw_left: defaultdict = defaultdict(Counter)
        raw_right: defaultdict = defaultdict(Counter)

        if pmi_set:
            nb_automaton = ahocorasick.Automaton()
            for w in pmi_set:
                nb_automaton.add_word(w, w)
            nb_automaton.make_automaton()

            for seq in segments:
                n = len(seq)
                for end_idx, w in nb_automaton.iter(seq):
                    start = end_idx - len(w) + 1
                    if start > 0:
                        raw_left[w][seq[start - 1]] += 1
                    if end_idx + 1 < n:
                        raw_right[w][seq[end_idx + 1]] += 1

            del nb_automaton

        del segments  # ~text 大小的副本，及时释放

        # 第二轮：自由度 + 名字+对白动词 终筛
        # 按长度顺序处理，保证 prefix in solid 检查时短词已先入 solid
        for w, wlen in pmi_passed:
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

        # ── 跨策略预计算缓存（性能加速，行为完全等价）──
        # _clean_set      候选词的 _clean_boundary 结果一次算定
        # _log_freq_p1/p2 log2(freq+1) / log2(freq+2) 一次算定
        # _vocab_starts   位置 → 该位置开头的 clean vocab 词列表（_vocab_after 用）
        # _vocab_ends     位置 → 该位置结尾的 clean vocab 词列表（_vocab_before 用）
        # _word_segs      词 → 出现段落 id 集合（co_topic 用）
        # _seg_words      段落 → 出现在该段的候选词集合（co_topic 用）
        # 所有索引由一次 Aho-Corasick 全文扫描产出，原 _build_segment_index +
        # 逐位置 vocab 切片在此被一次性吃掉。
        self._build_strategy_caches()
        self._built = True

    # ------------------------------------------------------------------ #
    # 跨策略预计算缓存（build_index 末尾调用，extract 时按需扩充）              #
    # ------------------------------------------------------------------ #

    def _build_strategy_caches(self):
        """对 self._candidates 一次性预算所有跨策略缓存。

        三步：
          1. _clean_set / _log_freq_p1/p2  ← 直接遍历 candidates
          2. _vocab_starts / _vocab_ends / _word_segs / _n_segments
             ← 一次 Aho-Corasick 扫描全文同时产出（_build_ac_index）
          3. _seg_words ← 反转 _word_segs

        策略代码的语义保持不变：
          • clean_set 与原 _clean_boundary 在 candidates 上一一对应
          • log_freq_p1/p2 与 math.log2(freq+1/2) 数值完全一致
          • word_segs 与原 _build_segment_index 输出等价（同样不过 clean_set）
          • vocab_starts/ends 与原 _vocab_after/before 中
            `if w in vocab and w in clean_set` 的过滤结果等价
          • seg_words 与 word_segs 互为倒置，无信息损失
        """
        cands = self._candidates

        clean_set: set[str] = set()
        log1: dict[str, float] = {}
        log2_: dict[str, float] = {}
        for w, info in cands.items():
            if _clean_boundary(w):
                clean_set.add(w)
            f = info['freq']
            log1[w] = math.log2(f + 1)
            log2_[w] = math.log2(f + 2)
        self._clean_set = clean_set
        self._log_freq_p1 = log1
        self._log_freq_p2 = log2_

        self._build_ac_index()

        seg_words: DefaultDict[int, set] = defaultdict(set)
        for w, segs in self._word_segs.items():
            for sid in segs:
                seg_words[sid].add(w)
        self._seg_words = dict(seg_words)

    def _build_ac_index(self):
        """用 Aho-Corasick 对全文做一次多模式匹配，同时产出三张索引：

          _vocab_starts: dict[int, list[str]]
              position → 在该位置开头、且通过 _clean_boundary 的候选词列表
              （_vocab_after 的 O(1) 查表后端）

          _vocab_ends: dict[int, list[str]]
              exclusive end → 在该位置结尾、且通过 _clean_boundary 的候选词
              （_vocab_before 的 O(1) 查表后端，end = start + len(word)）

          _word_segs: dict[str, set[int]]
              候选词 → 出现段落 id 集合（co_topic 用）
              这里**不**应用 clean_set 过滤，与原 _build_segment_index 等价

        以及 _n_segments 段落总数。

        替代了原 _build_segment_index 的全文（i, length）枚举（O(N·max_len)
        Python 循环）和策略侧 _vocab_after/before 的逐长度切片+dict 查询。
        """
        text = self._text
        seg_size = self._SEG_SIZE
        self._n_segments = max(
            1, (self._text_len + seg_size - 1) // seg_size)

        clean_set = self._clean_set
        cands = self._candidates

        vocab_starts: DefaultDict[int, list] = defaultdict(list)
        vocab_ends: DefaultDict[int, list] = defaultdict(list)
        word_segs: DefaultDict[str, set] = defaultdict(set)

        if not cands:
            self._vocab_starts = {}
            self._vocab_ends = {}
            self._word_segs = {}
            return

        # 自动机里只塞需要的最小集合（candidates 的并集），匹配命中即可
        # 同时分流到三张索引。
        automaton = ahocorasick.Automaton()
        for w in cands:
            automaton.add_word(w, w)
        automaton.make_automaton()

        # iter 给的是 (end_idx, value)，end_idx 是命中末字的 inclusive 下标
        for end_idx, word in automaton.iter(text):
            start = end_idx - len(word) + 1
            word_segs[word].add(start // seg_size)
            if word in clean_set:
                vocab_starts[start].append(word)
                vocab_ends[start + len(word)].append(word)

        self._vocab_starts = dict(vocab_starts)
        self._vocab_ends = dict(vocab_ends)
        self._word_segs = dict(word_segs)

    def _ensure_strategy_caches(self, vocab: dict):
        """extract() 在合并 channel-C extras 后调用，把新增词补进缓存。

        约束：候选集合不能因为缓存缺失而变小，所以只能"扩充"，不能"重建"。

        除了原本的 _clean_set / _log_freq_p1/p2，还需把 extras 在全文中的
        所有出现位置增量插入 _vocab_starts / _vocab_ends，否则
        _vocab_after / _vocab_before 看不到这些词，context_pattern /
        substitution / cooccurrence 会丢候选。

        _word_segs 不更新：原 _build_segment_index 也只在 build 期固化，
        co_topic 本来就对 channel-C extras 不可见，行为完全一致。
        """
        clean_set = self._clean_set
        log1 = self._log_freq_p1
        log2_ = self._log_freq_p2
        text = self._text
        vocab_starts = self._vocab_starts
        vocab_ends = self._vocab_ends

        for w, info in vocab.items():
            if w in log1:
                continue
            f = info['freq']
            log1[w] = math.log2(f + 1)
            log2_[w] = math.log2(f + 2)
            if not _clean_boundary(w):
                continue
            clean_set.add(w)
            wlen = len(w)
            start = 0
            while True:
                p = text.find(w, start)
                if p < 0:
                    break
                vocab_starts.setdefault(p, []).append(w)
                vocab_ends.setdefault(p + wlen, []).append(w)
                start = p + 1

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

    # _build_segment_index 已并入 _build_ac_index：AC 全文扫一遍即可同时
    # 产出 _word_segs / _vocab_starts / _vocab_ends，避免 O(N·max_len) 的
    # Python 双循环。

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
        """从 pos 位置开始，返回所有能匹配到的词表词。

        由 _build_ac_index 预算的 _vocab_starts 提供 O(1) 查表，candidates
        全集已在 build 期一次性 Aho-Corasick 扫出；channel-C extras 由
        _ensure_strategy_caches 在 extract 时增量补入。
        """
        hits = self._vocab_starts.get(pos)
        if not hits:
            return []
        # vocab 可能等于 _candidates 或 _candidates ∪ extras；过滤一遍兜底。
        # 索引侧已经做过 _clean_set 过滤，这里不再二次校验。
        return [w for w in hits if w in vocab]

    def _vocab_before(self, pos: int, vocab: dict) -> list[str]:
        """以 pos 位置（exclusive）结尾，返回所有能匹配到的词表词。"""
        hits = self._vocab_ends.get(pos)
        if not hits:
            return []
        return [w for w in hits if w in vocab]
