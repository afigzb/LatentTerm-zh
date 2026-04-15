"""
阶段二：四策略并行候选提取

StrategiesMixin 提供：
  _strategy_char_overlap    — 字符包含：从词表中找包含关键词字符的词，按位置加权
  _strategy_context_pattern — 上下文模式引导：学习种子词的上下文模板，用模板发现新词
  _strategy_cooccurrence    — 共现近邻：统计在种子词附近高频出现的词（Lift 比率）
  _strategy_morpheme        — 构词结构相似：找与关键词共享核心语素且处于相同位置的词

所有策略统一接收 vocab 参数、统一返回 StrategyResult。
"""

import math
from collections import Counter

from ._utils import _RE_CHINESE, _RE_PURE_CHINESE, _clean_boundary, StrategyResult


class StrategiesMixin:

    def _strategy_char_overlap(self, keyword: str,
                               vocab: dict) -> StrategyResult:
        """字符包含：按字符重叠比例和位置匹配度打分"""
        anchors = set(keyword)
        kw_len = len(keyword)
        kw_positions: dict[str, float] = {}
        for i, ch in enumerate(keyword):
            kw_positions[ch] = i / max(kw_len - 1, 1)

        candidates: set[str] = set()
        for ch in anchors:
            candidates |= self._char_index.get(ch, set())

        scores: dict[str, float] = {}
        for word in candidates:
            if not _clean_boundary(word):
                continue
            overlap = anchors & set(word)
            char_ratio = len(overlap) / len(anchors)
            w_len = len(word)

            pos_match = 0.0
            for c in overlap:
                kw_rel = kw_positions[c]
                w_rel = word.index(c) / max(w_len - 1, 1)
                pos_match += 1.0 - abs(kw_rel - w_rel)
            pos_match /= len(overlap)

            freq_w = math.log2(vocab[word]['freq'] + 1)
            scores[word] = char_ratio * pos_match * freq_w
        return StrategyResult(scores)

    def _strategy_context_pattern(self, seeds: list[str], vocab: dict,
                                  _max_tpl: int = 15,
                                  _max_pos: int = 500,
                                  _max_gap: int = 4) -> StrategyResult:
        """上下文模式引导：学习种子词左右模板，在全文中匹配新词"""
        left_tpl: Counter = Counter()
        right_tpl: Counter = Counter()
        for seed in seeds:
            positions = self._find_all(seed)
            for pos in positions:
                end = pos + len(seed)
                for cl in range(2, 5):
                    if pos >= cl:
                        lc = self._text[pos - cl:pos]
                        if _RE_PURE_CHINESE.match(lc):
                            left_tpl[lc] += 1
                    if end + cl <= self._text_len:
                        rc = self._text[end:end + cl]
                        if _RE_PURE_CHINESE.match(rc):
                            right_tpl[rc] += 1

        min_pf = 2
        lt = {p: c for p, c in left_tpl.items() if c >= min_pf}
        rt = {p: c for p, c in right_tpl.items() if c >= min_pf}
        if len(lt) + len(rt) < 3:
            lt, rt = dict(left_tpl), dict(right_tpl)

        lt_scored: dict[str, float] = {}
        for tmpl, seed_count in lt.items():
            total_count = len(self._find_all(tmpl))
            specificity = seed_count / max(total_count, 1)
            lt_scored[tmpl] = seed_count * (1.0 + specificity)
        rt_scored: dict[str, float] = {}
        for tmpl, seed_count in rt.items():
            total_count = len(self._find_all(tmpl))
            specificity = seed_count / max(total_count, 1)
            rt_scored[tmpl] = seed_count * (1.0 + specificity)

        lt = dict(sorted(lt_scored.items(),
                         key=lambda x: x[1], reverse=True)[:_max_tpl])
        rt = dict(sorted(rt_scored.items(),
                         key=lambda x: x[1], reverse=True)[:_max_tpl])

        decay = self.CONTEXT_DECAY
        cands: dict[str, dict] = {}
        seed_set = set(seeds)

        for tmpl, tw in lt.items():
            positions = self._find_all(tmpl)
            if len(positions) > _max_pos:
                positions = positions[::len(positions) // _max_pos + 1]
            for tpos in positions:
                base = tpos + len(tmpl)
                for gap in range(_max_gap + 1):
                    hits = self._vocab_after(base + gap, vocab)
                    if not hits:
                        continue
                    gap_w = decay ** gap
                    for w in hits:
                        if w not in seed_set:
                            if w not in cands:
                                cands[w] = {'score': 0.0, 'patterns': set()}
                            cands[w]['score'] += tw * gap_w
                            cands[w]['patterns'].add(f'{tmpl}…')
                    break

        for tmpl, tw in rt.items():
            positions = self._find_all(tmpl)
            if len(positions) > _max_pos:
                positions = positions[::len(positions) // _max_pos + 1]
            for tpos in positions:
                for gap in range(_max_gap + 1):
                    hits = self._vocab_before(tpos - gap, vocab)
                    if not hits:
                        continue
                    gap_w = decay ** gap
                    for w in hits:
                        if w not in seed_set:
                            if w not in cands:
                                cands[w] = {'score': 0.0, 'patterns': set()}
                            cands[w]['score'] += tw * gap_w
                            cands[w]['patterns'].add(f'…{tmpl}')
                    break

        scores: dict[str, float] = {}
        patterns: dict[str, list] = {}
        for w, info in cands.items():
            np_ = len(info['patterns'])
            scores[w] = info['score'] * math.sqrt(np_)
            patterns[w] = sorted(info['patterns'])
        return StrategyResult(scores, meta={'patterns': patterns})

    _SENT_BREAKS = frozenset('。！？…\n')

    def _strategy_cooccurrence(self, seeds: list[str], vocab: dict,
                               window: int = 50,
                               _max_occ: int = 200) -> StrategyResult:
        """共现近邻：统计在种子词窗口内高频出现的词，用 Lift 比率衡量关联强度"""
        kw_locs: list[tuple[int, int]] = []
        for s in seeds:
            kw_locs.extend((p, len(s)) for p in self._find_all(s))
        if not kw_locs:
            return StrategyResult({})
        kw_locs.sort()
        if len(kw_locs) > _max_occ:
            step = max(1, len(kw_locs) // _max_occ)
            kw_locs = kw_locs[::step]

        window_counts: Counter = Counter()
        seed_set = set(seeds)
        min_l, max_l = self.min_len, self.max_len
        text = self._text
        text_len = self._text_len
        sent_breaks = self._SENT_BREAKS

        merged_spans: list[tuple[int, int]] = []
        for kp, kl in kw_locs:
            ws = max(0, kp - window)
            we = min(text_len, kp + kl + window)
            for i in range(kp - 1, ws - 1, -1):
                if text[i] in sent_breaks:
                    ws = i + 1
                    break
            for i in range(kp + kl, we):
                if text[i] in sent_breaks:
                    we = i
                    break
            if we - ws < 10:
                ws = max(0, kp - window)
                we = min(text_len, kp + kl + window)

            if merged_spans and ws <= merged_spans[-1][1]:
                prev_s, prev_e = merged_spans[-1]
                merged_spans[-1] = (prev_s, max(prev_e, we))
            else:
                merged_spans.append((ws, we))

        n_windows = len(kw_locs)

        for ws, we in merged_spans:
            seen: set = set()
            for cm in _RE_CHINESE.finditer(text, ws, we):
                seq = cm.group()
                n = len(seq)
                for i in range(n):
                    for length in range(min_l, min(max_l + 1, n - i + 1)):
                        w = seq[i:i + length]
                        if w in vocab and w not in seed_set and w not in seen:
                            if _clean_boundary(w):
                                seen.add(w)
                                window_counts[w] += 1

        scores: dict[str, float] = {}
        for w, wc in window_counts.items():
            expected = vocab[w]['freq'] * n_windows / max(text_len, 1)
            lift = wc / max(expected, 0.01)
            scores[w] = lift
        return StrategyResult(scores)

    def _strategy_morpheme(self, keyword: str,
                           vocab: dict) -> StrategyResult:
        """构词结构相似：找与关键词共享核心语素且语素位置相同的词"""
        scores: dict[str, float] = {}

        morphemes: list[tuple[str, str]] = []
        kw_len = len(keyword)
        for i, ch in enumerate(keyword):
            if i == 0:
                pos = 'prefix'
            elif i == kw_len - 1:
                pos = 'suffix'
            else:
                pos = 'mid'
            morphemes.append((ch, pos))
        if kw_len >= 2:
            for i in range(kw_len - 1):
                bigram = keyword[i:i + 2]
                if i == 0:
                    pos = 'prefix'
                elif i + 2 == kw_len:
                    pos = 'suffix'
                else:
                    pos = 'mid'
                morphemes.append((bigram, pos))

        for morph, kw_pos in morphemes:
            is_bigram = len(morph) > 1
            base_weight = 2.0 if is_bigram else 1.0

            if is_bigram:
                candidates = self._bigram_index.get(morph, set())
            else:
                candidates = self._char_index.get(morph, set())

            for word in candidates:
                if word == keyword:
                    continue
                if not _clean_boundary(word):
                    continue

                if word.startswith(morph):
                    word_pos = 'prefix'
                elif word.endswith(morph):
                    word_pos = 'suffix'
                else:
                    word_pos = 'mid'

                if word_pos == kw_pos:
                    w_mult = 1.5
                elif word_pos == 'mid' or kw_pos == 'mid':
                    w_mult = 0.6
                else:
                    w_mult = 0.3

                freq_w = math.log2(vocab[word]['freq'] + 1)
                scores[word] = scores.get(word, 0) + \
                    base_weight * w_mult * freq_w

        return StrategyResult(scores)
