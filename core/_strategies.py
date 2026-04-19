"""
阶段二：六策略并行候选提取

StrategiesMixin 提供：
  _strategy_char_overlap    — 字符包含：从词表中找包含关键词字符的词，按位置加权
  _strategy_context_pattern — 上下文特征投票：提取种子词上下文指纹（最宽8字），反向投票发现语义相似词
  _strategy_cooccurrence    — 共现近邻：统计在种子词附近高频出现的词（Lift 比率）
  _strategy_morpheme        — 构词结构相似：找与关键词共享核心语素且处于相同位置的词
  _strategy_substitution    — 互替性：找能在相同上下文框架中替换种子词的词
  _strategy_co_topic        — 段落共主题：找与种子词在段落级别高度共现的词

所有策略统一接收 vocab 参数、统一返回 StrategyResult。
"""

import math
from bisect import bisect_left
from collections import Counter

from ._utils import StrategyResult


class StrategiesMixin:

    def _strategy_char_overlap(self, keyword: str,
                               vocab: dict) -> StrategyResult:
        """字符包含：按字符重叠比例和位置匹配度打分"""
        anchors = set(keyword)
        n_anchors = len(anchors)
        kw_len = len(keyword)
        kw_inv = 1.0 / max(kw_len - 1, 1)
        kw_positions: dict[str, float] = {}
        for i, ch in enumerate(keyword):
            kw_positions[ch] = i * kw_inv

        candidates: set[str] = set()
        for ch in anchors:
            candidates |= self._char_index.get(ch, set())

        clean_set = self._clean_set
        log_freq = self._log_freq_p1

        scores: dict[str, float] = {}
        for word in candidates:
            if word == keyword:
                continue
            if word not in clean_set:
                continue

            # 一次扫描同时拿到 overlap 字符与它们在 word 中的首位置，
            # 替代原 `set(word)` + `word.index(c)` 的两次扫描
            w_idx: dict[str, int] = {}
            for idx, c in enumerate(word):
                if c in anchors and c not in w_idx:
                    w_idx[c] = idx
            n_overlap = len(w_idx)

            char_ratio = n_overlap / n_anchors
            w_inv = 1.0 / max(len(word) - 1, 1)

            pos_match = 0.0
            for c, idx in w_idx.items():
                pos_match += 1.0 - abs(kw_positions[c] - idx * w_inv)
            pos_match /= n_overlap

            scores[word] = char_ratio * pos_match * log_freq[word]
        return StrategyResult(scores)

    # 特征宽度权重：1字×0.3, 2字×1.0, 3字×1.7, 4字×2.2, 5字×2.6, 6字×3.0,
    #              7字×3.3, 8字×3.6
    _WIDTH_MULT = (0.0, 0.3, 1.0, 1.7, 2.2, 2.6, 3.0, 3.3, 3.6)
    _PAIR_MULT = 2.5

    def _strategy_context_pattern(self, seeds: list[str], vocab: dict,
                                  _max_scan: int = 600) -> StrategyResult:
        """上下文特征投票：提取种子词多宽度上下文指纹，反向投票发现语义相似词

        算法：
        1. 遍历种子词出现位置，提取 1-8 字左/右上下文特征，
           以及 (左1-2字, 右1-2字) 配对特征
        2. 按 IDF × 宽度权重 加权，取 top-K
        3. 对每个高权重特征反向投票
        4. 得分 = 投票总分 × sqrt(命中特征种数) / log2(词频+2)
        """
        text = self._text
        text_len = self._text_len
        seed_set = set(seeds)

        def _cjk(ch: str) -> bool:
            return '\u4e00' <= ch <= '\u9fff'

        # ── 1. 提取种子上下文特征（1-10字 + 配对）──
        feat_left: Counter = Counter()
        feat_right: Counter = Counter()
        feat_pair: Counter = Counter()

        n_occ = 0
        for seed in seeds:
            positions = self._find_all(seed)
            n_occ += len(positions)
            slen = len(seed)
            for pos in positions:
                end = pos + slen
                lfs: list[str] = []
                for w in range(1, 9):
                    if pos < w or not _cjk(text[pos - w]):
                        break
                    lfs.append(text[pos - w:pos])
                rfs: list[str] = []
                for w in range(1, 9):
                    if end + w > text_len or not _cjk(text[end + w - 1]):
                        break
                    rfs.append(text[end:end + w])

                for f in lfs: feat_left[f] += 1
                for f in rfs: feat_right[f] += 1
                for lc in lfs[:2]:
                    for rc in rfs[:2]:
                        feat_pair[(lc, rc)] += 1

        if n_occ == 0:
            return StrategyResult({})

        # ── 2. IDF × 宽度权重，取 top-K ──
        freq_map = self._freq
        wm = self._WIDTH_MULT

        def _idf_top(counts: Counter, top_k: int) -> dict:
            # 特征长度上限 = range(1, 9) = 1..8，与 max_len 一致；_freq 在
            # vocab_builder Pass 1a 已覆盖单字 + 长度 min_len..max_len 的所有
            # ngram，且 Apriori 剪枝丢掉的全是 freq < 2 的（即 freq == 1）。
            # 因此 feat 不在 freq_map 时其全文出现次数必为 1，直接用 1 与原
            # text.count(feat) 兜底**完全等价**，省掉每次 O(N) 全文扫。
            # （在韩立 × 150 万字场景下，原路径 38606 次 str.count 吃掉 ~28s）
            scored = {}
            for feat, cnt in counts.items():
                gf = freq_map.get(feat, 1)
                idf = math.log2(max(text_len / gf, 1.0))
                scored[feat] = cnt * idf * wm[len(feat)]
            return dict(sorted(scored.items(),
                               key=lambda x: x[1], reverse=True)[:top_k])

        wL = _idf_top(feat_left, 30)
        wR = _idf_top(feat_right, 30)

        pm = self._PAIR_MULT
        wP: dict[tuple, float] = {}
        for (lc, rc), cnt in feat_pair.items():
            gf = min(freq_map.get(lc, 1), freq_map.get(rc, 1))
            idf = math.log2(max(text_len / max(gf, 1), 1.0))
            wP[(lc, rc)] = cnt * idf * pm
        wP = dict(sorted(wP.items(),
                         key=lambda x: x[1], reverse=True)[:15])

        # ── 3. 反向投票 ──
        votes: dict[str, float] = {}
        feats_hit: dict[str, set] = {}

        def _cap(positions: list) -> list:
            if len(positions) > _max_scan:
                return positions[::len(positions) // _max_scan + 1]
            return positions

        def _vote(word: str, weight: float, label: str):
            votes[word] = votes.get(word, 0) + weight
            feats_hit.setdefault(word, set()).add(label)

        for feat, wt in wL.items():
            flen = len(feat)
            for p in _cap(self._find_all(feat)):
                for w in self._vocab_after(p + flen, vocab):
                    if w not in seed_set:
                        _vote(w, wt, f'{feat}…')

        for feat, wt in wR.items():
            for p in _cap(self._find_all(feat)):
                for w in self._vocab_before(p, vocab):
                    if w not in seed_set:
                        _vote(w, wt, f'…{feat}')

        for (lc, rc), wt in wP.items():
            llen = len(lc)
            rlen = len(rc)
            for p in _cap(self._find_all(lc)):
                for w in self._vocab_after(p + llen, vocab):
                    if w in seed_set:
                        continue
                    w_end = p + llen + len(w)
                    if (w_end + rlen <= text_len
                            and text[w_end:w_end + rlen] == rc):
                        _vote(w, wt, f'{lc}…{rc}')

        # ── 3.5 特征内嵌候选词提取 ──
        # 与种子词强共现的词往往成为高权重特征的一部分（如"霍雨浩和"），
        # 导致反向投票只能发现特征后/前的词，而非特征中内嵌的词本身。
        min_l, max_l = self.min_len, self.max_len
        for feat, wt in list(wL.items()) + list(wR.items()):
            if len(feat) < min_l:
                continue
            for length in range(min_l, min(max_l + 1, len(feat) + 1)):
                for start in range(len(feat) - length + 1):
                    sub = feat[start:start + length]
                    if sub in vocab and sub not in seed_set:
                        embed_wt = wt * 0.6
                        votes[sub] = votes.get(sub, 0) + embed_wt
                        feats_hit.setdefault(sub, set()).add(
                            f'[嵌]{feat}')
        for (lc, rc), wt in wP.items():
            for part in (lc, rc):
                if len(part) < min_l:
                    continue
                for length in range(min_l, min(max_l + 1, len(part) + 1)):
                    for start in range(len(part) - length + 1):
                        sub = part[start:start + length]
                        if sub in vocab and sub not in seed_set:
                            embed_wt = wt * 0.4
                            votes[sub] = votes.get(sub, 0) + embed_wt
                            feats_hit.setdefault(sub, set()).add(
                                f'[嵌]{lc}…{rc}')

        # ── 4. 归一化得分 ──
        scores: dict[str, float] = {}
        patterns: dict[str, list] = {}
        log_freq_p2 = self._log_freq_p2
        for w, raw in votes.items():
            if w not in vocab:
                continue
            nf = len(feats_hit.get(w, ()))
            scores[w] = raw * math.sqrt(max(nf, 1)) / log_freq_p2[w]
            patterns[w] = sorted(feats_hit.get(w, ()))

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

        # 用 _build_ac_index 预算的 _vocab_starts 直接迭代 span 内的 vocab
        # 命中位置，省掉原 (i, length) 的 O(span_len · max_len) Python 双
        # 循环。_vocab_starts 已经过 _clean_set 过滤，与原 `w in clean_set`
        # 等价；vocab 成员检查兜底处理 channel-C extras 之外的差异。
        vocab_starts = self._vocab_starts
        for ws, we in merged_spans:
            seen: set = set()
            for pos in range(ws, we):
                hits = vocab_starts.get(pos)
                if not hits:
                    continue
                for w in hits:
                    if pos + len(w) > we:
                        continue
                    if w in seed_set or w in seen:
                        continue
                    if w not in vocab:
                        continue
                    seen.add(w)
                    window_counts[w] += 1

        scores: dict[str, float] = {}
        rate_factor = n_windows / max(text_len, 1)
        cond_factor = 1.0 / max(n_windows, 1)
        for w, wc in window_counts.items():
            expected = vocab[w]['freq'] * rate_factor
            lift = wc / max(expected, 0.01)
            log_wc = math.log2(wc + 1)
            lift_score = lift * log_wc
            cond_score = wc * cond_factor * log_wc * 10
            scores[w] = lift_score if lift_score > cond_score else cond_score
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

        clean_set = self._clean_set
        log_freq = self._log_freq_p1
        char_index = self._char_index
        bigram_index = self._bigram_index

        for morph, kw_pos in morphemes:
            is_bigram = len(morph) > 1
            base_weight = 2.0 if is_bigram else 1.0

            if is_bigram:
                candidates = bigram_index.get(morph, set())
            else:
                candidates = char_index.get(morph, set())

            for word in candidates:
                if word == keyword:
                    continue
                if word not in clean_set:
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

                scores[word] = scores.get(word, 0) + \
                    base_weight * w_mult * log_freq[word]

        return StrategyResult(scores)

    # 框架宽度配置：(左宽, 右宽, 权重)
    # 双侧2字框架最具区分度，单侧1字作为补充覆盖
    _FRAME_SPECS = ((2, 2, 2.5), (2, 1, 1.5), (1, 2, 1.5), (1, 1, 1.0))

    def _strategy_substitution(self, seeds: list[str], vocab: dict,
                               _max_scan: int = 400) -> StrategyResult:
        """互替性：找能在相同上下文框架中替换种子词的词

        算法：
        1. 在种子词每个出现位置，提取多宽度 (左n字, 右n字) 上下文框架
        2. 对每个框架，在全文中搜索能填入该框架的其他词表词
        3. 得分 = 共享框架加权总分 × sqrt(覆盖率)

        与 context_pattern 的区别：
          context_pattern 逐特征独立投票（左特征和右特征各自匹配）
          substitution 要求左右两侧同时匹配（完整框架），衡量的是
          "候选词能替换种子词出现在多少种上下文中"
        """
        text = self._text
        text_len = self._text_len
        seed_set = set(seeds)

        def _all_cjk(s: str) -> bool:
            for c in s:
                if not ('\u4e00' <= c <= '\u9fff'):
                    return False
            return True

        # ── 1. 收集种子词的上下文框架 ──
        seed_frames: dict[tuple[str, str], float] = {}

        for seed in seeds:
            slen = len(seed)
            for pos in self._find_all(seed):
                end = pos + slen
                for lw, rw, fw in self._FRAME_SPECS:
                    ls = pos - lw
                    re = end + rw
                    if ls < 0 or re > text_len:
                        continue
                    left = text[ls:pos]
                    right = text[end:re]
                    if not _all_cjk(left) or not _all_cjk(right):
                        continue
                    frame = (left, right)
                    seed_frames[frame] = seed_frames.get(frame, 0) + fw

        if not seed_frames:
            return StrategyResult({})

        n_frames = len(seed_frames)

        # ── 2. 对每个框架，搜索可替换词 ──
        word_frames: dict[str, set[tuple[str, str]]] = {}
        word_weight: dict[str, float] = {}

        sorted_frames = sorted(seed_frames.items(),
                               key=lambda x: x[1], reverse=True)

        for (left, right), frame_w in sorted_frames:
            llen = len(left)
            rlen = len(right)
            positions = self._find_all(left)
            if len(positions) > _max_scan:
                step = len(positions) // _max_scan + 1
                positions = positions[::step]

            for p in positions:
                ws = p + llen
                for w in self._vocab_after(ws, vocab):
                    if w in seed_set:
                        continue
                    w_end = ws + len(w)
                    if (w_end + rlen <= text_len
                            and text[w_end:w_end + rlen] == right):
                        word_frames.setdefault(w, set()).add((left, right))
                        word_weight[w] = word_weight.get(w, 0) + frame_w

        # ── 3. 计算得分 ──
        scores: dict[str, float] = {}
        for w, frames in word_frames.items():
            n_shared = len(frames)
            coverage = n_shared / max(n_frames, 1)
            weighted = sum(seed_frames[f] for f in frames)
            scores[w] = weighted * math.sqrt(coverage)

        return StrategyResult(scores)

    def _strategy_co_topic(self, seeds: list[str],
                           vocab: dict) -> StrategyResult:
        """段落共主题：找与种子词在段落级别高度共现的词

        算法：
        1. 收集种子词出现的段落集合 S_seed
        2. 遍历词表，计算每个词 w 与 S_seed 的段落重叠度
        3. 用段落级 Lift（实际重叠 / 随机预期重叠）衡量关联强度
           得分 = (lift - 1) × log2(实际重叠段落数 + 1)

        与 cooccurrence 的区别：
          cooccurrence 看 50 字窗口内的句子级共现
          co_topic 看 500 字段落级共现，捕获跨句、跨段的宏观主题关联
        """
        word_segs = self._word_segs
        seg_words = self._seg_words

        seed_segs: set[int] = set()
        for s in seeds:
            s_segs = word_segs.get(s)
            if s_segs:
                seed_segs |= s_segs

        if not seed_segs:
            return StrategyResult({})

        n_seed = len(seed_segs)
        n_total = self._n_segments
        seed_set = set(seeds)

        # 反向索引：只有在某个 seed 段里出现过的词才可能 observed >= 1。
        # 原实现 `for w in vocab` 要遍历整张候选池（长篇可达数十万），
        # 现在只遍历"种子段并集里实际出现过的词"的 Counter，规模通常
        # 缩小 10-100 倍。Counter.update 一次性给出每个词的 observed 计数，
        # 与原 `len(seed_segs & w_segs)` 数值完全一致。
        seg_obs: Counter = Counter()
        for sid in seed_segs:
            sw = seg_words.get(sid)
            if sw:
                seg_obs.update(sw)

        seed_seg_sorted = sorted(seed_segs)
        local_window = max(n_total // 10, 50)
        expected_factor = n_seed / max(n_total, 1)
        inv_local_window = 1.0 / max(local_window, 1)
        step = max(1, local_window // 2)

        scores: dict[str, float] = {}
        for w, observed in seg_obs.items():
            if w in seed_set:
                continue
            if observed < 2:
                continue
            if w not in vocab:
                continue
            w_segs = word_segs[w]
            expected = expected_factor * len(w_segs)

            global_lift = (observed / max(expected, 0.01)
                           if expected >= 0.5 else 0.0)

            if global_lift > 1.0:
                scores[w] = (global_lift - 1.0) * math.log2(observed + 1)
                continue

            # 全局 Lift 不足时，尝试局部窗口检测
            if len(w_segs) < 3:
                continue
            # 用 bisect 把每个窗口的计数从 O(|seed|+|w|) 的线性求和
            # 降到 O(log)。inter_sorted 预先把交集排好，避免内层
            # `s in seed_segs` 反复哈希查询。
            inter_sorted = sorted(seed_segs & w_segs)
            w_seg_sorted = sorted(w_segs)

            best_local_lift = 0.0
            best_local_obs = 0
            for win_start in range(0, n_total - local_window + 1, step):
                win_end = win_start + local_window
                local_seed = (bisect_left(seed_seg_sorted, win_end)
                              - bisect_left(seed_seg_sorted, win_start))
                local_w = (bisect_left(w_seg_sorted, win_end)
                           - bisect_left(w_seg_sorted, win_start))
                if local_seed < 2 or local_w < 2:
                    continue
                local_obs = (bisect_left(inter_sorted, win_end)
                             - bisect_left(inter_sorted, win_start))
                if local_obs < 2:
                    continue
                local_exp = local_seed * local_w * inv_local_window
                if local_exp > 0.5:
                    ll = local_obs / local_exp
                    if ll > best_local_lift:
                        best_local_lift = ll
                        best_local_obs = local_obs

            if best_local_lift > 2.0:
                scores[w] = ((best_local_lift - 1.0)
                             * math.log2(best_local_obs + 1) * 0.7)

        return StrategyResult(scores)
