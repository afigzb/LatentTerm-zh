"""
三阶段级联中文术语提取器

阶段一  build_index  — 全文词汇发现（与关键词无关，只需执行一次）
                       实现见 _vocab_builder.py → VocabBuilderMixin
阶段二  _strategy_*  — 六策略并行候选提取
                       字符包含 / 上下文 / 共现 / 构词 / 互替 / 段落共主题
                       实现见 _strategies.py  → StrategiesMixin
阶段三  extract      — 多信号归一化融合 + 交叉验证加分 + 后处理（本文件）
"""

from collections import Counter, defaultdict

from ._utils import _normalize, _LEFT_NOISE, _RIGHT_NOISE
from ._vocab_builder import VocabBuilderMixin
from ._strategies import StrategiesMixin
from ._pattern_miner import (
    infer_keyword_type, format_evidence_snippets,
)


# ── 默认配置（UI 层和算法层共用）────────────────────────────────────────
DEFAULT_CONFIG = {
    'min_freq': 3,
    'max_freq': 0,
    'top_n': 75,
    'w_char': 0.25, 'w_context': 0.40, 'w_cooccur': 0.40,
    'w_morph': 0.15, 'w_subst': 0.30, 'w_topic': 0.25,
}


class TermExtractor(VocabBuilderMixin, StrategiesMixin):

    # ── 集中管理的阈值与参数 ──────────────────────────────────────────────

    # Pass 1 双重过滤（build_index 两步法使用）
    SOLID_PMI_MIN = 1.0
    SOLID_FREE_MIN = 0.1

    # 词表构建（build_index 使用），log-PMI 尺度
    VOCAB_COH_STRICT = 3.0
    VOCAB_FREE_STRICT = 0.1
    VOCAB_COH_RELAXED = 1.5
    VOCAB_FREE_RELAXED = 0.03

    # 伪合成词过滤（_filter_extensions 使用），log-PMI 尺度
    FILTER_COH_STRICT = 3.0
    FILTER_COH_LENIENT = 2.0
    FILTER_DOM_STRICT = 8
    FILTER_DOM_LENIENT = 20

    # 评分融合（extract 使用）
    MISS_PENALTY = -0.03
    CROSS_BONUS = 0.1

    # 策略内部衰减
    CONTEXT_DECAY = 0.7

    # 类型先验 & 模板同族（R5 / R6）
    TYPE_MATCH_BONUS = 1.30          # 同类型乘子
    TYPE_MISMATCH_PENALTY = 0.85     # 异类型乘子（两端都非 misc 才生效）
    TEMPLATE_FAMILY_BONUS = 0.08     # 模板同族加分

    # 高置信度模板集合（L2 命中这些模板可豁免 min_freq 门槛）
    _HIGH_CONF_TEMPLATES = frozenset({
        'B1.dialog', 'B2.named', 'B5.role',
    })

    def __init__(self, min_len: int = 2, max_len: int = 8):
        self.min_len = min_len
        self.max_len = max_len
        self._text: str = ''
        self._text_len: int = 0
        self._total_chars: int = 0
        self._freq: Counter = Counter()
        self._left_nb: defaultdict = defaultdict(Counter)
        self._right_nb: defaultdict = defaultdict(Counter)
        self._vocab: dict = {}
        self._vocab_relaxed: dict = {}
        self._built: bool = False

    # ------------------------------------------------------------------ #
    # 通道 C：关键词自适应模板（extract 运行时扩充候选池）                        #
    # ------------------------------------------------------------------ #

    def _channel_c_expand(self, keyword: str) -> dict:
        """用关键词上下文指纹从原文反向挖掘候选，不污染 self._candidates。"""
        positions = self._find_all(keyword)
        if not positions or not hasattr(self, '_pattern_miner'):
            return {}
        # 把已有 L1/L2 候选作为对齐字典，消除"只听张无忌"→"张无忌"
        known_vocab = set(self._candidates.keys())
        c_hits = self._pattern_miner.channel_c(
            keyword, self._text, positions, known_vocab=known_vocab)
        extra: dict = {}
        for w, meta in c_hits.items():
            if w in self._candidates:
                # 已在池 → 补充证据/模板回写（只影响本次返回的 info）
                base = dict(self._candidates[w])
                base['templates'] = set(base.get('templates') or set()) \
                    | set(meta['templates'])
                base['origins'] = set(base.get('origins') or set()) \
                    | set(meta['origins'])
                if not base.get('type'):
                    base['type'] = meta['type']
                budget = 5 - len(base.get('evidence') or [])
                if budget > 0:
                    base['evidence'] = (base.get('evidence') or []) \
                        + list(meta['evidence'][:budget])
                extra[w] = base
                continue
            extra[w] = {
                'freq': meta['freq'],
                'tier': 'L2',
                'origins': set(meta['origins']),
                'templates': set(meta['templates']),
                'type': meta['type'],
                'evidence': list(meta['evidence'][:5]),
                'cohesion': 0.0,
                'freedom': 0.0,
            }
        return extra

    # ------------------------------------------------------------------ #
    # 关键词类型推断                                                        #
    # ------------------------------------------------------------------ #

    def _keyword_type(self, keyword: str, vocab: dict) -> str | None:
        """优先从候选池查；没查到再用提示词映射。"""
        info = vocab.get(keyword)
        if info and info.get('type') and info['type'] != 'misc':
            return info['type']
        return infer_keyword_type(keyword)

    # ------------------------------------------------------------------ #
    # 阶段三：多信号融合排序                                                  #
    # ------------------------------------------------------------------ #

    def extract(self, keyword: str, top_n: int = 75, min_freq: int = 2,
                w_char: float = 0.25, w_context: float = 0.40,
                w_cooccur: float = 0.40, w_morph: float = 0.15,
                w_subst: float = 0.30, w_topic: float = 0.25,
                max_freq: int = 0) -> list[dict]:
        if not self._built:
            raise RuntimeError('请先调用 build_index(text)')

        # ── 运行时候选池 = 持久候选 + 通道 C 扩充 ──
        extra = self._channel_c_expand(keyword)
        if extra:
            vocab = {**self._candidates, **extra}
        else:
            vocab = self._candidates

        raw_scores, pattern_info = self._run_strategies(keyword, vocab)

        n_char = _normalize(raw_scores['char_overlap'])
        n_context = _normalize(raw_scores['context_pattern'])
        n_cooccur = _normalize(raw_scores['cooccurrence'])
        n_morph = _normalize(raw_scores['morpheme'])
        n_subst = _normalize(raw_scores['substitution'])
        n_topic = _normalize(raw_scores['co_topic'])
        all_words = (set(n_char) | set(n_context) | set(n_cooccur)
                     | set(n_morph) | set(n_subst) | set(n_topic))

        # R5/R6 所需的关键词类型与模板集合
        kw_type = self._keyword_type(keyword, vocab)
        kw_info = vocab.get(keyword) or {}
        kw_templates = set(kw_info.get('templates') or set())

        miss = self.MISS_PENALTY
        results = []
        for word in all_words:
            info = vocab.get(word)
            if not info:
                continue
            if not self._passes_freq_gate(info, min_freq, max_freq):
                continue

            s_char = n_char.get(word, miss)
            s_context = n_context.get(word, miss)
            s_cooccur = n_cooccur.get(word, miss)
            s_morph = n_morph.get(word, miss)
            s_subst = n_subst.get(word, miss)
            s_topic = n_topic.get(word, miss)

            score = (w_char * s_char + w_context * s_context
                     + w_cooccur * s_cooccur + w_morph * s_morph
                     + w_subst * s_subst + w_topic * s_topic)
            hits = sum(1 for s in (s_char, s_context, s_cooccur,
                                   s_morph, s_subst, s_topic) if s > 0)
            score += self.CROSS_BONUS * max(0, hits - 1)

            # R5 类型先验
            w_type = info.get('type')
            if (kw_type and w_type and w_type != 'misc'
                    and kw_type != 'misc'):
                if w_type == kw_type:
                    score *= self.TYPE_MATCH_BONUS
                else:
                    score *= self.TYPE_MISMATCH_PENALTY

            # R6 模板同族
            w_templates = set(info.get('templates') or set())
            if kw_templates and (kw_templates & w_templates):
                score += self.TEMPLATE_FAMILY_BONUS

            labels = []
            if s_char > 0:
                labels.append('字符')
            if s_context > 0:
                labels.append('上下文')
            if s_cooccur > 0:
                labels.append('共现')
            if s_morph > 0:
                labels.append('构词')
            if s_subst > 0:
                labels.append('互替')
            if s_topic > 0:
                labels.append('共主题')

            evidence_snips = []
            ev = info.get('evidence') or []
            if ev:
                evidence_snips = format_evidence_snippets(
                    self._text, ev, word, max_count=2)

            results.append({
                'word': word,
                'freq': info['freq'],
                'score': round(score, 4),
                'strategies': '、'.join(labels),
                'hit_count': hits,
                'matched_patterns': '  '.join(
                    pattern_info.get(word, [])),
                'tier': info.get('tier', 'L1'),
                'type': info.get('type') or '',
                'templates': '、'.join(
                    sorted(info.get('templates') or set())),
                'evidence': '  ∥  '.join(evidence_snips),
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        results = [r for r in results if r['word'] != keyword][:top_n]

        kw_freq = self._freq.get(keyword)
        if kw_freq is None:
            kw_freq = self._text.count(keyword)
        if kw_freq > 0:
            # 命中词本身的证据（若有）
            kw_ev_snips = []
            if kw_info.get('evidence'):
                kw_ev_snips = format_evidence_snippets(
                    self._text, kw_info['evidence'], keyword, max_count=2)
            kw_entry = {
                'word': keyword,
                'freq': kw_freq,
                'score': 9999.0,
                'strategies': '用户输入',
                'hit_count': 6,
                'matched_patterns': '',
                'tier': kw_info.get('tier', 'L1'),
                'type': kw_info.get('type') or (kw_type or ''),
                'templates': '、'.join(
                    sorted(kw_info.get('templates') or set())),
                'evidence': '  ∥  '.join(kw_ev_snips),
            }
            results = [kw_entry] + results

        return self._group_by_parent(results)

    # ------------------------------------------------------------------ #
    # 频率门槛：L2 + 高置信模板可豁免 min_freq                                  #
    # ------------------------------------------------------------------ #

    def _passes_freq_gate(self, info: dict,
                          min_freq: int, max_freq: int) -> bool:
        freq = info['freq']
        tier = info.get('tier', 'L1')
        templates = info.get('templates') or set()

        if max_freq > 0 and freq > max_freq:
            return False

        if freq >= min_freq:
            return True

        # L2 + 高置信模板 → 豁免 min_freq 下限
        if tier == 'L2' and (templates & self._HIGH_CONF_TEMPLATES):
            return True
        return False

    # ------------------------------------------------------------------ #
    # 策略调度：单跳 + 结构驱动的小幅扩展                                        #
    # ------------------------------------------------------------------ #

    def _run_strategies(self, keyword: str, vocab: dict):
        """返回 (raw_scores_dict, pattern_info)

        raw_scores_dict 的 key 固定为六个策略名：
          char_overlap / context_pattern / cooccurrence /
          morpheme / substitution / co_topic
        """
        raw_char = self._strategy_char_overlap(keyword, vocab).scores
        raw_morph = self._strategy_morpheme(keyword, vocab).scores

        if self._find_all(keyword):
            seeds = [keyword]
        else:
            combined = {
                w: raw_char.get(w, 0) + raw_morph.get(w, 0)
                for w in set(raw_char) | set(raw_morph)
            }
            seeds = sorted(combined, key=combined.get, reverse=True)[:5]

        if seeds:
            res_ctx = self._strategy_context_pattern(seeds, vocab)
            raw_context = res_ctx.scores
            pattern_info = res_ctx.meta.get('patterns', {})
            raw_cooccur = self._strategy_cooccurrence(seeds, vocab).scores
            raw_subst = self._strategy_substitution(seeds, vocab).scores
            raw_topic = self._strategy_co_topic(seeds, vocab).scores
        else:
            raw_context, pattern_info = {}, {}
            raw_cooccur, raw_subst, raw_topic = {}, {}, {}

        return {
            'char_overlap': raw_char,
            'context_pattern': raw_context,
            'cooccurrence': raw_cooccur,
            'morpheme': raw_morph,
            'substitution': raw_subst,
            'co_topic': raw_topic,
        }, pattern_info

    _INDEP_FRAGMENT_THRESHOLD = 0.15
    _INDEP_PHRASE_FREQ_RATIO = 0.05

    def _group_by_parent(self, results: list[dict]) -> list[dict]:
        """按包含关系分组，用独立性比率判定谁是真正的父词。

        independence(w) = clean_freq(w) / raw_freq(w)
        衡量一个词有多少出现是"独立的"（不是更长词的一部分）。

        short ⊂ long 时三条规则：
          1. 短词独立性低（< 15%）→ 短词是长词的残片 → 合并到长词下
          2. 短词独立性高 + 长词频次远低于短词（< 5%）→ 长词是临时短语 → 挂到短词下
          3. 短词独立性高 + 长词频次也不低 → 两个独立术语，不合并
        """
        noise = _LEFT_NOISE | _RIGHT_NOISE
        raw_freq = self._freq
        words = [r['word'] for r in results]
        freq_map = {r['word']: r['freq'] for r in results}
        n = len(words)
        child_of: dict[int, int] = {}

        for i in range(n):
            for j in range(i + 1, n):
                wi, wj = words[i], words[j]
                if len(wi) == len(wj):
                    continue

                if len(wi) < len(wj):
                    if wi not in wj:
                        continue
                    short_idx, long_idx = i, j
                else:
                    if wj not in wi:
                        continue
                    short_idx, long_idx = j, i

                short_w = words[short_idx]
                long_w = words[long_idx]
                short_clean = freq_map[short_w]
                long_clean = freq_map[long_w]
                short_raw = raw_freq.get(short_w, short_clean)
                independence = short_clean / max(short_raw, 1)

                if independence < self._INDEP_FRAGMENT_THRESHOLD:
                    # 规则1：短词是长词的残片（如 雨浩→霍雨浩）
                    parent_idx, child_idx = long_idx, short_idx
                elif long_clean / max(short_clean, 1) < self._INDEP_PHRASE_FREQ_RATIO:
                    # 规则2：长词是围绕短词的临时短语（如 王冬联手→王冬）
                    parent_idx, child_idx = short_idx, long_idx
                elif independence > 0.4:
                    # 规则3：短词高度独立，两个术语不合并
                    continue
                else:
                    # 中间地带：退回噪声字启发式
                    extra = long_w.replace(short_w, '', 1)
                    if all(ch in noise for ch in extra):
                        parent_idx, child_idx = short_idx, long_idx
                    else:
                        parent_idx, child_idx = long_idx, short_idx

                if child_idx in child_of:
                    continue
                child_r = results[child_idx]
                parent_r = results[parent_idx]
                # 规则1命中时跳过频率保护（残片必须合并）
                if independence >= self._INDEP_FRAGMENT_THRESHOLD:
                    if child_r['freq'] > parent_r['freq'] * 3:
                        continue
                child_of[child_idx] = parent_idx

        for r in results:
            r.setdefault('children', [])

        for ci, pi in child_of.items():
            if pi in child_of:
                continue
            results[pi]['children'].append(results[ci])

        return [r for i, r in enumerate(results) if i not in child_of]
