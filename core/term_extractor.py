"""
三阶段级联中文术语提取器

阶段一  build_index  — 全文词汇发现（与关键词无关，只需执行一次）
                       实现见 _vocab_builder.py → VocabBuilderMixin
阶段二  _strategy_*  — 六策略并行候选提取
                       字符包含 / 上下文 / 共现 / 构词 / 互替 / 段落共主题
                       实现见 _strategies.py  → StrategiesMixin
阶段三  extract      — 多信号归一化融合 + 交叉验证加分 + 后处理（本文件）
"""

from bisect import bisect_right
from collections import Counter, defaultdict

from ._utils import _normalize, _LEFT_NOISE, _RIGHT_NOISE
from ._vocab_builder import VocabBuilderMixin
from ._strategies import StrategiesMixin
from ._pattern_miner import (
    infer_keyword_type, format_evidence_snippets,
)


# ── 默认配置（UI 层和算法层共用）────────────────────────────────────────
DEFAULT_CONFIG = {
    'min_freq': 5,
    'max_freq': 0,
    'top_n': 1000,
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

    # 类型先验 & 模板同族（R5 / R6）
    # 类型先验改为加法式"弱证据"：同类加分，异类不处理。
    # 原因：类型推断是启发式的，乘法放大会让一次误判同时砸掉/抬高
    # 一大截；而且异类扣分在双关键词模式下已经被关闭过一次，说明
    # 它的置信度不配当"仲裁器"，降级为和模板同族同量级的加分更稳。
    TYPE_MATCH_BONUS = 0.08          # 同类加分（加法）
    TEMPLATE_FAMILY_BONUS = 0.08     # 模板同族加分

    # 双关键词联合模式参数
    JOINT_WINDOW = 1000               # 联合锚点窗口（A、B 距离阈值，字符）
    JOINT_GATE_MIN_HITS = 1          # 联合证据硬门槛：至少要在多少联合窗口/段出现
    JOINT_BONUS_ALPHA = 0.7          # 联合证据加分系数（score *= 1+α·joint_ratio）

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

    def _channel_c_expand(self, keyword: str,
                          positions_override: list | None = None) -> dict:
        """用关键词上下文指纹从原文反向挖掘候选，不污染 self._candidates。

        positions_override: 双关键词模式下传入"靠近另一关键词的子集位置"，
            让模板挖矿只看 AB 共现处，避免引入只跟单边强相关的临时词。
        """
        if positions_override is not None:
            positions = positions_override
        else:
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
    # 双关键词联合模式：纯位置工具函数                                          #
    # ------------------------------------------------------------------ #

    def _compute_joint_anchors(self, kw_a: str, kw_b: str,
                               window: int):
        """扫描 A、B 互为邻居的位置对。

        返回 (pos_a_kept, pos_b_kept, pair_spans):
          pos_a_kept   — A 出现位置中能在 ±window 字内找到 B 的子集
          pos_b_kept   — 对称的 B 子集（去重排序）
          pair_spans   — 每个 (a_pos, b_pos) 对覆盖的 [min, max+len) 区间
        """
        pa = self._find_all(kw_a)
        pb = self._find_all(kw_b)
        if not pa or not pb:
            return [], [], []

        la, lb = len(kw_a), len(kw_b)
        pos_a_kept: list[int] = []
        pos_b_set: set[int] = set()
        pair_spans: list[tuple[int, int]] = []

        n_b = len(pb)
        j = 0
        for ai in pa:
            while j < n_b and pb[j] + lb < ai - window:
                j += 1
            k = j
            kept = False
            while k < n_b and pb[k] <= ai + la + window:
                bi = pb[k]
                kept = True
                pos_b_set.add(bi)
                left = ai if ai < bi else bi
                right = (ai + la) if (ai + la) > (bi + lb) else (bi + lb)
                pair_spans.append((left, right))
                k += 1
            if kept:
                pos_a_kept.append(ai)

        return pos_a_kept, sorted(pos_b_set), pair_spans

    def _segments_of(self, kw: str) -> set[int]:
        """词的段落集合，候选池有就直接取，没收录就按位置临时算。"""
        cached = getattr(self, '_word_segs', {}).get(kw)
        if cached is not None:
            return cached
        seg_size = getattr(self, '_SEG_SIZE', 500)
        return {p // seg_size for p in self._find_all(kw)}

    def _compute_joint_segments(self, kw_a: str, kw_b: str) -> set[int]:
        """A、B 同时出现的段落集合。"""
        return self._segments_of(kw_a) & self._segments_of(kw_b)

    @staticmethod
    def _max_fuse(scores_a: dict, scores_b: dict) -> dict:
        """两路分数取并集后逐词取 max。"""
        if not scores_b:
            return dict(scores_a)
        out = dict(scores_a)
        for w, s in scores_b.items():
            cur = out.get(w, 0.0)
            if s > cur:
                out[w] = s
        return out

    @staticmethod
    def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not spans:
            return []
        spans = sorted(spans)
        merged = [spans[0]]
        for s, e in spans[1:]:
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1] = (ps, e if e > pe else pe)
            else:
                merged.append((s, e))
        return merged

    def _count_in_merged_spans(self, word: str,
                               merged_spans: list[tuple[int, int]]) -> int:
        """word 出现位置中落在任一 merged_span 内的次数。"""
        if not merged_spans:
            return 0
        positions = self._find_all(word)
        if not positions:
            return 0
        starts = [s for s, _ in merged_spans]
        ends = [e for _, e in merged_spans]
        wlen = len(word)
        cnt = 0
        for p in positions:
            i = bisect_right(starts, p) - 1
            if i >= 0 and p + wlen <= ends[i]:
                cnt += 1
        return cnt

    # ------------------------------------------------------------------ #
    # 阶段三：多信号融合排序                                                  #
    # ------------------------------------------------------------------ #

    def extract(self, keyword: str, top_n: int = 75, min_freq: int = 2,
                w_char: float = 0.25, w_context: float = 0.40,
                w_cooccur: float = 0.40, w_morph: float = 0.15,
                w_subst: float = 0.30, w_topic: float = 0.25,
                max_freq: int = 0,
                aux_keyword: str = '') -> list[dict]:
        if not self._built:
            raise RuntimeError('请先调用 build_index(text)')

        aux = (aux_keyword or '').strip()
        if aux and aux == keyword:
            aux = ''
        dual = bool(aux)

        # ── 运行时候选池 = 持久候选 + 通道 C 扩充 ──
        if dual:
            # 双关键词：通道 C 仅在联合锚点位置展开，输出贴近 AB 共现语境
            pos_a, pos_b, _spans = self._compute_joint_anchors(
                keyword, aux, self.JOINT_WINDOW)
            extra = self._channel_c_expand(
                keyword, positions_override=pos_a or None)
            extra_b = self._channel_c_expand(
                aux, positions_override=pos_b or None)
            for w, meta in extra_b.items():
                if w in extra:
                    base = extra[w]
                    base['templates'] = (set(base.get('templates') or set())
                                         | set(meta['templates']))
                    base['origins'] = (set(base.get('origins') or set())
                                       | set(meta['origins']))
                    if not base.get('type'):
                        base['type'] = meta['type']
                    budget = 5 - len(base.get('evidence') or [])
                    if budget > 0:
                        base['evidence'] = ((base.get('evidence') or [])
                                            + list(meta['evidence'][:budget]))
                else:
                    extra[w] = meta
        else:
            extra = self._channel_c_expand(keyword)

        if extra:
            vocab = {**self._candidates, **extra}
            self._ensure_strategy_caches(extra)
        else:
            vocab = self._candidates

        raw_scores, pattern_info, joint_meta = self._run_strategies(
            keyword, vocab, aux_keyword=aux if dual else None)

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
        aux_info = vocab.get(aux) or {} if dual else {}

        # 双关键词联合证据：预算合并区间，循环里用作硬门槛与加分输入
        merged_pair_spans: list[tuple[int, int]] = []
        joint_segs: set[int] = set()
        if joint_meta:
            merged_pair_spans = self._merge_spans(joint_meta['pair_spans'])
            joint_segs = joint_meta['joint_segs']

        results = []
        for word in all_words:
            info = vocab.get(word)
            if not info:
                continue
            if not self._passes_freq_gate(info, min_freq, max_freq):
                continue

            # ── 双关键词：联合证据硬门槛 ──
            joint_window_hits = 0
            joint_seg_hits = 0
            w_segs_total = 0
            if dual:
                joint_window_hits = self._count_in_merged_spans(
                    word, merged_pair_spans)
                w_segs = self._word_segs.get(word, set())
                w_segs_total = len(w_segs)
                joint_seg_hits = (len(w_segs & joint_segs)
                                  if w_segs_total else 0)
                if (joint_window_hits < self.JOINT_GATE_MIN_HITS
                        and joint_seg_hits < self.JOINT_GATE_MIN_HITS):
                    continue

            # 缺席策略直接给 0。原本的 MISS_PENALTY 是为了配合"命中越多
            # 策略越好"的 CROSS_BONUS 机制，两者方向相反但逻辑同源；现在
            # 六策略只作为多角度捞候选的通道，命中数不再参与打分，缺席
            # 自然就是 0 贡献。
            s_char = n_char.get(word, 0.0)
            s_context = n_context.get(word, 0.0)
            s_cooccur = n_cooccur.get(word, 0.0)
            s_morph = n_morph.get(word, 0.0)
            s_subst = n_subst.get(word, 0.0)
            s_topic = n_topic.get(word, 0.0)

            score = (w_char * s_char + w_context * s_context
                     + w_cooccur * s_cooccur + w_morph * s_morph
                     + w_subst * s_subst + w_topic * s_topic)
            # hits 只用于 UI 展示 (hit_count)，不再参与打分。
            # 低频真词天然命中策略少（物理上限），用命中数加分会
            # 系统性地打压它们，与"多角度捞候选"的初衷冲突。
            hits = sum(1 for s in (s_char, s_context, s_cooccur,
                                   s_morph, s_subst, s_topic) if s > 0)

            # R5 类型先验：同类加分（加法式弱证据）
            # 双关键词模式下关闭，避免 kw_type=A.type 错误加分到 B 的亲族词
            if not dual:
                w_type = info.get('type')
                if (kw_type and w_type and w_type != 'misc'
                        and kw_type != 'misc' and w_type == kw_type):
                    score += self.TYPE_MATCH_BONUS

            # R6 模板同族
            w_templates = set(info.get('templates') or set())
            if kw_templates and (kw_templates & w_templates):
                score += self.TEMPLATE_FAMILY_BONUS

            # ── 双关键词：联合证据加分（纯位置，无分类假设）──
            if dual:
                w_freq = max(info['freq'], 1)
                ratio_win = joint_window_hits / w_freq
                ratio_seg = (joint_seg_hits / w_segs_total
                             if w_segs_total else 0.0)
                joint_ratio = min(max(ratio_win, ratio_seg), 1.0)
                score *= 1.0 + self.JOINT_BONUS_ALPHA * joint_ratio

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
        exclude_top = {keyword}
        if dual:
            exclude_top.add(aux)
        results = [r for r in results if r['word'] not in exclude_top][:top_n]

        # 双关键词：先把辅关键词置顶（让它在主关键词后、算法结果前）
        if dual:
            aux_freq = self._freq.get(aux)
            if aux_freq is None:
                aux_freq = self._text.count(aux)
            if aux_freq > 0:
                aux_ev_snips = []
                if aux_info.get('evidence'):
                    aux_ev_snips = format_evidence_snippets(
                        self._text, aux_info['evidence'], aux, max_count=2)
                aux_entry = {
                    'word': aux,
                    'freq': aux_freq,
                    'score': 9999.0,
                    'strategies': '用户输入(辅)',
                    'hit_count': 6,
                    'matched_patterns': '',
                    'tier': aux_info.get('tier', 'L1'),
                    'type': aux_info.get('type') or '',
                    'templates': '、'.join(
                        sorted(aux_info.get('templates') or set())),
                    'evidence': '  ∥  '.join(aux_ev_snips),
                }
                results = [aux_entry] + results

        kw_freq = self._freq.get(keyword)
        if kw_freq is None:
            kw_freq = self._text.count(keyword)
        if kw_freq > 0:
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

    def _run_strategies(self, keyword: str, vocab: dict,
                        aux_keyword: str | None = None):
        """返回 (raw_scores_dict, pattern_info, joint_meta)

        raw_scores_dict 的 key 固定为六个策略名：
          char_overlap / context_pattern / cooccurrence /
          morpheme / substitution / co_topic

        aux_keyword 非空 → 双关键词联合模式：
          • char/morph：A 与 B 各跑一次后逐词取 max（任一边即可贡献信号）
          • context/cooccur：seeds=[A,B] 但仅在联合锚点位置抽取/统计
          • co_topic：seed_segs 改用 word_segs[A] ∩ word_segs[B]
          • substitution：保持原行为（seeds=[A,B] 全位置），由用户后续调权
        joint_meta：单关键词时为 None；双关键词时含 pair_spans / joint_segs，
                    供 extract 做硬门槛与加分。
        """
        raw_char = self._strategy_char_overlap(keyword, vocab).scores
        raw_morph = self._strategy_morpheme(keyword, vocab).scores

        joint_meta = None

        if aux_keyword:
            # ── 字面、构词通道：A、B 各跑一次取 max ──
            raw_char = self._max_fuse(
                raw_char,
                self._strategy_char_overlap(aux_keyword, vocab).scores,
            )
            raw_morph = self._max_fuse(
                raw_morph,
                self._strategy_morpheme(aux_keyword, vocab).scores,
            )

            # ── 联合锚点 / 联合段落 ──
            pos_a, pos_b, pair_spans = self._compute_joint_anchors(
                keyword, aux_keyword, self.JOINT_WINDOW)
            joint_segs = self._compute_joint_segments(keyword, aux_keyword)
            joint_meta = {
                'pair_spans': pair_spans,
                'joint_segs': joint_segs,
                'pos_a': pos_a,
                'pos_b': pos_b,
            }

            seeds = [keyword, aux_keyword]
            seed_positions = {keyword: pos_a, aux_keyword: pos_b}
            has_joint = bool(pair_spans) or bool(joint_segs)

            if has_joint:
                res_ctx = self._strategy_context_pattern(
                    seeds, vocab, seed_positions=seed_positions)
                raw_context = res_ctx.scores
                pattern_info = res_ctx.meta.get('patterns', {})
                raw_cooccur = self._strategy_cooccurrence(
                    seeds, vocab, seed_positions=seed_positions).scores
                raw_topic = self._strategy_co_topic(
                    seeds, vocab,
                    seed_segs_override=joint_segs).scores
            else:
                # 没有任何联合证据时降级为 [A,B] 双种子常规跑，避免空结果
                res_ctx = self._strategy_context_pattern(seeds, vocab)
                raw_context = res_ctx.scores
                pattern_info = res_ctx.meta.get('patterns', {})
                raw_cooccur = self._strategy_cooccurrence(seeds, vocab).scores
                raw_topic = self._strategy_co_topic(seeds, vocab).scores

            # 互替：按用户要求纳入体系，维持原行为
            raw_subst = self._strategy_substitution(seeds, vocab).scores

        else:
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
        }, pattern_info, joint_meta

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
