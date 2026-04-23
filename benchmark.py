"""
AIWA 2.0 运行时数据采集脚本
==========================================

对一本 GB18030 编码的小说跑完整流水线，采集三类数据：

1. 各步骤算法耗时（解码 / 清洗 / build_index / 单关键词 extract / 双关键词 extract），
   含每阶段的 cProfile 函数级剖析；
2. 分词结果（build_index 产出的词表 + 完整候选池）；
3. 两份术语提取结果：
     - 单关键词：王冬
     - 双关键词联合模式：魂兽 + 魂环

用法：
    python benchmark.py
    python benchmark.py --input "斗罗大陆II绝世唐门.txt" --encoding gb18030

输出目录默认 ./benchmark_output/
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import os
import platform
import pstats
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.text_cleaner import clean_text
from core.term_extractor import TermExtractor, DEFAULT_CONFIG
from core.dict_filter import DictFilter
from core.filters.linguistic import DEFAULT_LING_PIPELINE, L1_LING_PIPELINE


_TYPE_ZH = {
    'person': '人物', 'place': '地点', 'creature': '生物',
    'skill': '技·物', 'group': '组织', 'misc': '其他', '': '',
}


# ──────────────────────────────────────────────────────────────
# 工具：计时 + 剖析 + 内存峰值
# ──────────────────────────────────────────────────────────────
def run_with_profile(label: str, fn, *args, **kwargs):
    """执行 fn(*args, **kwargs)，返回 (result, stage_info)。

    stage_info 包含 wall 时间、cpu 时间、内存峰值、profile 文本。
    """
    profiler = cProfile.Profile()
    tracemalloc.start()
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()

    profiler.enable()
    try:
        result = fn(*args, **kwargs)
    finally:
        profiler.disable()

    t_wall = time.perf_counter() - t_wall0
    t_cpu = time.process_time() - t_cpu0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats('cumulative').print_stats(40)
    stats.sort_stats('tottime').print_stats(20)
    profile_text = buf.getvalue()

    info = {
        'label': label,
        'wall_sec': t_wall,
        'cpu_sec': t_cpu,
        'peak_mem_mb': peak / (1024 * 1024),
        'profile': profile_text,
    }
    print(f"  [{label}] wall={t_wall:.3f}s  cpu={t_cpu:.3f}s  "
          f"peak_mem={info['peak_mem_mb']:.1f}MB")
    return result, info


def write_text(path: Path, content: str):
    path.write_text(content, encoding='utf-8')


def write_profile(out_dir: Path, info: dict):
    safe = info['label'].replace('/', '_').replace(' ', '_')
    (out_dir / f"profile_{safe}.txt").write_text(
        f"=== {info['label']} ===\n"
        f"wall_sec    : {info['wall_sec']:.6f}\n"
        f"cpu_sec     : {info['cpu_sec']:.6f}\n"
        f"peak_mem_mb : {info['peak_mem_mb']:.3f}\n\n"
        f"{info['profile']}",
        encoding='utf-8',
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    # utf-8-sig 让 Excel 直接双击打开不乱码
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ──────────────────────────────────────────────────────────────
# 结果序列化
# ──────────────────────────────────────────────────────────────
def dump_vocab_csv(extractor: TermExtractor, path: Path):
    """build_index 产出的严格词表 self._vocab（L1 词表，凝固度+自由度双通过）。"""
    vocab = extractor._vocab
    rows = []
    for word, info in vocab.items():
        rows.append({
            '词语':   word,
            '长度':   len(word),
            '频次':   info.get('freq', 0),
            '凝固度': info.get('cohesion', 0.0),
            '自由度': info.get('freedom', 0.0),
        })
    rows.sort(key=lambda r: (-r['频次'], -r['长度'], r['词语']))
    write_csv(path, rows, ['词语', '长度', '频次', '凝固度', '自由度'])
    return len(rows)


def dump_candidates_csv(extractor: TermExtractor, path: Path):
    """完整候选池 self._candidates（L1 + 模板挖矿 L2）。"""
    cands = extractor._candidates
    rows = []
    for word, info in cands.items():
        templates = info.get('templates') or set()
        origins = info.get('origins') or set()
        rows.append({
            '词语':       word,
            '长度':       len(word),
            '层':         info.get('tier', ''),
            '类型':       _TYPE_ZH.get(info.get('type', ''), info.get('type', '')),
            '频次':       info.get('freq', 0),
            '凝固度':     round(float(info.get('cohesion', 0.0) or 0.0), 4),
            '自由度':     round(float(info.get('freedom', 0.0) or 0.0), 4),
            '命中模板':   '|'.join(sorted(templates)) if templates else '',
            '来源通道':   '|'.join(sorted(origins)) if origins else '',
        })
    rows.sort(key=lambda r: (-r['频次'], -r['长度'], r['词语']))
    write_csv(path, rows, [
        '词语', '长度', '层', '类型', '频次', '凝固度', '自由度',
        '命中模板', '来源通道',
    ])
    return len(rows)


def dump_extract_csv(results: list[dict], path: Path):
    """把 extractor.extract() 返回值（已含 pos_group）扁平化写成 CSV。"""
    rows = []
    for r in results:
        children = r.get('children', [])
        rows.append({
            '词语':       r['word'],
            '层':         r.get('tier', ''),
            '类型':       _TYPE_ZH.get(r.get('type', ''), r.get('type', '')),
            '词性':       r.get('pos_group', ''),
            '出现频次':   r['freq'],
            '综合评分':   round(float(r.get('score', 0.0)), 6),
            '命中策略':   r.get('strategies', ''),
            '命中数':     r.get('hit_count', 0),
            '命中模板':   r.get('templates', ''),
            '匹配模式':   r.get('matched_patterns', ''),
            '原文证据':   r.get('evidence', ''),
            '子术语':     '、'.join(c['word'] for c in children),
            '子术语数':   len(children),
        })
    write_csv(path, rows, [
        '词语', '层', '类型', '词性', '出现频次', '综合评分',
        '命中策略', '命中数', '命中模板', '匹配模式', '原文证据',
        '子术语', '子术语数',
    ])
    return len(rows)


def dump_children_csv(results: list[dict], path: Path):
    """主词 → 子术语 的一对多展开表，便于分析层级关系。"""
    rows = []
    for r in results:
        for c in r.get('children', []):
            rows.append({
                '主词':       r['word'],
                '主词频次':   r['freq'],
                '子术语':     c['word'],
                '子术语频次': c.get('freq', 0),
                '子术语评分': round(float(c.get('score', 0.0)), 6),
                '子术语层':   c.get('tier', ''),
                '子术语类型': _TYPE_ZH.get(c.get('type', ''), c.get('type', '')),
                '子术语词性': c.get('pos_group', ''),
            })
    write_csv(path, rows, [
        '主词', '主词频次', '子术语', '子术语频次', '子术语评分',
        '子术语层', '子术语类型', '子术语词性',
    ])
    return len(rows)


def dump_filter_stats_csv(path: Path):
    """汇总 L1 和 L2 管道的拒绝统计并写出 CSV。"""
    merged_stats: dict[str, dict[str, int]] = {}
    
    for pipeline, prefix in [(L1_LING_PIPELINE, 'L1'), (DEFAULT_LING_PIPELINE, 'L2')]:
        for reason, word_counts in pipeline.stats.items():
            full_reason = f"{prefix}_{reason}"
            if full_reason not in merged_stats:
                merged_stats[full_reason] = {}
            for w, c in word_counts.items():
                merged_stats[full_reason][w] = merged_stats[full_reason].get(w, 0) + c

    rows = []
    for reason, word_counts in merged_stats.items():
        # 对每个 reason，按频次降序取前 100 个词作为 sample 展示
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        total_count = sum(c for w, c in sorted_words)
        unique_count = len(sorted_words)
        
        # 拼装 sample 字符串，例如 "我们(150) | 这个(120) | ..."
        samples = " | ".join(f"{w}({c})" for w, c in sorted_words[:100])
        
        rows.append({
            '拒绝原因': reason,
            '总拦截频次': total_count,
            '独立词数': unique_count,
            '拦截样本(Top100)': samples
        })
        
    rows.sort(key=lambda r: -r['总拦截频次'])
    write_csv(path, rows, ['拒绝原因', '总拦截频次', '独立词数', '拦截样本(Top100)'])
    return len(rows)


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='斗罗大陆II绝世唐门.txt',
                        help='输入的小说文件路径')
    parser.add_argument('--encoding', default='gb18030',
                        help="文件编码，默认 gb18030；设为 auto 使用 charset_normalizer")
    parser.add_argument('--enable-cleaning', action='store_true', default=True,
                        help='启用文本清洗（去脏字符 + NFKC），默认开启')
    parser.add_argument('--no-cleaning', dest='enable_cleaning',
                        action='store_false', help='禁用文本清洗')
    parser.add_argument('--min-len', type=int, default=2)
    parser.add_argument('--max-len', type=int, default=8)
    parser.add_argument('--min-freq', type=int, default=DEFAULT_CONFIG['min_freq'])
    parser.add_argument('--max-freq', type=int, default=DEFAULT_CONFIG['max_freq'])
    parser.add_argument('--top-n', type=int, default=DEFAULT_CONFIG['top_n'])
    parser.add_argument('--keyword-single', default='王冬')
    parser.add_argument('--keyword-dual-a', default='魂兽')
    parser.add_argument('--keyword-dual-b', default='魂环')
    parser.add_argument('--output-dir', default='benchmark_output')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    input_path = (project_root / args.input).resolve()
    out_dir = (project_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[错误] 找不到输入文件：{input_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"AIWA 2.0 Benchmark  @ {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 70)
    print(f"输入文件  : {input_path}")
    print(f"文件大小  : {input_path.stat().st_size:,} 字节")
    print(f"目标编码  : {args.encoding}")
    print(f"输出目录  : {out_dir}")
    print("-" * 70)

    stage_infos: list[dict] = []

    # ── Stage 1：读取字节 ─────────────────────────────────────
    print("\n[Stage 1] 读取原始字节")
    def _read_bytes():
        with open(input_path, 'rb') as f:
            return f.read()
    raw_bytes, info = run_with_profile('01_read_bytes', _read_bytes)
    stage_infos.append(info)
    print(f"  读取字节数: {len(raw_bytes):,}")

    # ── Stage 2：解码 + 清洗（clean_text）──────────────────────
    print("\n[Stage 2] 解码 + 文本清洗 (clean_text)")
    enc_arg = None if args.encoding == 'auto' else args.encoding
    (cleaned_text, raw_text, encoding, stats), info = run_with_profile(
        '02_clean_text',
        clean_text,
        raw_bytes,
        encoding=enc_arg,
        enable_cleaning=args.enable_cleaning,
    )
    stage_infos.append(info)
    if cleaned_text is None:
        print(f"[错误] 解码失败：{raw_text}")
        sys.exit(1)
    print(f"  检测到编码: {stats.get('detected_encoding')}")
    print(f"  原始字符数: {stats.get('original_length'):,}")
    print(f"  清洗后字符: {stats.get('final_length'):,}")
    print(f"  清除脏字符: {stats.get('dirty_chars_removed')}")

    # 开启过滤统计
    DEFAULT_LING_PIPELINE.enable_stats = True
    L1_LING_PIPELINE.enable_stats = True

    # ── Stage 3：build_index（分词核心算法）────────────────────
    print("\n[Stage 3] 构建词表索引 (build_index) —— 分词核心算法")
    extractor = TermExtractor(min_len=args.min_len, max_len=args.max_len)
    _, info = run_with_profile(
        '03_build_index',
        extractor.build_index,
        cleaned_text,
    )
    stage_infos.append(info)
    n_vocab = len(extractor._vocab)
    n_vocab_relaxed = len(extractor._vocab_relaxed)
    n_candidates = len(extractor._candidates)
    n_freq = len(extractor._freq)
    print(f"  n-gram 频次表大小: {n_freq:,}")
    print(f"  L1 严格词表大小  : {n_vocab:,}")
    print(f"  L1 宽松词表大小  : {n_vocab_relaxed:,}")
    print(f"  总候选池大小     : {n_candidates:,}")

    # ── Stage 4：加载 jieba 词典（DictFilter）─────────────────
    print("\n[Stage 4] 加载 jieba 词典 (DictFilter)")
    dict_filter, info = run_with_profile(
        '04_load_dict_filter',
        DictFilter,
    )
    stage_infos.append(info)
    print(f"  词典词条数: {len(dict_filter._word_pos):,}")

    # ── Stage 5：单关键词提取 王冬 ─────────────────────────────
    print(f"\n[Stage 5] 单关键词提取：{args.keyword_single}")
    results_single, info = run_with_profile(
        '05_extract_single',
        extractor.extract,
        args.keyword_single,
        top_n=args.top_n,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        w_char=DEFAULT_CONFIG['w_char'],
        w_context=DEFAULT_CONFIG['w_context'],
        w_cooccur=DEFAULT_CONFIG['w_cooccur'],
        w_morph=DEFAULT_CONFIG['w_morph'],
        w_subst=DEFAULT_CONFIG['w_subst'],
        w_topic=DEFAULT_CONFIG['w_topic'],
    )
    stage_infos.append(info)
    dict_filter.tag_results(results_single)
    n_single_children = sum(len(r.get('children', [])) for r in results_single)
    print(f"  主词数    : {len(results_single):,}")
    print(f"  子术语数  : {n_single_children:,}")
    print(f"  合计候选  : {len(results_single) + n_single_children:,}")

    # ── Stage 6：双关键词联合提取 魂兽 + 魂环 ────────────────
    print(f"\n[Stage 6] 双关键词联合提取：{args.keyword_dual_a} + {args.keyword_dual_b}")
    results_dual, info = run_with_profile(
        '06_extract_dual',
        extractor.extract,
        args.keyword_dual_a,
        top_n=args.top_n,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        w_char=DEFAULT_CONFIG['w_char'],
        w_context=DEFAULT_CONFIG['w_context'],
        w_cooccur=DEFAULT_CONFIG['w_cooccur'],
        w_morph=DEFAULT_CONFIG['w_morph'],
        w_subst=DEFAULT_CONFIG['w_subst'],
        w_topic=DEFAULT_CONFIG['w_topic'],
        aux_keyword=args.keyword_dual_b,
    )
    stage_infos.append(info)
    dict_filter.tag_results(results_dual)
    n_dual_children = sum(len(r.get('children', [])) for r in results_dual)
    print(f"  主词数    : {len(results_dual):,}")
    print(f"  子术语数  : {n_dual_children:,}")
    print(f"  合计候选  : {len(results_dual) + n_dual_children:,}")

    # ── 写出所有结果文件 ──────────────────────────────────────
    print("\n[写出结果文件]")

    n_vocab_rows = dump_vocab_csv(extractor, out_dir / '分词结果_L1词表.csv')
    print(f"  分词结果_L1词表.csv          —— {n_vocab_rows:,} 行")

    n_cand_rows = dump_candidates_csv(extractor, out_dir / '分词结果_全量候选池.csv')
    print(f"  分词结果_全量候选池.csv      —— {n_cand_rows:,} 行")

    n_filter_stats = dump_filter_stats_csv(out_dir / '过滤统计_filter_stats.csv')
    print(f"  过滤统计_filter_stats.csv    —— {n_filter_stats:,} 行")

    n_s_rows = dump_extract_csv(results_single, out_dir / f'提取_{args.keyword_single}.csv')
    n_s_child = dump_children_csv(results_single, out_dir / f'提取_{args.keyword_single}_子术语展开.csv')
    print(f"  提取_{args.keyword_single}.csv                —— {n_s_rows:,} 行主词")
    print(f"  提取_{args.keyword_single}_子术语展开.csv     —— {n_s_child:,} 行子术语")

    dual_name = f"{args.keyword_dual_a}+{args.keyword_dual_b}"
    n_d_rows = dump_extract_csv(results_dual, out_dir / f'提取_{dual_name}.csv')
    n_d_child = dump_children_csv(results_dual, out_dir / f'提取_{dual_name}_子术语展开.csv')
    print(f"  提取_{dual_name}.csv         —— {n_d_rows:,} 行主词")
    print(f"  提取_{dual_name}_子术语展开.csv —— {n_d_child:,} 行子术语")

    for si in stage_infos:
        write_profile(out_dir, si)

    # ── 运行时总览 runtime_log.txt ────────────────────────────
    total_wall = sum(si['wall_sec'] for si in stage_infos)
    total_cpu = sum(si['cpu_sec'] for si in stage_infos)
    peak_mem = max(si['peak_mem_mb'] for si in stage_infos)

    lines: list[str] = []
    lines.append("AIWA 2.0 运行时数据采集报告")
    lines.append("=" * 70)
    lines.append(f"生成时间     : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Python 版本  : {sys.version.split()[0]}")
    lines.append(f"平台         : {platform.platform()}")
    lines.append(f"处理器       : {platform.processor() or platform.machine()}")
    lines.append("")
    lines.append("【输入数据】")
    lines.append(f"  文件路径     : {input_path}")
    lines.append(f"  文件大小     : {input_path.stat().st_size:,} 字节")
    lines.append(f"  指定编码     : {args.encoding}")
    lines.append(f"  实际编码     : {stats.get('detected_encoding')}")
    lines.append(f"  原始字符数   : {stats.get('original_length'):,}")
    lines.append(f"  清洗字符数   : {stats.get('final_length'):,}")
    lines.append(f"  清除脏字符   : {stats.get('dirty_chars_removed')}")
    lines.append(f"  启用清洗     : {args.enable_cleaning}")
    lines.append("")
    lines.append("【参数配置】")
    lines.append(f"  min_len / max_len : {args.min_len} / {args.max_len}")
    lines.append(f"  min_freq          : {args.min_freq}")
    lines.append(f"  max_freq          : {args.max_freq}  (0 = 无上限)")
    lines.append(f"  top_n             : {args.top_n}")
    lines.append(f"  单关键词          : {args.keyword_single}")
    lines.append(f"  双关键词 (主+辅)  : {args.keyword_dual_a} + {args.keyword_dual_b}")
    lines.append("")
    lines.append("【分词 / 词表规模】")
    lines.append(f"  n-gram 频次表    : {n_freq:,}")
    lines.append(f"  L1 严格词表      : {n_vocab:,}")
    lines.append(f"  L1 宽松词表      : {n_vocab_relaxed:,}")
    lines.append(f"  完整候选池       : {n_candidates:,}")
    lines.append("")
    lines.append("【提取结果规模】")
    lines.append(f"  {args.keyword_single:<8} 主词 / 子术语 : "
                 f"{len(results_single):,} / {n_single_children:,}")
    lines.append(f"  {dual_name:<8} 主词 / 子术语 : "
                 f"{len(results_dual):,} / {n_dual_children:,}")
    lines.append("")
    lines.append("【各阶段耗时 (按执行顺序)】")
    lines.append(
        f"  {'阶段':<24}{'Wall(s)':>10}{'CPU(s)':>10}{'Peak Mem(MB)':>16}"
    )
    lines.append("  " + "-" * 60)
    for si in stage_infos:
        lines.append(
            f"  {si['label']:<24}"
            f"{si['wall_sec']:>10.3f}"
            f"{si['cpu_sec']:>10.3f}"
            f"{si['peak_mem_mb']:>16.2f}"
        )
    lines.append("  " + "-" * 60)
    lines.append(
        f"  {'合计':<24}{total_wall:>10.3f}{total_cpu:>10.3f}{peak_mem:>16.2f}"
    )
    lines.append("")
    lines.append("【输出文件清单】")
    lines.append("  ─ 结果文件（分析用）")
    lines.append("    分词结果_L1词表.csv               —— L1 严格词表（双通过）")
    lines.append("    分词结果_全量候选池.csv           —— L1 + 模板挖矿 L2 全部候选")
    lines.append("    过滤统计_filter_stats.csv         —— 语言学过滤器拦截原因与高频词采样")
    lines.append(f"    提取_{args.keyword_single}.csv                —— 单关键词提取主词表")
    lines.append(f"    提取_{args.keyword_single}_子术语展开.csv     —— 主词对应子术语展开")
    lines.append(f"    提取_{dual_name}.csv         —— 双关键词联合提取主词表")
    lines.append(f"    提取_{dual_name}_子术语展开.csv —— 主词对应子术语展开")
    lines.append("  ─ 运行时剖析（调优用）")
    for si in stage_infos:
        safe = si['label'].replace('/', '_').replace(' ', '_')
        lines.append(f"    profile_{safe}.txt")

    write_text(out_dir / 'runtime_log.txt', '\n'.join(lines) + '\n')

    print("\n" + "=" * 70)
    print(f"总耗时 Wall: {total_wall:.3f}s   CPU: {total_cpu:.3f}s   "
          f"峰值内存: {peak_mem:.2f} MB")
    print(f"所有结果已写入: {out_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
