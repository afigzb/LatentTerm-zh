"""验证 L1 分解反验：'真空地带' 这类短语拼凑能被识别丢弃，
真专名如 '九阳神功' 即使看起来能拆，由于 Trie 长词优先切分
压低子词频次，其子词往往也不在 L1 vocab → 保留。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from core._pattern_miner import can_decompose_by_l1


# ── 直接单元测试 can_decompose_by_l1 ──

print('=== 直接测试 can_decompose_by_l1 ===')
l1 = {'真空', '地带', '银月', '倚天', '九幽', '明教', '武当',
      '施展', '使出', '修炼', '神秘', '生物',
      '九阳', '神功'}   # 注意：这里人为把两部分都放进去

test_cases = [
    # (词, 期望能分解, 说明)
    ('真空地带',  True,  '真空+地带 → 短语拼凑'),
    ('神秘生物',  True,  '神秘+生物 → 短语拼凑'),
    ('银月皇',    False, '银月+皇(单字) → 不能分解'),
    ('倚天剑',    False, '倚天+剑(单字) → 不能分解'),
    ('九幽',      False, '长度不够(4字起)'),
    ('明教弟子',  False, '"弟子"不在 L1 词集 → 不能分解'),
    ('九阳神功',  True,  '两部分都在 L1 → 被判为短语(但实际场景中两部分不一定都进L1)'),
    ('施展使出',  True,  '两个 L1 词硬接'),
    ('xxxxx',    False, '非汉字，全部不在 L1'),
]

failed = 0
for w, expect, note in test_cases:
    got = can_decompose_by_l1(w, l1)
    ok = '✓' if got == expect else '✗'
    if got != expect:
        failed += 1
    print(f'  {ok} {w:12s} got={got!s:5s} expect={expect!s:5s}  # {note}')
print(f'\n失败: {failed}/{len(test_cases)}')


# ── 实际场景：Trie 切分保护真词 ──
# 构造语料：'九阳神功' 总是整体出现 → L1 Trie 会把'九阳'/'神功'子词频次压低
# 而'真空地带'是偶然组合 → '真空'和'地带'各自高频 → 被判为短语
print('\n=== 实际场景：用 TermExtractor 跑完整流程 ===')

from core.term_extractor import TermExtractor

lines = []
# '九阳神功' 几乎只作为整体出现
for _ in range(40):
    lines.append('张无忌修炼九阳神功威力惊人')
    lines.append('只见一道金光原是九阳神功护体')
    lines.append('施展九阳神功化解了攻势')

# '真空' 和 '地带' 各自频繁独立出现
for _ in range(40):
    lines.append('真空之中无物可生')
    lines.append('这里形成真空可怕至极')
    lines.append('沙漠地带干燥异常')
    lines.append('走到山脉地带就要小心')
# 再让它们偶尔组合一次
for _ in range(3):
    lines.append('真空地带之中暗藏凶险')

# 添加一些专名让 B 模板能激活
for _ in range(20):
    lines.append('张无忌说道你们小心')
    lines.append('赵敏喝道快退')

text = '\n'.join(lines)
print(f'Corpus: {len(text)} chars')

te = TermExtractor()
te.build_index(text)

interested = ['九阳神功', '九阳', '神功', '真空地带', '真空', '地带']
for w in interested:
    c = te._candidates.get(w)
    if c is None:
        print(f'  {w}: 不在候选池（被过滤）')
    else:
        print(f'  {w}: tier={c["tier"]} freq={c["freq"]} '
              f'origins={sorted(c.get("origins", set()))} '
              f'templates={sorted(c.get("templates", set()))}')
