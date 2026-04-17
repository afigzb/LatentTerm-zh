"""快速验证新的候选池架构：
- 通道 B 的模板狙击是否在运行
- L1 / L2 的分布
- extract 的新输出字段
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import random
from collections import Counter
from core.term_extractor import TermExtractor

random.seed(42)

# 用一个短小但含多种模板的合成语料
names = ['张无忌', '赵敏', '周芷若', '谢逊', '灭绝师太']
rare_names = ['柔骨魅兔', '银月皇', '银角大王']  # 低频专名
skills = ['九阳神功', '乾坤大挪移', '太极拳']
places = ['光明顶', '武当山', '冰火岛']
groups = ['明教', '武当派', '少林派']

lines = []

# 高频人物循环（产生 L1 vocab）
for _ in range(300):
    n = random.choice(names)
    n2 = random.choice([x for x in names if x != n])
    s = random.choice(skills)
    p = random.choice(places)
    g = random.choice(groups)
    templates = [
        f'{n}说道你们且听我说',
        f'只听{n}朗声说道各位请了',
        f'{n}施展{s}',
        f'{n}来到{p}',
        f'{n}率领{g}弟子齐聚{p}',
        f'{n}使出{s}化解了那致命一击',
        f'这{s}乃{g}至高绝学',
        f'{n}接过宝物名为倚天剑',
    ]
    lines.append(random.choice(templates))

# 低频专名只出现 3~5 次（孤岛）
for rn in rare_names:
    for _ in range(random.randint(3, 5)):
        tmpl = random.choice([
            f'一头{rn}冲了出来',
            f'{rn}说道不要小看我',
            f'众人见到{rn}大吃一惊',
            f'名为{rn}的神秘生物',
        ])
        lines.append(tmpl)

text = '\n'.join(lines)
print(f'Corpus: {len(text)} chars, {len(lines)} sentences')

te = TermExtractor()
te.build_index(text)

# ── 候选池分布 ──
tier_cnt = Counter(c['tier'] for c in te._candidates.values())
type_cnt = Counter(c.get('type') or '-' for c in te._candidates.values())
print(f'\nTotal candidates: {len(te._candidates)}')
print(f'Tier distribution: {dict(tier_cnt)}')
print(f'Type distribution: {dict(type_cnt)}')

# ── 核心验证：孤岛词是否进了 L2 ──
print('\n=== Low-freq island words (should be in L2) ===')
for w in rare_names:
    c = te._candidates.get(w)
    if c is None:
        # 可能被去噪切掉，试试看带在词表
        print(f'  {w}: NOT IN CANDIDATES')
        continue
    print(f'  {w}: tier={c["tier"]} freq={c["freq"]} '
          f'type={c.get("type")} templates={sorted(c.get("templates", set()))} '
          f'origins={sorted(c.get("origins", set()))}')

# ── 模板命中 top 样本 ──
print('\n=== Samples with template hits (showing type/tier) ===')
hits_with_template = [(w, c) for w, c in te._candidates.items()
                       if c.get('templates')]
hits_with_template.sort(key=lambda x: -x[1]['freq'])
for w, c in hits_with_template[:15]:
    print(f'  {w}: tier={c["tier"]} freq={c["freq"]} '
          f'type={c.get("type")} templates={sorted(c.get("templates", set()))}')

# ── extract 新字段测试 ──
print('\n=== Extract (keyword=魂兽 - 映射为 creature，但语料里没有，看类型先验 fallback) ===')
print('\n=== Extract (keyword=张无忌) ===')
results = te.extract('张无忌', top_n=10, min_freq=2)
for r in results:
    print(f'  {r["word"]}: freq={r["freq"]} score={r["score"]:.3f} '
          f'tier={r.get("tier")} type={r.get("type")} '
          f'templates={r.get("templates")}')
    ev = r.get('evidence') or ''
    if ev:
        print(f'    EVIDENCE: {ev[:120]}')

print('\n=== Extract (keyword=柔骨魅兔 - 孤岛专名) ===')
results2 = te.extract('柔骨魅兔', top_n=10, min_freq=2)
for r in results2:
    print(f'  {r["word"]}: freq={r["freq"]} score={r["score"]:.3f} '
          f'tier={r.get("tier")} type={r.get("type")} '
          f'templates={r.get("templates")}')
    ev = r.get('evidence') or ''
    if ev:
        print(f'    EVIDENCE: {ev[:120]}')
