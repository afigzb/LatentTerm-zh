"""验证两步法核心机制：
  - "灭绝师太" ✓、"灭绝" ✓、"师太" ✓
  - "灭绝师" ✗（右边永远是"太" → 自由度=0 → 不进 Trie）
  - "植物系魂兽" ✓、"植物系" ✓、"魂兽" ✓
  - "物系魂兽" ✗（左边永远是"植" → 自由度=0 → 不进 Trie）
"""
import random
from core.term_extractor import TermExtractor
from core._utils import _entropy

random.seed(42)

names = ['张无忌', '赵敏', '周芷若', '谢逊', '殷素素', '杨逍',
         '范遥', '韦一笑', '殷天正', '灭绝师太']
skills = ['九阳神功', '乾坤大挪移', '太极拳', '太极剑', '圣火令']
weapons = ['倚天剑', '屠龙刀']
places = ['光明顶', '冰火岛', '武当山', '少林寺', '蝴蝶谷']
groups = ['明教', '武当派', '少林派', '峨眉派']
adj = ['武林', '天下', '江湖', '中原']

templates = [
    '{n}说道此事不可等闲视之',
    '{n}说道你们且听我说',
    '只听{n}朗声说道各位请了',
    '{n}微微一笑说道你太小看我了',
    '这时{n}忽然大喝一声冲了上来',
    '众人看向{n}面面相觑',
    '那{n}心中暗暗打定主意不再犹豫',
    '{n}暗自思忖此人武功深不可测',
    '{n}一掌拍出震得地面微微颤动',
    '一旁的{n}默默无言良久不语',
    '只见{n}手持{w}威风凛凛',
    '{n}修炼{s}已至大成境界',
    '全靠{s}护体{n}方能抵御那一掌',
    '若非{n}及时赶到后果不堪设想',
    '{n}率领{g}弟子齐聚{p}',
    '提起{n}谁人不知谁人不晓',
    '在{p}上{n}力战群雄',
    '{n}拉住{n2}的手说道你别走',
    '{n}看着{n2}心中百感交集',
    '{n}与{n2}联手对敌将来犯之人尽数击退',
    '待{n}走远{n2}才长叹一声',
    '{s}乃{a}至高绝学',
    '传闻{s}为前辈高人所创',
    '{n}施展{s}将敌人逼退数丈',
    '练成{s}之人已有百年未见',
    '{s}配合{s2}更是天下无敌',
    '{g}弟子忠心耿耿誓死追随教主',
    '{g}教主之位空悬已久',
    '当年{g}盛极一时威震{a}',
    '{g}总坛设在{p}',
    '{p}上有{g}历代教主遗留的秘籍',
    '各派高手围攻{p}',
    '{w}削铁如泥乃{a}至宝',
    '得到{w}者号令天下莫敢不从',
    '{n}接过{w}心中感慨万千',
    '谁能同时得到{w}与{w2}便可一统{a}',
    '{n}与{n2}携手走出{p}共赴{a}',
    '在那{p}之巅{n}立下赫赫战功',
    '{n}凝视远方心想若能再见{n2}一面该多好',
    '{n}轻叹一声缓缓转身离去',
    '{n}抱拳说道在下告辞',
    '{n}使出{s}化解了那致命一击',
    '{n}厉声说道休要胡说',
    '传闻{n}已修炼{s}至第九重',
    '{n}双掌齐出招招凌厉',
    '{n}一剑刺出剑光如虹',
    '{n}纵身一跃稳稳落在{p}之上',
    '{n}冷冷注视着{n2}一言不发',
    '{n}苦笑道此事说来话长',
    '那{n}不过是个无名之辈',
    '义父曾教{n}读书习武',
    '{n}摇头道我不会答应你',
    '{n}心知此事非同小可',
    '{n}面色凝重缓缓说道',
    '{n}推开房门走了进去',
    '{n}手握{w}傲然而立',
    '{n}望着{p}方向出神',
    '想起{n}心中不禁一酸',
    '这{s}果然名不虚传',
    '{n}左手使{s}右手握{w}',
]

sents = []
for _ in range(2000):
    tmpl = random.choice(templates)
    n = random.choice(names)
    n2 = random.choice([x for x in names if x != n])
    s = random.choice(skills)
    s2 = random.choice([x for x in skills if x != s])
    w = random.choice(weapons)
    w2 = random.choice([x for x in weapons if x != w])
    p = random.choice(places)
    g = random.choice(groups)
    a = random.choice(adj)
    sent = tmpl.format(n=n, n2=n2, s=s, s2=s2, w=w, w2=w2, p=p, g=g, a=a)
    sents.append(sent)

text = '\n'.join(sents)
print(f'Corpus: {len(text)} chars, {len(sents)} sentences')

te = TermExtractor()
te.build_index(text)

print(f'total_chars: {te._total_chars}')
print(f'vocab: {len(te._vocab)}, relaxed: {len(te._vocab_relaxed)}')

# ── 核心验证：跨边界垃圾是否被过滤 ──
print('\n=== Cross-boundary fragment check ===')
garbage = ['灭绝师', '绝师太', '持倚天', '天剑威', '阳神功', '坤大挪']
for w in garbage:
    in_v = w in te._vocab
    in_r = w in te._vocab_relaxed
    print(f'  {w}: vocab={in_v}, relaxed={in_r}'
          f'  {"GOOD (filtered)" if not in_v and not in_r else "BAD (leaked!)"}')

# ── 核心验证：合法词是否保留 ──
print('\n=== Legitimate word check ===')
good_words = ['灭绝师太', '张无忌', '赵敏', '周芷若', '九阳神功',
              '乾坤大挪移', '倚天剑', '屠龙刀', '明教', '光明顶',
              '灭绝', '师太', '太极拳', '武当派', '少林寺']
for w in good_words:
    in_v = w in te._vocab
    in_r = w in te._vocab_relaxed
    v = te._vocab.get(w, te._vocab_relaxed.get(w))
    info = f'freq={v["freq"]}, coh={v["cohesion"]}, free={v["freedom"]}' if v else 'MISSING'
    status = 'OK' if in_v or in_r else 'MISS'
    print(f'  {w}: {status}  {info}')

# ── 词表总览 ──
print(f'\nTop vocab entries ({len(te._vocab)} total):')
for w, v in sorted(te._vocab.items(), key=lambda x: -x[1]['freq'])[:25]:
    print(f'  {w}: freq={v["freq"]}, coh={v["cohesion"]}, free={v["freedom"]}')

# ── extract 端到端测试 ──
print('\nExtract (keyword=张无忌):')
results = te.extract('张无忌', top_n=15)
for r in results:
    children = r.get('children', [])
    ch_str = f'  [{", ".join(c["word"] for c in children)}]' if children else ''
    print(f'  {r["word"]}: freq={r["freq"]}, score={r["score"]:.3f}, '
          f'strat={r["strategies"]}{ch_str}')
