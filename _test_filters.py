"""针对用户报告的垃圾候选，逐条验证是否被过滤。"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from core._pattern_miner import _valid_candidate, _BLACK_WORDS

# 用户反馈里出现的垃圾候选 —— 全部必须被拒
BAD_CASES = [
    # ── 组织类误报 ──
    '一名斗宗', '三名斗宗', '习会', '种族', '势力',
    # ── 生物类误报 ──
    '天材地宝', '强者', '真空地带', '目光', '佣兵',
    # ── 人物类误报（X地Y 副词结构）──
    '清楚地知', '做记名', '美其名', '忍不住地轻',
    '淡淡地开口', '无奈地开口', '惊骇地失声', '他清楚地知',
    '虽然那名', '客气地回', '冲着这名', '最后在那名',
    '见到这名', '暴怒地咆哮', '她清楚地知', '淡淡地轻',
    '淡淡地吩咐', '开怀地大',
    # ── 量词开头误报 ──
    '另外一名', '另外两名', '记名', '核心', '二人',
    '两名慕兰', '三名白衣', '三名冰河谷', '三名云岚宗',
]

# 真正的专名 —— 全部必须通过
GOOD_CASES = [
    '萧炎', '张无忌', '赵敏', '周芷若', '谢逊', '灭绝师太',
    '九幽地冥蟒',   # 注意:"地"在中间,但这是网文生物名
                    # 这个案例会暴露一个真实的 trade-off（见下）
    '九阳神功', '乾坤大挪移', '太极拳',
    '光明顶', '武当山', '冰火岛',
    '明教', '武当派', '少林派', '丐帮',
    '倚天剑', '银月皇', '银角大王',
]

print('=== BAD cases (should be rejected) ===')
failed_bad = []
for w in BAD_CASES:
    ok = _valid_candidate(w)
    status = 'FAIL (未过滤)' if ok else 'OK 拒绝'
    if ok:
        failed_bad.append(w)
    print(f'  {status:15s} {w}')

print('\n=== GOOD cases (should pass) ===')
failed_good = []
for w in GOOD_CASES:
    ok = _valid_candidate(w)
    status = 'OK 通过' if ok else 'FAIL (误伤)'
    if not ok:
        failed_good.append(w)
    print(f'  {status:15s} {w}')

print('\n---')
print(f'垃圾漏网: {len(failed_bad)}/{len(BAD_CASES)}  {failed_bad}')
print(f'专名误伤: {len(failed_good)}/{len(GOOD_CASES)}  {failed_good}')
