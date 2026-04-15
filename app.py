import streamlit as st
import pandas as pd
from core.text_cleaner import clean_text
from core.term_extractor import TermExtractor, MODE_PRESETS

st.set_page_config(page_title="AIWA 文本调试台", layout="wide")
st.title("AIWA 2.0 - 文本调试台")

# ── 文件读取设置 ──────────────────────────────────────────────────────────
_ENC_OPTIONS = {
    "auto":    "自动检测（charset_normalizer）",
    "utf-8":   "UTF-8",
    "gb18030": "GB18030（网络小说大概率是这种格式）",
    "gbk":     "GBK",
    "big5":    "Big5（繁体中文）",
    "utf-16":  "UTF-16",
    "latin-1": "Latin-1",
    "custom":  "自定义…",
}

with st.expander("⚙️ 文件读取设置", expanded=True):
    col_enc, col_cln = st.columns([3, 2])
    with col_enc:
        enc_key = st.selectbox(
            "文件编码",
            list(_ENC_OPTIONS.keys()),
            format_func=lambda k: _ENC_OPTIONS[k],
            help="不确定时选「自动检测」；绝大多数国产网文选 GB18030 即可",
        )
        custom_enc = ""
        if enc_key == "custom":
            custom_enc = st.text_input(
                "自定义编码名称",
                placeholder="例：gb2312、shift_jis、euc-kr",
            )
    with col_cln:
        enable_cleaning = st.checkbox(
            "启用文本清洗",
            value=False,
            help=(
                "勾选后执行两步清洗：\n"
                "① 去除零宽字符、控制字符等不可见脏字符\n"
                "② Unicode NFKC 标准化（全角→半角等）"
            ),
        )

encoding_to_use = (
    None        if enc_key == "auto"
    else custom_enc if enc_key == "custom"
    else enc_key
)

# ── 文件上传（全局，两个标签页共用同一份文件）──────────────────────────
uploaded_file = st.file_uploader("上传 TXT 小说文件（所有功能共用此文件）", type="txt")

if uploaded_file is not None:
    file_id = (
        uploaded_file.name
        + str(uploaded_file.size)
        + (encoding_to_use or "auto")
        + str(enable_cleaning)
    )

    if st.session_state.get("file_id") != file_id:
        with st.spinner("正在建立词表……"):
            raw_bytes = uploaded_file.read()
            cleaned_text, raw_text, encoding, stats = clean_text(
                raw_bytes,
                encoding=encoding_to_use,
                enable_cleaning=enable_cleaning,
            )

            if cleaned_text is None:
                st.error(f"解析失败！{raw_text}")
                st.stop()

            extractor = TermExtractor(min_len=2, max_len=8)
            extractor.build_index(cleaned_text)

            st.session_state["file_id"] = file_id
            st.session_state["cleaned_text"] = cleaned_text
            st.session_state["raw_text"] = raw_text
            st.session_state["encoding"] = encoding
            st.session_state["stats"] = stats
            st.session_state["extractor"] = extractor

    cleaned_text = st.session_state["cleaned_text"]
    raw_text     = st.session_state["raw_text"]
    encoding     = st.session_state["encoding"]
    stats        = st.session_state["stats"]
    extractor    = st.session_state["extractor"]

    st.success(f"文件已加载：{uploaded_file.name}（编码：{encoding}，共 {len(cleaned_text):,} 字符）")

    # ── 两个功能标签页 ────────────────────────────────────────────────────
    tab_clean, tab_extract = st.tabs(["📄 文本清洗", "🔍 术语提取"])

    # ════════════════════════════════════════════════════════════════════
    # 标签页一：文本清洗
    # ════════════════════════════════════════════════════════════════════
    with tab_clean:
        st.subheader("清洗报告")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("检测到编码",   stats["detected_encoding"])
        c2.metric("原始字符数",   f'{stats["original_length"]:,}')
        c3.metric("清除脏字符数", stats["dirty_chars_removed"])
        c4.metric("清洗后字符数", f'{stats["final_length"]:,}')

        st.divider()

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("原始文本（前 1000 字）")
            st.text_area("raw", raw_text[:1000], height=400,
                         disabled=True, label_visibility="collapsed")
        with col_r:
            st.subheader("清洗后文本（前 1000 字）")
            st.text_area("clean", cleaned_text[:1000], height=400,
                         disabled=True, label_visibility="collapsed")

    # ════════════════════════════════════════════════════════════════════
    # 标签页二：术语提取
    # ════════════════════════════════════════════════════════════════════
    with tab_extract:
        st.subheader("关键词术语聚合")

        mode = st.radio(
            "提取模式",
            list(MODE_PRESETS.keys()),
            format_func=lambda k: MODE_PRESETS[k]['label'],
            horizontal=True,
            help="均衡＝通用｜高频＝只看常见词｜低频＝挖掘稀有词（两跳发散）",
        )
        cfg = MODE_PRESETS[mode]
        st.caption(cfg['desc'])

        col_kw, col_freq, col_top = st.columns([2, 1, 1])
        with col_kw:
            keyword = st.text_input("关键词", placeholder="例：丹药、功法、剑诀")
        with col_freq:
            mn, mx = cfg['min_freq_range']
            min_freq = st.slider(
                "最低出现频次", min_value=mn, max_value=mx,
                value=cfg['min_freq'], key=f"min_freq_{mode}",
                help="低于此频次的候选词直接过滤",
            )
        with col_top:
            top_n = st.slider(
                "最多展示条数", min_value=10, max_value=1000,
                value=cfg['top_n'], key=f"top_n_{mode}",
            )

        max_freq = 0
        if mode == 'low_freq':
            max_freq = st.slider(
                "最高出现频次（排除高频词）",
                min_value=10, max_value=500, value=cfg['max_freq'],
                key="max_freq_low",
                help="高于此频次的词不会出现在结果中",
            )

        with st.expander("策略权重调节", expanded=False):
            st.caption("调节四种提取策略的权重，权重越高该策略对排序的影响越大。")
            wc1, wc2, wc3, wc4 = st.columns(4)
            with wc1:
                w_char = st.slider("字符包含", 0.0, 1.0, cfg['w_char'], 0.05,
                                   key=f"w_char_{mode}")
            with wc2:
                w_context = st.slider("上下文模式", 0.0, 1.0, cfg['w_context'], 0.05,
                                      key=f"w_context_{mode}")
            with wc3:
                w_cooccur = st.slider("共现近邻", 0.0, 1.0, cfg['w_cooccur'], 0.05,
                                      key=f"w_cooccur_{mode}")
            with wc4:
                w_morph = st.slider("构词结构", 0.0, 1.0, cfg['w_morph'], 0.05,
                                    key=f"w_morph_{mode}")

        if keyword:
            spinner_msg = f"正在以{cfg['label']}提取与「{keyword}」相关的术语……"
            with st.spinner(spinner_msg):
                results = extractor.extract(
                    keyword, top_n=top_n, min_freq=min_freq,
                    w_char=w_char, w_context=w_context,
                    w_cooccur=w_cooccur, w_morph=w_morph,
                    mode=mode, max_freq=max_freq,
                )

            if not results:
                st.warning(
                    f"没有找到与「{keyword}」相关且频次 ≥ {min_freq} 的术语，"
                    "请尝试降低最低频次或更换关键词。"
                )
            else:
                pinned = results[0] if results[0].get('strategies') == '用户输入' else None
                algo_results = results[1:] if pinned else results

                if pinned:
                    pinned_children = pinned.get('children', [])
                    suffix = (f"，含 **{len(pinned_children)}** 个从属子术语"
                              if pinned_children else "")
                    st.info(
                        f"📌 **直接命中**　关键词「**{pinned['word']}**」"
                        f"在全文中共出现 **{pinned['freq']:,}** 次{suffix}"
                    )

                algo_count = sum(
                    1 + len(r.get('children', []))
                    for r in algo_results
                )
                st.success(f"算法共找到 {algo_count} 个关联候选术语"
                           f"（{len(algo_results)} 个主词 + "
                           f"{algo_count - len(algo_results)} 个从属子术语）")

                if results:
                    flat_rows = []
                    for r in results:
                        flat_rows.append({
                            '词语': r['word'],
                            '出现频次': r['freq'],
                            '综合评分': r['score'],
                            '命中策略': r['strategies'],
                            '命中数': r['hit_count'],
                            '匹配模式': r['matched_patterns'],
                            '子术语': '、'.join(
                                c['word'] for c in r.get('children', [])
                            ),
                        })

                    df = pd.DataFrame(flat_rows)
                    df.index = df.index + 1

                    st.dataframe(
                        df.style
                          .background_gradient(subset=["综合评分"], cmap="Blues")
                          .background_gradient(subset=["出现频次"], cmap="Greens")
                          .background_gradient(subset=["命中数"], cmap="Oranges")
                          .format({"综合评分": "{:.4f}"}),
                        use_container_width=True,
                        height=600,
                    )

                    has_children = any(r.get('children') for r in results)
                    if has_children:
                        with st.expander("展开查看子术语详情"):
                            for r in results:
                                children = r.get('children', [])
                                if not children:
                                    continue
                                st.markdown(f"**{r['word']}** 的从属子术语：")
                                child_df = pd.DataFrame([
                                    {
                                        '子术语': c['word'],
                                        '出现频次': c['freq'],
                                        '综合评分': c['score'],
                                        '命中策略': c['strategies'],
                                    }
                                    for c in children
                                ])
                                child_df.index = child_df.index + 1
                                st.dataframe(child_df, use_container_width=True)

                    st.caption(
                        "**命中策略**：字符=字符包含、上下文=上下文模式引导、"
                        "共现=共现近邻、构词=构词结构相似　｜　"
                        "**命中数**：被几种策略同时发现（越多越可信）　｜　"
                        "**匹配模式**：命中的上下文模板　｜　"
                        "**子术语**：包含该主词的更长术语（从属关系）"
                    )
        else:
            st.info("请在上方输入关键词，开始提取。")

else:
    st.info("请先上传一个 TXT 文件，所有功能将在此文件上运行。")
