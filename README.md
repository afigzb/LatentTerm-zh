# LatentTerm-zh

针对中文网络小说的无监督术语挖掘工具。

输入一本小说和一个种子词，输出原文里和它"潜在相关"的一批词。

举个例子，喂一本《斗罗大陆》、关键词填「魂兽」，会得到「柔骨兔」「人面魔蛛」「泰坦巨猿」之类——它们和「魂兽」**字面上没有重合**，但在原文语境里属于同一类东西。

## 它能做什么

- **术语聚合**：给一个种子词，找出原文里和它同类/相关的词。
- **双关键词联合**：填两个关键词（比如「魂环」+「武魂」），只输出在原文中同时和两者强相关的词，用来缩小话题。
- **文本预处理**：自动识别 GB18030 / UTF-8 等常见编码，可选清掉零宽字符并做 NFKC 标准化。
- **结果筛选**：基于 jieba 词典做词性分组，可以一键过滤动词、副词、虚词这些一般不需要的成分。

整个流程都是本地跑，不调任何在线接口。

## 怎么用

```powershell
# 首次运行（PowerShell 需要先放行脚本）
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

streamlit run app.py
```

浏览器会自动打开页面，上传 TXT 文件后输入关键词即可。

## 算法是怎么做的

整个流水线分三步：

**第一步：从全文里"取"出一份词表。**
直接拿 n-gram 计频会被高频长词污染（比如统计「张无忌」会把「张无」「无忌」也带高），所以走两步法：先用 PMI（凝固度）和左右邻字熵（自由度）筛出"结实"的 n-gram，再用 Trie 长词优先重新切一遍原文，得到干净频数。同时跑一遍正则模板（对白动词、量词、命名套式等），把那些只出现两三次但模式特征极强的孤岛专名也捞进候选池。

**第二步：从六个角度找候选词。**

| 策略 | 看什么 |
|---|---|
| 字符包含 | 候选词是否包含关键词的字，且字的相对位置一致 |
| 上下文模式 | 候选词是否常出现在和关键词相同的左右文环境里 |
| 共现近邻 | 候选词在关键词附近 50 字内是否高频出现（用 Lift 排除"的""了"这类常见词） |
| 构词结构 | 候选词是否和关键词共享相同的字/双字片段，且位置相同（同为前缀、同为后缀） |
| 互替性 | 候选词能不能填进关键词所在的"左 2 字 + ___ + 右 2 字"框架里 |
| 段落共主题 | 候选词和关键词是否常出现在同一个段落里（500 字粒度） |

**第三步：融合排序。**
六路分数各自归一化后按权重相加，命中策略越多分越高。还会做几件后处理的事：根据关键词的隐式类型（人物 / 地点 / 生物 / 招式·物 / 组织 / 其他）做类型匹配加分；按"残片 / 临时短语 / 独立术语"三种关系把短词挂到长词下面（比如「雨浩」会被合并到「霍雨浩」之下）。

## 项目结构

```
app.py                       # Streamlit 前端
core/
  text_cleaner.py            # 编码识别 + 文本清洗
  term_extractor.py          # 三阶段流水线主控 + 融合排序
  _vocab_builder.py          # 第一步：词表构建
  _pattern_miner.py          # 正则模板狙击 + 关键词自适应模板
  _strategies.py             # 第二步：六大策略
  dict_filter.py             # 基于 jieba 词典的词性过滤
  _utils.py                  # 噪声字表、熵、归一化等工具
```

## 依赖

`streamlit`、`pandas`、`jieba`、`pyahocorasick`、`charset-normalizer`、`matplotlib`，全部见 `requirements.txt`。Python 3.10+。

## 打包成便携版

项目内置了一键打包脚本，产出"解压即用"的 Windows 便携版，目标用户双击 `启动.bat` 就能在浏览器里使用，无需安装 Python 或任何依赖。

```powershell
# 在仓库根目录执行
powershell -ExecutionPolicy Bypass -File scripts\build_portable.ps1

# 同时压缩为 zip（方便上传 Release / 网盘）
powershell -ExecutionPolicy Bypass -File scripts\build_portable.ps1 -Zip
```

脚本做的事：

1. 从 python.org 下载 `python-3.11.9-embed-amd64.zip`（首次约 10 MB，之后走缓存）
2. 用 `get-pip.py` 引导 pip，再把 `requirements.txt` 装到独立的 `libs\` 目录
3. 拷贝 `core/`、`app.py`、`.streamlit/`、`README.md`、`LICENSE` 到产物目录
4. 生成 `启动.bat` 和 `使用说明.txt`

产物结构（`dist\LatentTerm\`，约 300–500 MB）：

```text
LatentTerm\
├─ python\          嵌入版 Python 3.11.9
├─ libs\            streamlit / jieba / pandas / ...
├─ core\            核心算法
├─ app.py
├─ .streamlit\
├─ 启动.bat          用户双击入口
└─ 使用说明.txt
```

最终用户无需装任何东西，下载 zip → 解压 → 双击「启动.bat」→ 浏览器自动弹出页面即可使用。

### 常见问题

- **Windows Defender / SmartScreen 拦截**：因为 `.bat` + 独立 Python 被误判。点「更多信息 → 仍要运行」即可。程序完全离线、不联网。
- **端口 8965 被占用**：关掉占用的程序，或修改 `.streamlit\config.toml` 里的 `port`。
- **国内网络下载 pip 包慢**：编辑 `scripts\build_portable.ps1`，在 `pip install` 那行加上 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。


## 许可

MIT，见 `LICENSE`。
