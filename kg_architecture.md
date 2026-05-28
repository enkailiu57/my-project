# 事件知识图谱构建系统架构设计文档

**v2.0 · 事件抽取规划版**

---

## 1. 系统概述

本系统是一个开放域事件知识图谱自动构建流水线，不依赖预定义 Schema，自动归纳并规范化关系、实体、地点类型，并为每个 Schema 元素生成自然语言定义。

当前实现采用两段式抽取策略：

1. 先做篇章级逐句事件抽取规划。
2. 再仅对规划出的“事件单元”执行开放信息抽取。

这样做的目标不是把句子强行改写成简单句，而是先用一个专门的规划器判断：

- 哪些句子值得抽取。
- 句内哪些原文片段才是真正应该进入五元组抽取的博弈事件。
- 哪些内容必须过滤，例如来源链、评论、宣传载体本身、后勤动作、结果描述、内宣内容等。

整个系统当前的主线对应如下三类问题：

- **事件规划**：在开放抽取之前，先从篇章上下文中识别可抽取的博弈事件片段。
- **EDC 风格关系规范化**：先为实体、关系、地点生成自然语言描述，再基于描述做规范化与 Schema 汇总。
- **KGGen 风格聚类去重**：通过向量聚类、近邻检索和 LLM 判重，把跨文档的同义表达归并为 canonical 表达。

与上一版设计相比，当前实现已取消以下阶段：

- S1 共指消解
- S2A 句子分类
- S2B 句子简化

当前主流程阶段编号为：

- S0 新闻预处理
- S1 事件抽取规划
- S2 开放五元组抽取与初始概念池构建
- S3 描述生成
- S4 向量嵌入
- S5 聚类去重
- S6 Schema 汇总
- S7 三元组规范化

---

## 2. API 规范

### 2.1 LLM API（SiliconFlow OpenAI 兼容接口）

当前工程统一通过 SiliconFlow 提供聊天、批处理与 Embedding 服务。

**聊天接口示意**

```text
POST https://api.siliconflow.cn/v1/chat/completions
Authorization: Bearer {API_KEY}
Content-Type: application/json
```

**关键参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `model` | str | 同步调用默认使用 `llm_model`，批处理调用使用 `batch_llm_model` |
| `messages` | list | 固定为 `[system, user]` 两条 |
| `temperature` | float | 结构化抽取一般使用 `0.1`，描述生成一般使用 `0.3` |
| `max_tokens` | int | 统一由配置项 `max_tokens` 控制 |
| `response_format` | dict | 统一设置为 `{"type": "json_object"}` |

### 2.2 Embedding API

**接口示意**

```text
POST https://api.siliconflow.cn/v1/embeddings
Authorization: Bearer {API_KEY}
Content-Type: application/json
```

**关键参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `model` | str | 当前默认 `Qwen/Qwen3-Embedding-4B` |
| `input` | str \| list[str] | 同步嵌入按批次调用 |
| `dimensions` | int | 当前默认 512 |

---

## 3. 流水线总体描述

当前流水线由 8 个顺序阶段组成，其中主图谱构建主线为 S1、S2、S3-S7；S0 是前置新闻压缩预处理。

```text
data/sources 或 data/cleaned
  │
  ├─ S0 新闻预处理           → data/processed/*.txt
  │
  ├─ 独立时间线校正脚本       → data/processed_timeline/*.txt
  │
  ├─ S1 事件抽取规划         → data/extract/*.json
  │                           + output/s1_extract_plan_manifest.jsonl
  │                           + output/s1_sentence_index.jsonl
  ├─ S2 开放五元组抽取       → data/tuple_extrct/*.json
  │                           + data/concept/*.json
  │                           + output/s2_tuple_extract_manifest.jsonl
  │                           + output/s2_raw_tuples.jsonl
  ├─ S3 描述生成             → output/s3_{entity,relation,location}_desc.jsonl
  ├─ S4 向量嵌入             → output/s4_embeddings.npz
  ├─ S5 聚类去重             → output/s5_canonical.jsonl + output/s5_alias_map.json
  ├─ S6 Schema 汇总          → output/s6_schema.json
  └─ S7 三元组规范化         → output/s7_canonical_tuples.jsonl
```

断点续跑规则：

- 所有 JSONL 阶段输出文件末尾追加 `{"__done__": true}` 哨兵行。
- 启动时若检测到阶段输出完整存在，则自动跳过该阶段。
- `data/extract/*.json` 是篇章级业务结果文件，不带 done 哨兵；S1 的完成性由 `output/s1_extract_plan_manifest.jsonl` 和 `output/s1_sentence_index.jsonl` 共同表示。

---

## 4. 各阶段详细说明

### S0：新闻预处理

**目标**：把原始新闻篇章压缩为主事件链文本，供后续时间线校正与事件抽取规划使用。

**输入**：

- `data/cleaned/*.txt`，若存在则优先使用。
- 否则回退到 `data/sources/*.txt`。

**处理逻辑**：

- 规则分句。
- 使用 `prompts/md/news_process.md` 对整篇新闻做主事件链压缩。
- 输出到 `data/processed/*.txt`。

**输出**：

- `data/processed/*.txt`
- `output/s0_processed_manifest.jsonl`

### 时间线校正（独立脚本，不计入主阶段编号）

**目标**：基于标题与正文推断篇章结束时间，生成带时间锚点的派生文本集。

**输入**：

- `data/processed/*.txt`

**输出**：

- `data/processed_timeline/*.txt`
- `output/event_evolution/timeline_manifest.jsonl`
- `output/event_evolution/timeline_review_manifest.jsonl`
- `output/event_evolution/timeline_summary.json`

### S1：事件抽取规划

**目标**：在开放抽取前，对 `data/processed_timeline` 中的每篇篇章逐句规划，判断哪些句子包含可抽取的政治军事博弈事件，以及句内哪些原文片段应进入五元组抽取。

**处理逻辑**：

1. 读取 `data/processed_timeline/*.txt`。
2. 对每篇文档进行规则分句，生成连续的句子编号。
3. 将该篇章的全部分句列表固定写入系统提示词 `prompts/md/extract_plan.md`。
4. 逐句构造用户提示词：
   - `句子编号`
   - `当前请求句`
5. 对每句返回一个结构化规划结果，包含：
   - 句子类型
   - 事件分析
   - 事件单元列表
6. 把整篇所有句子的规划结果汇总为一个 JSON 文件，写入 `data/extract/`。

**输出**：

- `data/extract/<篇章同名>.json`
- `output/s1_extract_plan_manifest.jsonl`
- `output/s1_sentence_index.jsonl`

其中：

- `s1_extract_plan_manifest.jsonl` 用于阶段完成态和篇章级统计。
- `s1_sentence_index.jsonl` 是后续 S2 的句子索引与上下文回填依据。

### S2：开放五元组抽取与初始概念池构建

**目标**：对 S1 规划中“事件单元”非空的句子逐句执行开放五元组抽取，并同步维护数据集共享的初始实体、关系、地点概念池。

**处理逻辑**：

1. 读取 `output/s1_sentence_index.jsonl` 和 `data/extract/*.json`。
2. 对每篇文档构造完整篇章上下文：标题 + 与 S1 相同的分句列表。
3. 只对 `事件单元` 非空的句子逐句调用 `prompts/md/tuple_extract.md`，第一轮不再拼接全量概念池。
4. 第一轮模型输出初始概念化主体、关系、客体、时间、地点、置信度，以及实体/关系/地点描述。
5. 程序对第一轮抽取出的每个实体、关系、地点描述生成嵌入，并在对应的 `data/concept/*_vectors.npz` 中检索 top-k 相似概念。
6. 程序调用 `prompts/md/tuple_normalize.md`，根据相似候选决定复用已有概念并扩充描述，或创建新的合格概念。
7. 程序按句增量写入 `data/tuple_extrct/<篇章同名>.json`，并实时更新 `data/concept/*.json` 与 `data/concept/*_vectors.npz`。
8. 为后续规范化流程，同步导出 `output/s2_raw_tuples.jsonl`；S3、S5 直接读取 `data/concept/*.json`，不再依赖元素池交换格式 JSONL。

**输出**：

- `data/tuple_extrct/<篇章同名>.json`
- `data/concept/entity_pool.json`
- `data/concept/relation_pool.json`
- `data/concept/location_pool.json`
- `data/concept/entity_vectors.npz`
- `data/concept/relation_vectors.npz`
- `data/concept/location_vectors.npz`
- `data/concept/entity_vector_meta.json`
- `data/concept/relation_vector_meta.json`
- `data/concept/location_vector_meta.json`
- `output/s2_tuple_extract_manifest.jsonl`
- `output/s2_raw_tuples.jsonl`

原独立 S4 元素池构建阶段已经并入 S2，不再作为可执行主阶段单列。

### S3：描述生成

**目标**：为实体、关系、地点生成自然语言描述，作为后续规范化和聚类的语义锚点。

**输出**：

- `output/s3_entity_desc.jsonl`
- `output/s3_relation_desc.jsonl`
- `output/s3_location_desc.jsonl`

### S4：向量嵌入

**目标**：把 S3 的描述嵌入成 512 维向量，供 S5 聚类去重。

**输出**：

- `output/s4_embeddings.npz`

### S5：聚类去重

**目标**：对实体、关系、地点分别做聚类与 canonical 归并。

**输出**：

- `output/s5_canonical.jsonl`
- `output/s5_alias_map.json`

### S6：Schema 汇总

**目标**：汇总 canonical 元素，输出最终 Schema。

**输出**：

- `output/s6_schema.json`

### S7：三元组规范化

**目标**：将原始五元组中的 subject、relation、object、location 替换为 canonical_id。

**输出**：

- `output/s7_canonical_tuples.jsonl`

---

## 5. 核心数据结构

```python
# S1 每句规划结果（写入 data/extract/<doc>.json）
ExtractPlanSentenceResult = {
    "句子编号": str,
    "原句": str,
    "句子类型": str,           # 事件句 | 混合句 | 不抽取句
    "事件分析": str,
    "事件单元": [
        {
            "编号": str,
            "文本片段": str,
            "单元类型": str,   # 事实事件 | 重要表态事件 | 计划事件
            "分析方法建议": str,
            "抽取动作建议": str,
        }
    ],
}

# S1 句子索引（output/s1_sentence_index.jsonl）
SentenceIndexRow = {
    "sent_id": str,          # 如 M016_s001
    "doc_id": str,
    "sentence_index": int,
    "text": str,
    "source_file": str,
    "extract_file": str,
}

# S2 输出五元组（output/s2_raw_tuples.jsonl）
RawTuple = {
  "tuple_id": str,         # 如 M016_s001_t001
    "doc_id": str,
  "sent_id": str,          # 兼容字段，当前等于 tuple_id
    "subject": str,
    "relation": str,
    "object": str,
    "location": str | None,
  "time": str,
    "source_sent_id": str,   # 原始句 ID，如 M016_s001
    "confidence": float,
}
```

S2 的业务级结果另存于 `data/tuple_extrct/`，其中会保留每句抽取结果、概念化字段及来源句。S3-S7 的兼容 JSONL 数据结构与上一版保持一致，不再重复展开。

---

## 6. 文件存储结构

```text
project/
├── config.yaml
├── main.py
├── run_news_cleanup.py
├── run_timeline_correction.py
├── prompts/
│   ├── news_process.py
│   ├── extract_plan.py
│   ├── tuple_extract.py
│   ├── tuple_normalize.py
│   ├── describe_entity.py
│   ├── describe_relation.py
│   ├── describe_location.py
│   ├── dedup_check.py
│   ├── canonical_select.py
│   └── md/
│       ├── news_process.md
│       ├── extract_plan.md
│       ├── tuple_extract.md
│       ├── tuple_normalize.md
│       ├── describe_entity.md
│       ├── describe_relation.md
│       ├── describe_location.md
│       ├── dedup_check.md
│       └── canonical_select.md
├── pipeline/
│   ├── s0_news_process.py
│   ├── s1_extract_plan.py
│   ├── s2_open_ie.py
│   ├── s3_describe.py
│   ├── s4_embed.py
│   ├── s5_dedup.py
│   ├── s6_schema.py
│   └── s7_canonicalize.py
├── data/
│   ├── sources/
│   ├── cleaned/
│   ├── processed/
│   ├── processed_timeline/
│   ├── extract/
│   ├── tuple_extrct/
│   └── concept/
└── output/
    ├── s0_processed_manifest.jsonl
    ├── s1_extract_plan_manifest.jsonl
    ├── s1_sentence_index.jsonl
    ├── s2_tuple_extract_manifest.jsonl
    ├── s2_raw_tuples.jsonl
    ├── s3_entity_desc.jsonl
    ├── s3_relation_desc.jsonl
    ├── s3_location_desc.jsonl
    ├── s4_embeddings.npz
    ├── s5_canonical.jsonl
    ├── s5_alias_map.json
    ├── s6_schema.json
    └── s7_canonical_tuples.jsonl
```

---

## 7. 提示词文件规范

当前仍采用统一约定：

```python
def build_prompt(context: dict) -> list[dict]:
    ...

def parse_response(text: str) -> dict:
    ...
```

当前保留的提示词模块如下：

| 文件 | 职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| `news_process.md` | 新闻预处理，提炼主事件链 | `title`, `sentences` | `timeline` |
| `extract_plan.md` | 逐句事件抽取规划 | `document_sentences`, `sentence_id`, `sentence_text` | 每句规划结果 |
| `tuple_extract.md` | 第一轮开放五元组初抽取与概念描述生成 | `document_context`, `extraction_plan` | `tuples` |
| `tuple_normalize.md` | 第二轮相似概念替换、新建与描述修正 | `document_context`, `extraction_plan`, `raw_tuples`, `similar_candidates` | `tuples` |
| `describe_entity.md` | 实体描述生成 | `raw_str`, `contexts` | `description` |
| `describe_relation.md` | 关系描述生成 | `raw_str`, `contexts` | `description` |
| `describe_location.md` | 地点描述生成 | `raw_str`, `contexts` | `description` |
| `dedup_check.md` | 近邻概念判重 | `target`, `candidates` | `duplicates` |
| `canonical_select.md` | 选择 canonical 表达 | `candidates` | `canonical_str` |

已删除的旧提示词：

- `coref`
- `classify`
- `simplify_cx`
- `simplify_cd`
- `simplify_cc`

---

## 8. Batch API 适用性

| 阶段 | 是否适用 Batch | 说明 |
| --- | --- | --- |
| S0 新闻预处理 | ✅ | 每篇文档独立 |
| S1 事件抽取规划 | ✅ | 每句规划独立，同一篇内部只共享固定 system 内容 |
| S2 开放五元组抽取与概念池构建 | 不建议 batch | 概念池需随抽取顺序持续更新，sync 更符合当前设计 |
| S3 描述生成 | ✅ | 每个元素独立 |
| S4 向量嵌入 | 理论可批，当前实现回退 sync | SiliconFlow 官方 Batch 当前仅支持聊天接口 |
| S5 去重内层循环 | ❌ | 每轮依赖上一轮结果 |

---

## 9. alias_map

结构与上一版设计一致：

```json
{
  "entity": {
    "解放军": "ent_a1b2c3d4"
  },
  "relation": {
    "宣布部署": "rel_i9j0k1l2"
  },
  "location": {
    "台湾海峡": "loc_m3n4o5p6"
  }
}
```

---

## 10. 配置文件（config.yaml）

当前核心配置项如下：

```yaml
api_key: ""
base_url: "https://api.siliconflow.cn/v1"
llm_model: "Pro/deepseek-ai/DeepSeek-V3.2"
batch_llm_model: "deepseek-ai/DeepSeek-V3"
embed_model: "Qwen/Qwen3-Embedding-4B"
embed_dimensions: 512
temperature_extract: 0.1
temperature_describe: 0.3
max_tokens: 20000
embed_batch_size: 64
tuple_concept_top_k: 16
dedup_cluster_size: 128
dedup_top_k: 16
dedup_max_iterations: 10
min_confidence: 0.6
context_token_budget: 6000
context_window_size: 5
execution_mode: "sync"
strict_dependency_check: true
strict_canonical_resolution: false
data_loader: "news_txt"
```

注意：`classify_batch_size` 已随旧 S2A 删除，不再属于当前配置项。

---

## 11. core 模块说明

### `core/llm_client.py`

封装同步聊天调用，统一处理：

- OpenAI 兼容客户端创建
- `response_format={"type": "json_object"}`
- 指数退避重试
- 将结果交给 prompt 模块的 `parse_response()` 解析

### `core/task_executor.py`

统一调度 sync / batch 两条执行通道：

- `run_prompt_tasks()` 用于 S0、S1、S2、S3、S5
- `run_embedding_tasks()` 用于 S4

### `core/embed_client.py`

封装同步 Embedding 调用；当前 batch 模式下，S4 会自动退回同步嵌入。

---

## 12. 流水线入口（main.py）

### 12.1 命令行接口

```bash
python main.py --run-all
python main.py --from s2
python main.py --only s1
python main.py --only s3 s4 s5
```

当前支持的阶段名：

| 参数名 | 阶段 |
| --- | --- |
| `s0` | 新闻预处理 |
| `s1` | 事件抽取规划 |
| `s2` | 开放五元组抽取与概念池构建 |
| `s3` | 描述生成 |
| `s4` | 向量嵌入 |
| `s5` | 聚类去重 |
| `s6` | Schema 汇总 |
| `s7` | 三元组规范化 |

### 12.2 前置依赖定义

当前依赖关系如下：

```python
STAGE_DEPS = {
    "s0": [],
    "s1": [],
    "s2": ["s1_extract_plan_manifest.jsonl", "s1_sentence_index.jsonl"],
    "s3": [
        "data/concept/entity_pool.json",
        "data/concept/relation_pool.json",
        "data/concept/location_pool.json",
    ],
    "s4": ["s3_entity_desc.jsonl", "s3_relation_desc.jsonl", "s3_location_desc.jsonl"],
    "s5": [
        "data/concept/entity_pool.json",
        "data/concept/relation_pool.json",
        "data/concept/location_pool.json",
        "s3_entity_desc.jsonl",
        "s3_relation_desc.jsonl",
        "s3_location_desc.jsonl",
        "s4_embeddings.npz",
    ],
    "s6": ["s5_canonical.jsonl", "s5_alias_map.json"],
    "s7": ["s2_raw_tuples.jsonl", "s5_canonical.jsonl", "s5_alias_map.json"],
}
```

### 12.3 范围处理

`--news-scope` 当前对以下阶段生效：

- S0 新闻预处理
- S1 事件抽取规划

`--file-scope` 当前对以下阶段生效：

- S2 开放五元组抽取与概念池构建

支持写法：

- `all`
- `M016`
- `M016 示例新闻＜2025.12.30＞`
- `M016 示例新闻＜2025.12.30＞.txt`
- `M016 示例新闻＜2025.12.30＞.json`
- `M016 示例新闻＜2025.12.30＞.txt,M020 示例新闻＜2025.12.30＞.txt`

支持写法：

- `all`
- `M016`
- `M016-M032`
- `M016,M020-M032`

### 12.4 实际运行入口示例

以 `M016-M032` 为例，推荐执行顺序如下：

```bash
# 1. 可选：先做规则清洗
python run_news_cleanup.py --scope M016-M032

# 2. 新闻预处理，生成 data/processed
python main.py --only s0 --news-scope M016-M032

# 3. 时间线校正，生成 data/processed_timeline
python run_timeline_correction.py --scope M016-M032

# 4. 事件抽取规划，生成 data/extract
python main.py --only s1 --news-scope M016-M032

# 5. 从开放信息抽取开始继续跑到末尾
python main.py --from s2

# 6. 只重跑某一个篇章的 S2 抽取
python main.py --only s2 --file-scope "M016 示例新闻＜2025.12.30＞.txt"
```

如果你只想重跑某一批文档的规划结果，可以直接执行：

```bash
python main.py --only s1 --news-scope M016-M032
```

如果你已经准备好了 `data/processed_timeline`，最常用的入口就是：

```bash
python main.py --only s1 --news-scope M016-M032
python main.py --from s2
```

### 12.5 S1 事件抽取规划入口示例

如果只执行 S1，可按输入准备情况选择下面的入口。

**场景一：`data/processed_timeline/` 已准备完毕，处理全部篇章**

```bash
python main.py --only s1 --news-scope all
```

**场景二：只处理单篇，例如 `M016`**

```bash
python main.py --only s1 --news-scope M016
```

**场景三：处理连续编号区间，例如 `M016-M032`**

```bash
python main.py --only s1 --news-scope M016-M032
```

**场景四：处理离散 + 区间混合范围，例如 `M016,M020-M032`**

```bash
python main.py --only s1 --news-scope M016,M020-M032
```

**场景五：从原始新闻一路跑到 S1**

```bash
python main.py --only s0 --news-scope M016-M032
python run_timeline_correction.py --scope M016-M032
python main.py --only s1 --news-scope M016-M032
```

S1 执行完成后，应看到以下产物：

- `data/extract/<篇章同名>.json`
- `output/s1_extract_plan_manifest.jsonl`
- `output/s1_sentence_index.jsonl`
