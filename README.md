# 事件知识图谱构建流水线

本项目当前实现的是“事件抽取规划 + 开放信息抽取 + Schema 规范化”的离线知识图谱流水线。

当前真实生效的阶段是：

- S0 新闻预处理
- S1 事件抽取规划
- S2 开放五元组抽取与初始概念池构建
- S3 描述生成
- S4 向量嵌入
- S5 聚类去重
- S6 Schema 汇总
- S7 三元组规范化

旧的以下阶段已经移除，不再属于当前实现：

- S1 共指消解
- S2A 句子分类
- S2B 句子简化

对应地，项目中与这些旧阶段相关的提示词、Python 模块、配置项和测试分支也已经清理掉。

## 1. 当前流程

当前推荐的数据流如下：

```text
data/sources 或 data/cleaned
  │
  ├─ S0 新闻预处理            → data/processed/*.txt
  ├─ run_timeline_correction  → data/processed_timeline/*.txt
  ├─ S1 事件抽取规划          → data/extract/*.json
  ├─ S2 开放五元组抽取        → data/tuple_extrct/*.json
  │                            + data/concept/*.json
  │                            + output/s2_raw_tuples.jsonl
  ├─ S3 描述生成              → output/s3_*.jsonl
  ├─ S4 向量嵌入              → output/s4_embeddings.npz
  ├─ S5 聚类去重              → output/s5_canonical.jsonl + output/s5_alias_map.json
  ├─ S6 Schema 汇总           → output/s6_schema.json
  └─ S7 三元组规范化          → output/s7_canonical_tuples.jsonl
```

其中最关键的变化是：

1. 不再对全文做共指消解、句型分类、句子简化。
2. 改为先对 `data/processed_timeline` 中的每篇文本逐句做事件抽取规划。
3. 只有规划结果里的“事件单元”非空句子会进入开放五元组抽取。
4. S2 同时构建初始实体、关系、地点概念池；S3 之后仍会基于数据集来源句重新描述并进一步规范化。

## 2. 输入与输出目录

### 输入目录

- `data/sources/`：原始新闻文本。
- `data/cleaned/`：可选，规则清洗后的文本。S0 会优先使用它。
- `data/processed/`：S0 输出的主事件链压缩文本。
- `data/processed_timeline/`：时间线校正脚本输出的派生文本，S1 直接消费这里。
- `data/extract/`：S1 输出的逐句事件抽取规划。
- `data/tuple_extrct/`：S2 输出的篇章级开放五元组抽取结果。
- `data/concept/`：S2 输出的数据集共享初始概念池。

### 关键输出目录

- `data/extract/`：每篇文档一个 JSON 文件，保存整篇逐句规划结果。
- `data/tuple_extrct/`：每篇文档一个 JSON 文件，只保存已请求抽取的句子结果。
- `data/concept/`：实体、关系、地点三个共享概念池文件。
- `output/`：阶段性结构化产物。
- `output/batch/`：batch 模式下的中间文件。
- `output/event_evolution/`：时间线校正与事件演化工具相关输出。

### 当前主要输出文件

- `output/s0_processed_manifest.jsonl`
- `output/s1_extract_plan_manifest.jsonl`
- `output/s1_sentence_index.jsonl`
- `output/s2_tuple_extract_manifest.jsonl`
- `output/s2_raw_tuples.jsonl`
- `data/concept/entity_pool.json`
- `data/concept/relation_pool.json`
- `data/concept/location_pool.json`
- `data/concept/entity_vectors.npz`
- `data/concept/relation_vectors.npz`
- `data/concept/location_vectors.npz`
- `output/s3_entity_desc.jsonl`
- `output/s3_relation_desc.jsonl`
- `output/s3_location_desc.jsonl`
- `output/s4_embeddings.npz`
- `output/s5_canonical.jsonl`
- `output/s5_alias_map.json`
- `output/s6_schema.json`
- `output/s7_canonical_tuples.jsonl`

## 3. 环境准备

建议使用 Python 3.11。

安装依赖：

```powershell
pip install -r requirements.txt
```

## 4. 配置文件

当前统一使用 [config.yaml](config.yaml) 作为主配置文件。

核心配置项示例如下：

```yaml
api_key: ""
base_url: "https://api.siliconflow.cn/v1"
llm_model: "Pro/deepseek-ai/DeepSeek-V3.2"
enable_thinking: "disabled"
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
dedup_bm25_weight: 0.5
dedup_max_iterations: 10
min_confidence: 0.6
context_token_budget: 6000
context_window_size: 5
execution_mode: "sync"
strict_dependency_check: true
strict_canonical_resolution: false
data_loader: "news_txt"
```

说明：

- 聊天、Embedding、Batch 统一通过 SiliconFlow OpenAI 兼容接口访问。
- 如果 `api_key` 留空，程序会兼容读取环境变量 `SILICONFLOW_API_KEY`。
- `enable_thinking` 支持 `enabled`、`disabled`、`omit` 三个值，分别表示显式传 `true`、显式传 `false`、或完全不传该字段。
- 旧配置项 `classify_batch_size` 已删除，因为旧的 S2A 已经移除。

## 5. 保留的提示词模块

当前项目中保留且正在使用的提示词模块只有：

- `prompts/news_process.py`
- `prompts/extract_plan.py`
- `prompts/tuple_extract.py`
- `prompts/tuple_normalize.py`
- `prompts/describe_entity.py`
- `prompts/describe_relation.py`
- `prompts/describe_location.py`
- `prompts/dedup_check.py`
- `prompts/canonical_select.py`

对应的 system 模板位于 `prompts/md/`。

已删除的无用旧提示词：

- `coref`
- `classify`
- `simplify_cx`
- `simplify_cd`
- `simplify_cc`

## 6. 运行方式

### 6.1 查看执行计划

```powershell
python main.py --run-all --dry-run
```

### 6.2 执行全部主阶段

```powershell
python main.py --run-all
```

说明：

- 这会按注册顺序执行 `s0 → s1 → s2 → s3 → s4 → s5 → s6 → s7`。
- 但 S1 依赖 `data/processed_timeline/`，所以在实际项目使用中，通常会先手动跑时间线校正脚本。

### 6.3 从某一阶段继续执行

```powershell
python main.py --from s2
python main.py --from s3
```

### 6.4 只执行指定阶段

```powershell
python main.py --only s1
python main.py --only s5
python main.py --only s6 s7
```

### 6.5 临时覆盖执行模式

```powershell
python main.py --run-all --mode batch
python main.py --from s3 --mode sync
```

说明：

- 当前 batch 模式主要用于聊天阶段。
- S4 在 batch 模式下会自动回退为同步嵌入，因为 SiliconFlow 官方 Batch 当前只支持聊天接口。

### 6.6 按文档范围处理

`--news-scope` 当前对以下阶段生效：

- `s0` 新闻预处理
- `s1` 事件抽取规划

`--file-scope` 当前对以下阶段生效：

- `s2` 开放五元组抽取与概念池构建

支持写法：

- `all`
- `M016`
- `M016 示例新闻＜2025.12.30＞`
- `M016 示例新闻＜2025.12.30＞.txt`
- `M016 示例新闻＜2025.12.30＞.json`
- `M016 示例新闻＜2025.12.30＞.txt,M020 示例新闻＜2025.12.30＞.txt`

支持格式：

- `all`
- `M016`
- `M016-M032`
- `M016,M020-M032`

示例：

```powershell
python main.py --only s0 --news-scope M016-M032
python main.py --only s1 --news-scope M016-M032
python main.py --only s2 --file-scope "M016 示例新闻＜2025.12.30＞.txt"
```

### 6.7 实际运行入口示例

如果你要处理 `M016-M032` 这一批文档，推荐直接按下面的顺序运行：

```powershell
# 1. 可选：先做规则清洗
python run_news_cleanup.py --scope M016-M032

# 2. 新闻预处理，生成 data/processed
python main.py --only s0 --news-scope M016-M032

# 3. 时间线校正，生成 data/processed_timeline
python run_timeline_correction.py --scope M016-M032

# 4. 事件抽取规划，生成 data/extract 和 S1 索引文件
python main.py --only s1 --news-scope M016-M032

# 5. 从开放信息抽取开始跑完整个图谱主线
python main.py --from s2
```

如果你已经有 `data/processed_timeline/`，最短入口就是：

```powershell
python main.py --only s1 --news-scope M016-M032
python main.py --from s2
```

如果你只想重跑某篇文档的规划结果：

```powershell
python main.py --only s1 --news-scope M016
```

### 6.8 S1 抽取规划入口示例

如果你只关心“事件抽取规划”这一步，直接使用下面几种入口即可。

1. [ ] 已经准备好 `data/processed_timeline/`，跑全部篇章：

```powershell
python main.py --only s1 --news-scope all
```

1. 只跑单篇文档，例如 `M016`：

```powershell
python main.py --only s1 --news-scope M016
```

1. 跑一个连续区间，例如 `M016-M032`：

```powershell
python main.py --only s1 --news-scope M016-M032
```

1. 跑离散 + 区间混合范围，例如 `M016,M020-M032`：

```powershell
python main.py --only s1 --news-scope M016,M020-M032
```

1. 如果当前只有原始新闻，完整前置到 S1 的最短命令序列是：

```powershell
python main.py --only s0 --news-scope M016-M032
python run_timeline_correction.py --scope M016-M032
python main.py --only s1 --news-scope M016-M032
```

执行完成后，S1 的直接产物会出现在：

- `data/extract/*.json`
- `output/s1_extract_plan_manifest.jsonl`
- `output/s1_sentence_index.jsonl`

## 7. 时间线校正脚本

根目录提供独立脚本 [run_timeline_correction.py](run_timeline_correction.py)，用于把 `data/processed/` 转换为 `data/processed_timeline/`。

示例：

```powershell
python run_timeline_correction.py
python run_timeline_correction.py --scope M016-M032
python run_timeline_correction.py --limit 10 --dry-run
```

其主要输出包括：

- `data/processed_timeline/*.txt`
- `output/event_evolution/timeline_manifest.jsonl`
- `output/event_evolution/timeline_review_manifest.jsonl`
- `output/event_evolution/timeline_summary.json`

## 8. 阶段依赖说明

当前阶段依赖关系是：

```text
s0: 无 output 依赖
s1: 无 output 依赖
s2: 依赖 s1_extract_plan_manifest.jsonl, s1_sentence_index.jsonl；产出 s2_tuple_extract_manifest.jsonl、s2_raw_tuples.jsonl、data/concept/*.json 和 data/concept/*_vectors.npz
s3: 依赖 data/concept/entity_pool.json, data/concept/relation_pool.json, data/concept/location_pool.json
s4: 依赖 s3_entity_desc.jsonl, s3_relation_desc.jsonl, s3_location_desc.jsonl
s5: 依赖 data/concept/*.json、s3_*_desc.jsonl 和 s4_embeddings.npz
s6: 依赖 s5_canonical.jsonl, s5_alias_map.json
s7: 依赖 s2_raw_tuples.jsonl, s5_canonical.jsonl, s5_alias_map.json
```

JSONL 文件末尾存在 `{"__done__": true}` 时，默认视为阶段完成，可用于断点续跑。

## 9. 测试

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

当前测试覆盖：

- 配置加载与环境变量回退。
- 提示词输出解析。
- JSONL/JSON 工具与阶段依赖检查。
- 时间线校正脚本。
- 使用假执行器的规划到抽取流水线闭环测试。
- S5 alias 合并测试。
- S7 fallback canonical 回写测试。

## 10. 常见问题

### 10.1 报错“未配置 API Key”

说明你执行了依赖 SiliconFlow 的阶段，但 [config.yaml](config.yaml) 里的 `api_key` 为空。

优先做法：

- 直接在 [config.yaml](config.yaml) 中填写 `api_key`

兼容做法：

- 设置环境变量 `SILICONFLOW_API_KEY`

### 10.2 为什么 S1 不直接读 data/processed

因为当前实现把事件抽取规划建立在“时间线校正后的篇章”之上，所以 S1 直接消费 `data/processed_timeline/`。

### 10.3 为什么 `--run-all` 之前还要先跑时间线校正

因为时间线校正目前仍是独立脚本，不在 `main.py` 注册阶段内。实际使用时，推荐先跑：

```powershell
python main.py --only s0 --news-scope M016-M032
python run_timeline_correction.py --scope M016-M032
python main.py --only s1 --news-scope M016-M032
python main.py --from s2
```

### 10.4 S7 为什么可能会修改 canonical 文件

这是 fallback 行为：当未命中 alias_map 且未开启严格模式时，S7 会自动补写兜底 canonical，并同步更新 alias_map。
