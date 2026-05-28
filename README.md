# 事件知识图谱构建项目说明

这是一个面向新闻文本的离线事件知识图谱构建项目。主线目标是从新闻文档中抽取事件相关五元组，构建实体、关系、地点概念池，再通过概念描述、聚类、归并和映射回写，把开放抽取得到的概念逐步规范化。

当前最新、最活跃的工作重点是 `concept_abstraction` 子流程：对 `entity`、`relation`、`location` 三类概念池做合规过滤、结构化抽象描述、字段加权 embedding、Leiden 聚类、簇内 LLM 归并、人工审核映射，并把审核后的概念映射应用回五元组。

本 README 是给后续维护者或 Cursor 接手项目时使用的最新入口说明。若其他旧文档与本文冲突，以本文和代码为准。

## 1. 项目结构

```text
.
├─ main.py                       # 主流水线入口：S0-S7
├─ config.yaml                   # SiliconFlow / 模型 / 重试 / 执行模式配置
├─ requirements.txt              # Python 依赖
├─ pipeline/                     # main.py 注册的主流水线阶段
├─ core/                         # 配置、LLM client、embedding client、task executor
├─ prompts/                      # prompt 模块与解析器
│  └─ md/                        # system prompt 模板
├─ tools/concept_abstraction/     # 最新概念抽象、聚类、归并、可视化逻辑
├─ utils/                        # IO、文本、阶段依赖等工具
├─ data/                         # 输入、中间数据、五元组和概念池
├─ output/                       # 主流水线与概念抽象输出
├─ tests/                        # 单元测试；临时测试脚本也放这里
├─ run_concept_*.py              # 概念抽象子流程入口
├─ run_tuple_*.py                # 五元组清理、排序、导出等辅助入口
├─ run_news_cleanup.py           # 新闻规则清洗
├─ run_timeline_correction.py    # 时间线校正
└─ run_document_graph.py         # 文档图谱相关辅助脚本
```

## 2. 环境准备

推荐 Python 3.11。当前开发环境使用 conda 的 `py311` 环境。

```powershell
pip install -r requirements.txt
```

主要依赖：

- `openai`：访问 SiliconFlow OpenAI 兼容接口。
- `numpy`、`scikit-learn`：向量处理、相似度、聚类和降维。
- `igraph`、`leidenalg`：Leiden balanced 聚类。
- `matplotlib`、`umap-learn`：聚类可视化。
- `PyYAML`：读取配置。

如果 `igraph` 或 `leidenalg` 安装后仍无法 import，可以在当前解释器下直接执行：

```powershell
python -m pip install python-igraph leidenalg
```

## 3. 配置

统一配置文件是 `config.yaml`。聊天、embedding、batch 都通过 SiliconFlow OpenAI 兼容接口访问。

关键配置项：

```yaml
api_key: ""
base_url: "https://api.siliconflow.cn/v1"
llm_model: "Qwen/Qwen3.6-27B"
enable_thinking: "disabled"
batch_llm_model: "deepseek-ai/DeepSeek-V3"
embed_model: "Qwen/Qwen3-Embedding-8B"
embed_dimensions: 512
request_timeout_seconds: 300
api_retry_count: 6
api_retry_base_delay_seconds: 2.0
api_min_interval_seconds: 2.0
temperature_extract: 0.1
temperature_describe: 0.3
max_tokens: 2000
execution_mode: "sync"
```

API key 可以写在 `config.yaml` 的 `api_key`，也可以通过环境变量 `SILICONFLOW_API_KEY` 提供。`api_retry_count`、`api_retry_base_delay_seconds` 和 `api_min_interval_seconds` 控制 429 / 5xx / 临时网络错误的重试与节流；描述生成和归并调用 LLM 较多时，优先调大 `api_min_interval_seconds`。

## 4. 主流水线 S0-S7

主流水线入口是 `main.py`，阶段注册在 `pipeline/registry.py`。

```text
S0 新闻预处理
  输入：data/sources 或 data/cleaned
  输出：data/processed/*.txt, output/s0_processed_manifest.jsonl

时间线校正，独立脚本，不在 main.py 的 S0-S7 注册里
  输入：data/processed/*.txt
  输出：data/processed_timeline/*.txt, output/event_evolution/*

S1 事件抽取规划
  输入：data/processed_timeline/*.txt
  输出：data/extract/*.json, output/s1_extract_plan_manifest.jsonl, output/s1_sentence_index.jsonl

S2 开放五元组抽取与概念池构建
  输入：data/extract/*.json
  输出：data/tuple_extrct/*.json, data/concept/*.json, output/s2_raw_tuples.jsonl

S3 描述生成
  输入：data/concept/{entity,relation,location}_pool.json
  输出：output/s3_entity_desc.jsonl, output/s3_relation_desc.jsonl, output/s3_location_desc.jsonl

S4 向量嵌入
  输入：output/s3_*_desc.jsonl
  输出：output/s4_embeddings.npz

S5 聚类去重
  输入：data/concept/*.json, output/s3_*_desc.jsonl, output/s4_embeddings.npz
  输出：output/s5_canonical.jsonl, output/s5_alias_map.json

S6 Schema 汇总
  输入：output/s5_canonical.jsonl, output/s5_alias_map.json
  输出：output/s6_schema.json

S7 三元组规范化
  输入：output/s2_raw_tuples.jsonl, output/s5_canonical.jsonl, output/s5_alias_map.json
  输出：output/s7_canonical_tuples.jsonl
```

常用命令：

```powershell
# 只展示执行计划和依赖检查
python main.py --run-all --dry-run

# 跑全部已注册阶段；注意时间线校正仍需单独跑
python main.py --run-all

# 从某一阶段继续
python main.py --from s2
python main.py --from s3

# 只跑指定阶段
python main.py --only s1
python main.py --only s6 s7

# 临时覆盖执行模式
python main.py --run-all --mode sync
python main.py --from s3 --mode batch
```

实际从原始新闻推进到 S2 的常用顺序：

```powershell
python run_news_cleanup.py --scope M016-M032
python main.py --only s0 --news-scope M016-M032
python run_timeline_correction.py --scope M016-M032
python main.py --only s1 --news-scope M016-M032
python main.py --from s2
```

范围参数：

- `--news-scope` 对 S0、S1 生效，支持 `all`、`M016`、`M016-M032`、`M016,M020-M032`。
- `--file-scope` 对 S2 生效，支持按 `source_file`、`extract_file`、文件 stem 或 `doc_id` 过滤。

## 5. 最新概念抽象子流程

概念抽象子流程不通过 `main.py` 跑，而是使用根目录下的 `run_concept_*.py` 脚本。它消费 S2 之后产生的 `data/concept` 和后续整理出的 `data/tuple_pocessed&sorted`，输出到 `output/concept_abstraction`，最后可把审核后的映射应用回五元组。

```text
data/concept/*.json
data/tuple_pocessed&sorted/*.json
  │
  ├─ run_concept_compliance_filter.py
  │    → output/concept_abstraction/compliant_pools
  │    → output/concept_abstraction/invalid
  │
  ├─ run_concept_description_generate.py
  │    → output/concept_abstraction/descriptions
  │
  ├─ run_concept_cluster.py
  │    → output/concept_abstraction/embeddings
  │    → output/concept_abstraction/clusters/{entity,relation,location}
  │    → output/concept_abstraction/cluster_summary.json
  │
  ├─ run_concept_cluster_visualize.py
  │    → output/concept_abstraction/visualizations/*.png
  │
  ├─ run_concept_merge_mapping.py
  │    → output/concept_abstraction/mappings/*_mapping__from_<cluster-file>.json
  │    → output/concept_abstraction/mappings/*_merge_decisions__from_<cluster-file>.jsonl
  │
  └─ 人工审核 mapping 后 run_concept_apply_mapping.py
       → data/tuple_pocessed&sorted&abstracted/*.json
```

### 5.1 合规性过滤

```powershell
python run_concept_compliance_filter.py --mode sync --chunk-size 20
```

默认输入：`data/concept/{entity,relation,location}_pool.json` 和 `data/tuple_pocessed&sorted/*.json`。

默认输出：

- `output/concept_abstraction/compliance/concept_compliance_results.jsonl`
- `output/concept_abstraction/compliance/summary.json`
- `output/concept_abstraction/compliant_pools/{entity,relation,location}_pool.json`
- `output/concept_abstraction/invalid/invalid_concepts.json`
- `output/concept_abstraction/invalid/invalid_tuples.jsonl`

默认支持断点续跑：已有 `concept_compliance_results.jsonl` 记录会被跳过。需要重跑时加 `--overwrite`。

只跑某一类：

```powershell
python run_concept_compliance_filter.py --element-types relation --mode sync --chunk-size 10
```

### 5.2 生成结构化抽象描述

```powershell
python run_concept_description_generate.py --mode sync --chunk-size 20
```

默认读取 `output/concept_abstraction/compliant_pools`。输出：

- `output/concept_abstraction/descriptions/concept_description_results.jsonl`
- `output/concept_abstraction/descriptions/{entity,relation,location}_descriptions.json`
- `output/concept_abstraction/descriptions/summary.json`

描述 schema 当前版本是 `structured_fields_v2`。LLM 只输出 `fields`，不再输出单段 `description`。代码会根据字段合成展示用 description，但 embedding 直接基于字段生成。

字段与权重定义在 `prompts/concept_description.py`：

```text
entity:
  主体类别 = 0.45
  核心角色 = 0.55

relation:
  关系类别 = 0.40
  核心议题 = 0.40
  战略作用 = 0.20

location:
  空间类型 = 0.35
  区位范围 = 0.40
  议题作用 = 0.25
```

三类描述 prompt：

- `prompts/md/concept_description_entity.md`
- `prompts/md/concept_description_relation.md`
- `prompts/md/concept_description_location.md`

关系字段里的 `关系类别` 应该是真实事件、行为或机制类别，例如 `军事冲突事件`、`国际组织参与事件`、`表态信号`，不要写 `结果状态` 这类元标签。

描述生成也是断点续跑式：结果一条写入一条，默认跳过当前 schema 下已有 `custom_id`。需要清空当前描述缓存后重跑时加 `--overwrite`。

### 5.3 字段加权 embedding 与聚类

默认聚类方法已经是 `leiden_balanced`，即 mutual-kNN 图 + Leiden RBConfiguration partition + 递归切分大簇 + 可选小簇吸收。它不需要预设聚类数量，适合把整个空间划成较紧密的局部簇。

默认命令：

```powershell
python run_concept_cluster.py
```

常用 Leiden 参数：

- `--target-cluster-size`：期望簇大小，默认 `24`。
- `--min-cluster-size`：希望大多数簇不低于的大小，默认 `20`。
- `--max-cluster-size`：递归切分时尽量压到的最大簇大小，默认 `30`。
- `--knn-k`：构图时每个点的近邻数，默认 `20`。越大图越连通，越容易减少 singleton，但也可能混入弱相关点。
- `--initial-resolution`：Leiden 初始 resolution，默认 `0.2`。越大越容易切碎。
- `--resolution-multiplier`：resolution 搜索放大倍数，默认 `1.5`。
- `--resolution-rounds`：每层尝试的 resolution 数，默认 `8`。
- `--max-split-depth`：递归切分大簇的最大深度，默认 `4`。
- `--min-edge-similarity`：保留图边的最小 cosine 相似度，默认 `0.0`。调高会让簇更纯，但可能增加小簇和 singleton。
- `--small-cluster-absorb-threshold`：把过小簇并回邻近簇所需的平均相似度，默认 `0.0` 表示关闭。调高可以减少 singleton，但过低会误吸收。
- `--overwrite-embeddings`：强制重算字段 embedding。

偏紧、偏纯的关系聚类示例：

```powershell
python run_concept_cluster.py --element-types relation --target-cluster-size 10 --min-cluster-size 5 --max-cluster-size 50 --knn-k 28 --initial-resolution 0.15 --min-edge-similarity 0.03 --small-cluster-absorb-threshold 0.22
```

较保守、减少 singleton 的关系聚类示例：

```powershell
python run_concept_cluster.py --element-types relation --target-cluster-size 40 --min-cluster-size 30 --max-cluster-size 50 --knn-k 45 --initial-resolution 0.15 --min-edge-similarity 0.03 --small-cluster-absorb-threshold 0.22
```

输出：

- `output/concept_abstraction/embeddings/*_description_vectors.npz`
- `output/concept_abstraction/embeddings/*_description_field_vectors.npz`
- `output/concept_abstraction/embeddings/*_description_vector_meta.json`
- `output/concept_abstraction/reports/*_similarity_distribution.json`
- `output/concept_abstraction/clusters/{entity,relation,location}/*.json`
- `output/concept_abstraction/cluster_summary.json`

Leiden 聚类文件名会写入关键参数和聚类结果，例如：

```text
relation_leiden_balanced_t10_mn5_mx50_k28_ir0p15_me0p03_sa0p22_fw0p4-0p4-0p2_sr1p581203_c105_mm105_h32419a5b.json
```

文件名 token 含义：

- `t`：target cluster size。
- `mn`：min cluster size。
- `mx`：max cluster size。
- `k`：knn-k。
- `ir`：initial resolution。
- `me`：min edge similarity。
- `sa`：small cluster absorb threshold。
- `fw`：field weights。
- `sr`：selected top-level resolution。
- `c`：最终簇数量。
- `mm`：多成员簇数量。
- `h`：参数与结果摘要 hash。

如果需要旧的固定 K 聚类，可显式使用 agglomerative：

```powershell
python run_concept_cluster.py --method agglomerative --entity-cluster-counts 120,150,180 --relation-cluster-counts 260,320,380 --location-cluster-counts 45,60,75
```

### 5.4 聚类可视化

可视化脚本读取聚类文件中的 `parameters.field_weights`，把每个概念的字段 embedding 加权聚合为一个向量，再用 UMAP 或 t-SNE 降到 2D。每个点代表一个概念，不显示概念文本标签；点旁边的小数字是 cluster 序号。

```powershell
python run_concept_cluster_visualize.py --cluster-file output/concept_abstraction/clusters/relation/<cluster-file>.json --method umap
```

常用参数：`--method umap|tsne`、`--umap-n-neighbors`、`--umap-min-dist`、`--tsne-perplexity`、`--point-size`、`--alpha`、`--cluster-number-fontsize`、`--figsize`、`--dpi`。

输出文件在 `output/concept_abstraction/visualizations`，文件名为 `<cluster-file-stem>_<method>.png`。

### 5.5 簇内 LLM 归并映射

当前归并方法是整簇单次调用：每个 cluster 中的全部概念直接交给 LLM，由 LLM 输出 `merge_groups`，也就是一个或多个可归并子集。没有进入任何 merge group 的概念会自动作为 singleton 保留。

三类归并 prompt：

- `prompts/md/concept_merge_entity.md`
- `prompts/md/concept_merge_relation.md`
- `prompts/md/concept_merge_location.md`

运行示例：

```powershell
python run_concept_merge_mapping.py --cluster-file output/concept_abstraction/clusters/relation/relation_leiden_balanced_t10_mn5_mx50_k28_ir0p15_me0p03_sa0p22_fw0p4-0p4-0p2_sr1p581203_c105_mm105_h32419a5b.json --mode sync --source-sentence-limit 2 --overwrite
```

常用参数：

- `--cluster-file`：必填，指定某个聚类 JSON 文件。
- `--concept-dir`：默认 `output/concept_abstraction/compliant_pools`，用于给 prompt 补充来源句。
- `--source-sentence-limit`：每个概念注入 prompt 的来源句数量，默认 `2`。
- `--temperature`：默认 `0.0`。
- `--max-tokens`：默认 `4096`；客户端对截断输出有扩容重试逻辑。
- `--mode sync|batch`：默认读取 `config.yaml`。
- `--overwrite`：清空当前 cluster 对应的 decision 缓存后重跑。

这些旧参数仍保留在 CLI 中，但当前整簇归并不再使用：`--candidate-size`、`--reference-size`、`--max-iterations-per-cluster`、`--embedding-dir`。

归并输出文件名会标明使用的聚类文件，避免不同参数运行互相覆盖：

```text
output/concept_abstraction/mappings/relation_mapping__from_<cluster-file-stem>.json
output/concept_abstraction/mappings/relation_merge_decisions__from_<cluster-file-stem>.jsonl
```

`mapping` 文件的关键字段：

- `cluster_file`：本次使用的聚类文件。
- `decision_file`：本次 decision JSONL 文件。
- `parameters.cluster_call_strategy`：当前为 `single_llm_call_per_cluster`。
- `parameters.singleton_strategy`：当前为 `auto_preserve_if_not_in_merge_groups`。
- `mapping`：人工审核和最终替换真正依赖的字段，以合并后概念为键。
- `merged_groups`：审计用分组明细。
- `cluster_reports`：每个 cluster 的输入数量、接受归并组数量、singleton 数量等。

`concept_merge.parse_response` 对顶层和组内 `reason` 做容错归一化，因为 LLM 有时会把 reason 返回为 `null`、列表或对象；核心的 `merge_groups` 字段仍然严格校验。

### 5.6 人工审核 mapping

应用映射前必须人工审阅 `output/concept_abstraction/mappings/*_mapping__from_*.json`。

真正用于替换的是 `mapping` 字段。它以合并后的概念为键，值里最关键的是 `source_concepts`。

如果不同意某个合并，把源概念从原合并键的 `source_concepts` 移出，并为它单独建立同名键。例如：

```json
"美国总统": {
  "merged_description": "美国总统",
  "source_concepts": ["美国总统"],
  "group_ids": ["entity_merge_0001"],
  "cluster_ids": ["c0001"]
}
```

如果要手工新增合并，把多个源概念放进同一个键的 `source_concepts` 列表。

### 5.7 应用映射到五元组

人工审核完成后运行：

```powershell
python run_concept_apply_mapping.py
```

默认输入：`data/tuple_pocessed&sorted` 和 `output/concept_abstraction/mappings`。

默认输出：

- `data/tuple_pocessed&sorted&abstracted/*.json`
- `data/tuple_pocessed&sorted&abstracted/apply_mapping_report.json`

读取 mapping 的逻辑：

- 对每个类型，优先读取最新修改的 `*_mapping__from_*.json`。
- 如果没有新命名文件，则回退到旧版固定文件名 `entity_mapping.json`、`relation_mapping.json`、`location_mapping.json`。
- 如果同一类型保留了多份 mapping 变体，运行 apply 前要确认最新修改时间对应的是你已审核的那份。

替换规则：

- `subject` 和 `object` 使用 entity mapping。
- `relation` 使用 relation mapping。
- `location` 使用 location mapping。
- 字段为列表时会逐项替换并去重。
- 替换后如果两条五元组的 `subject/relation/object/location/time` 完全相同，只保留第一次出现的记录。

## 6. 断点续跑与缓存

- 主流水线 JSONL 文件末尾存在 `{"__done__": true}` 时，默认视为阶段完成，可用于跳过已完成阶段。
- 合规性过滤：`concept_compliance_results.jsonl` 已有记录会被跳过；`--overwrite` 删除并重跑。
- 描述生成：`concept_description_results.jsonl` 一条生成一条写入，按当前 `structured_fields_v2` 和 `custom_id` 续跑。
- 聚类 embedding：默认复用 `output/concept_abstraction/embeddings` 中的向量；`--overwrite-embeddings` 强制重算。
- 聚类文件：Leiden 输出文件名带参数和结果摘要，通常不会覆盖旧文件。
- 归并 decision：`*_merge_decisions__from_<cluster-file>.jsonl` 按 cluster signature 缓存；`--overwrite` 清空当前 cluster 文件对应缓存。
- 归并 mapping：`*_mapping__from_<cluster-file>.json` 按 cluster 文件名保存，不覆盖其他 cluster 的结果。

## 7. Prompt 和解析器

主流水线 prompt：

- `prompts/md/news_process.md`
- `prompts/md/extract_plan.md`
- `prompts/md/tuple_extract.md`
- `prompts/md/describe_entity.md`
- `prompts/md/describe_relation.md`
- `prompts/md/describe_location.md`
- `prompts/md/dedup_check.md`
- `prompts/md/canonical_select.md`
- `prompts/md/tuple_normalize.md`

概念抽象 prompt：

- `prompts/md/concept_compliance.md`
- `prompts/md/concept_description_entity.md`
- `prompts/md/concept_description_relation.md`
- `prompts/md/concept_description_location.md`
- `prompts/md/concept_merge_entity.md`
- `prompts/md/concept_merge_relation.md`
- `prompts/md/concept_merge_location.md`

对应 Python 解析器在 `prompts/*.py`。修改 prompt 输出 schema 时，一定同步修改 parser 和 tests。

## 8. 常用开发命令

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

运行 concept abstraction 主测试：

```powershell
python -m unittest tests.test_concept_abstraction_workflow
python -m unittest tests.test_cluster_visualization
```

检查主流水线依赖和计划：

```powershell
python main.py --run-all --dry-run
```

注意：仓库规则要求临时测试脚本也放到 `tests/` 目录下。

## 9. 常见问题

### 9.1 报错“未配置 API Key”

说明执行阶段依赖 LLM 或 embedding API，但 `config.yaml` 的 `api_key` 为空，且没有设置 `SILICONFLOW_API_KEY`。

### 9.2 频繁 HTTP 429

优先调大 `config.yaml` 中的 `api_min_interval_seconds`，必要时降低 `--chunk-size`，并保持 `api_retry_count` 足够大。描述生成和归并阶段都是高频 LLM 调用，最容易触发限流。

### 9.3 为什么 S1 不直接读 data/processed

当前事件抽取规划建立在时间线校正后的篇章上，所以 S1 直接消费 `data/processed_timeline`。时间线校正仍是独立脚本：

```powershell
python run_timeline_correction.py --scope M016-M032
```

### 9.4 为什么 apply mapping 用的不是我想要的 mapping 文件

`run_concept_apply_mapping.py` 当前只接收 `--mapping-dir`，每个类型会自动选择最新修改的 `*_mapping__from_*.json`。如果目录里同类型有多份候选 mapping，确认最新修改时间，或临时移走不想使用的旧变体。

### 9.5 S7 为什么可能会修改 canonical 文件

这是主流水线的 fallback 行为：当未命中 alias map 且未开启严格模式时，S7 会自动补写兜底 canonical，并同步更新 alias map。

## 10. 给接手者的重点提示

- 当前最新概念抽象逻辑主要在 `tools/concept_abstraction/workflow.py`。
- 聚类可视化逻辑在 `tools/concept_abstraction/cluster_visualization.py`。
- 结构化字段 schema 和权重在 `prompts/concept_description.py`。
- 整簇归并 prompt 在 `prompts/md/concept_merge_*.md`。
- Leiden 聚类输出文件名已经包含关键参数、field weights、簇数量和 hash。
- 归并 mapping / decision 输出文件名已经包含所用聚类文件 stem。
- `run_concept_apply_mapping.py` 会优先读取最新的新命名 mapping 文件，并兼容旧固定命名。
- 旧的 candidate/reference 迭代归并参数仍在 CLI 中，但现在不参与算法。
- 修改 prompt 或输出 schema 后，优先补 `tests/test_concept_abstraction_workflow.py`。
