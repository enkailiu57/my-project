# 概念描述聚类与抽象合并流程指南

本流程用于对 `data/concept` 中的 `entity`、`relation`、`location` 三类概念做合规性过滤、抽象描述生成、语义聚类、簇内 LLM 合并判断，并在人工审核后把概念映射应用回 `data/tuple_pocessed&sorted` 五元组。

当前概念池规模：`entity=834`，`relation=1858`，`location=329`，合计 `3021` 个概念。完整合规检验和描述生成至少约 `6042` 次 LLM 调用，簇内合并还会继续增加调用。建议优先用 `sync` 小范围试跑，确认 prompt 风格后再切到 `batch` 或全量运行。

## 1. 合规性检验

```powershell
python run_concept_compliance_filter.py --mode sync --chunk-size 20
```

默认输入：

- `data/concept/entity_pool.json`
- `data/concept/relation_pool.json`
- `data/concept/location_pool.json`
- `data/tuple_pocessed&sorted/*.json`

默认输出：

- `output/concept_abstraction/compliance/concept_compliance_results.jsonl`
- `output/concept_abstraction/compliance/summary.json`
- `output/concept_abstraction/compliant_pools/{entity,relation,location}_pool.json`
- `output/concept_abstraction/invalid/invalid_concepts.json`
- `output/concept_abstraction/invalid/invalid_tuples.jsonl`

审阅重点：先看 `invalid_concepts.json`，确认被剔除的概念确实不适合进入抽象聚类；再看 `invalid_tuples.jsonl`，它记录了这些不合规概念被哪些五元组引用。

支持续跑：默认会跳过 `concept_compliance_results.jsonl` 中已有结果。需要重跑时加 `--overwrite`。

小范围试跑示例：

```powershell
python run_concept_compliance_filter.py --element-types relation --mode sync --chunk-size 10
```

## 2. 生成结构化抽象描述

```powershell
python run_concept_description_generate.py --mode sync --chunk-size 20
```

默认读取合规池：`output/concept_abstraction/compliant_pools`。

描述生成现在按概念类型使用三套独立提示词：

- entity：`主体类别`、`核心角色`
- relation：`关系类别`、`核心议题`、`战略作用`
- location：`空间类型`、`区位范围`、`议题作用`

关系的 `关系类别` 必须是真实事件或行为类别，例如 `军事行动`、`表态信号`、`法律政策动作`、`军事冲突事件`、`国际组织参与事件`；不要写 `结果状态` 这类元标签。

描述生成的 prompt 输出现在只有 `fields`，不再要求模型额外输出 `description`。聚类和后续归并中看到的 `description` 是程序根据字段自动拼出的内部展示文本，不参与字段 embedding。

默认输出：

- `output/concept_abstraction/descriptions/concept_description_results.jsonl`
- `output/concept_abstraction/descriptions/{entity,relation,location}_descriptions.json`
- `output/concept_abstraction/descriptions/summary.json`

审阅重点：抽查每类字段，确认字段更偏“可聚类语义”，而不是复述来源句。后续 embedding 直接基于字段文本生成。

支持续跑：默认跳过当前结构化 schema 下已有的 `custom_id`。旧版单段 `description` 缓存不会被复用；需要清空当前描述缓存后重跑时加 `--overwrite`。

## 3. 字段加权 Embedding 与固定簇数聚类

```powershell
python run_concept_cluster.py --entity-cluster-counts 120,150,180 --relation-cluster-counts 260,320,380 --location-cluster-counts 45,60,75
```

默认输入：

- `output/concept_abstraction/compliant_pools`
- `output/concept_abstraction/descriptions`

默认输出：

- `output/concept_abstraction/embeddings/*_description_vectors.npz`
- `output/concept_abstraction/embeddings/*_description_field_vectors.npz`
- `output/concept_abstraction/embeddings/*_description_vector_meta.json`
- `output/concept_abstraction/reports/*_similarity_distribution.json`
- `output/concept_abstraction/clusters/{entity,relation,location}/*.json`
- `output/concept_abstraction/cluster_summary.json`

聚类方法是 `AgglomerativeClustering`，使用字段加权 cosine 和 `average` linkage。程序会分别嵌入每个结构化字段，将字段向量归一化后按权重拼接成综合向量，再按你指定的 K 值直接切出固定数量的候选簇。

当前字段权重：

- entity：`主体类别=0.45`，`核心角色=0.55`
- relation：`关系类别=0.45`，`核心议题=0.35`，`战略作用=0.20`
- location：`空间类型=0.35`，`区位范围=0.40`，`议题作用=0.25`

`*_description_vectors.npz` 保存的是加权拼接后的综合向量，仍供后续概念归并使用；`*_description_field_vectors.npz` 保存逐字段向量，主要用于排查和复核。

调参建议：

- K 越小，单个簇越大，人工审阅压力更高，误合并风险也更高。
- K 越大，簇越细，漏合并更多，但簇内纯度通常更高。
- 第一轮建议宁可稍微分细一点，因为后续还有簇内 LLM 合并。
- 当前建议首轮 K：实体 `120/150/180`，关系 `260/320/380`，地点 `45/60/75`。
- 如果只跑某一类，也可以用通用参数 `--cluster-counts 260,320,380`。

## 4. 人工选择聚类结果

打开 `output/concept_abstraction/cluster_summary.json`，选择每个类型最适合人工审核的聚类文件。例如：

- `output/concept_abstraction/clusters/entity/entity_agglomerative_k150.json`
- `output/concept_abstraction/clusters/relation/relation_agglomerative_k320.json`
- `output/concept_abstraction/clusters/location/location_agglomerative_k60.json`

选择标准：多成员簇数量适中，簇内概念语义接近，且没有大量明显跨类混合。

## 5. 簇内 LLM 合并映射

对三类概念分别运行一次，`--cluster-file` 换成你选中的聚类结果文件：

```powershell
python run_concept_merge_mapping.py --cluster-file output/concept_abstraction/clusters/entity/entity_agglomerative_k150.json --candidate-size 16 --reference-size 24 --mode sync
python run_concept_merge_mapping.py --cluster-file output/concept_abstraction/clusters/relation/relation_agglomerative_k320.json --candidate-size 16 --reference-size 24 --mode sync
python run_concept_merge_mapping.py --cluster-file output/concept_abstraction/clusters/location/location_agglomerative_k60.json --candidate-size 12 --reference-size 18 --mode sync
```

默认输出：

- `output/concept_abstraction/mappings/entity_mapping.json`
- `output/concept_abstraction/mappings/relation_mapping.json`
- `output/concept_abstraction/mappings/location_mapping.json`
- `output/concept_abstraction/mappings/*_merge_decisions.jsonl`

合并逻辑：在每个簇内维护两个集合：待处理原始概念池和不断扩充的合并概念集。每轮先从待处理池中按向量相似度采样一个高内聚候选集，再选取若干已有合并概念作为参考交给 LLM。LLM 可以把候选集中的原始概念链接到已有合并概念，也可以从剩余候选中生成一个新的合并概念并加入合并概念集。已经建立链接或被新合并的原始概念会从待处理池删除；新生成的合并概念不会放回待处理池参与后续采样。如果某轮候选集没有产生任何链接或新合并，后续采样会围绕这个无进展集合混入集合外的相似近邻，形成部分重叠的桥接候选，避免同一高内聚集合反复原样进入 LLM。当当前待处理池和已有合并概念集保持不变，并且采样器可构造的候选窗口都已经被 LLM 判定为无链接、无新合并时，聚类以 `saturated_no_actions` 退出。如果已有合并概念数量超过 `--reference-size`，退出前还会用全部已有合并概念做一轮饱和验证，避免漏掉可链接到较远已有概念的候选；验证轮数记录在 `cluster_reports[*].saturation_validation_pass_count`。如果待处理池被清空，则以 `cluster_empty` 退出。`--max-iterations-per-cluster` 只是异常安全上限，触发时退出原因为 `max_iterations_safety_limit`。

支持续跑：`*_merge_decisions.jsonl` 会缓存每个“候选集 + 已有合并概念参考集”的判断。旧版 anchor 或旧版去中心化缓存不会被新算法复用；需要清空当前阶段缓存后重跑时加 `--overwrite`。

## 6. 人工审核映射表

应用映射前，必须人工审阅 `output/concept_abstraction/mappings/*_mapping.json`。真正用于替换的是其中的 `mapping` 字段。现在 `mapping` 以合并后的概念为键，值里最关键的是 `source_concepts`。

如果不同意某个合并，就把该源概念从原键的 `source_concepts` 中移出，并为它单独建一个同名键。例如：

```json
"美国总统": {
  "merged_description": "美国最高行政与对外政策决策层的一部分。",
  "source_concepts": ["美国总统"],
  "group_ids": ["entity_merge_0001"],
  "cluster_ids": ["c0001"]
}
```

如果要手工新增合并，就把多个源概念放进同一个键的 `source_concepts` 列表。`merged_groups` 主要用于审计，不参与五元组替换。

## 7. 应用映射到五元组

人工审核完成后运行：

```powershell
python run_concept_apply_mapping.py
```

默认输入：

- `data/tuple_pocessed&sorted`
- `output/concept_abstraction/mappings`

默认输出：

- `data/tuple_pocessed&sorted&abstracted/*.json`
- `data/tuple_pocessed&sorted&abstracted/apply_mapping_report.json`

替换规则：

- `subject` 和 `object` 使用 `entity_mapping.json`。
- `relation` 使用 `relation_mapping.json`。
- `location` 使用 `location_mapping.json`。
- 字段为列表时会逐项替换并去重。
- 替换后如果两条五元组的 `subject/relation/object/location/time` 完全相同，只保留第一次出现的记录。

## 常用参数

- `--element-types entity`：只跑实体。
- `--element-types relation,location`：只跑关系和地点。
- `--cluster-counts 260,320,380`：给所有待聚类类型使用同一组 K。
- `--entity-cluster-counts 120,150,180`：单独设置实体 K，关系和地点也有对应参数。
- `--candidate-size 16`：每轮提供给 LLM 的候选概念数。越大越容易发现跨局部的合并，但单次 prompt 更长。
- `--reference-size 24`：每轮提供给 LLM 的已有合并概念参考数量。设为 `0` 表示提供全部已有合并概念。
- `--max-iterations-per-cluster 500`：单个聚类最多迭代次数的安全上限；正常终止依赖 `cluster_empty` 或 `saturated_no_actions`。
- `--mode batch`：使用 Batch 通道。流程已为分批任务生成唯一 task name，可续跑。
- `--chunk-size 10`：降低单批 LLM 任务量，适合调试。
- `--overwrite`：清空当前阶段缓存后重跑。
- `--overwrite-embeddings`：重新生成描述向量。

## 推荐全流程命令

```powershell
python run_concept_compliance_filter.py --mode sync --chunk-size 20
python run_concept_description_generate.py --mode sync --chunk-size 20
python run_concept_cluster.py --entity-cluster-counts 120,150,180 --relation-cluster-counts 260,320,380 --location-cluster-counts 45,60,75
```

人工选择每类聚类文件后：

```powershell
python run_concept_merge_mapping.py --cluster-file output/concept_abstraction/clusters/entity/entity_agglomerative_k150.json --candidate-size 16 --reference-size 24 --mode sync
python run_concept_merge_mapping.py --cluster-file output/concept_abstraction/clusters/relation/relation_agglomerative_k320.json --candidate-size 16 --reference-size 24 --mode sync
python run_concept_merge_mapping.py --cluster-file output/concept_abstraction/clusters/location/location_agglomerative_k60.json --candidate-size 12 --reference-size 18 --mode sync
```

人工审核三个 mapping 后：

```powershell
python run_concept_apply_mapping.py
```
