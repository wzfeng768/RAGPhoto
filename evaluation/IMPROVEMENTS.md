# 评估系统使用指南

## 更新日期: 2026-01-10 v4

---

## 评估指标体系

### 指标概览（共 6 个核心指标）

| 指标 | 类型 | 说明 | Direct LLM |
|------|------|------|------------|
| **context_precision** | Custom | 最相关上下文的排名位置 (1=最佳) | NaN |
| **context_recall** | RAGAS | 上下文覆盖率 | NaN |
| **answer_similarity** | RAGAS | 答案与 ground truth 的语义相似度 | ✓ |
| **answer_relevancy** | Custom | 答案与问题的相关性 | ✓ |
| **faithfulness** | Custom | 答案是否忠实于上下文 | NaN |
| **answer_correctness** | Custom | 答案的事实正确性 | ✓ |

---

## 命令行选项

### 基本用法

```bash
python evaluate_rag.py --dataset <数据集> --num_questions <数量> \
    --answer-model <模型> --eval-model <评估模型>
```

### 完整参数列表

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集选择 | `computational` |
| `--mode` | 评估模式 | `all` |
| `--num_questions` | 测试题目数量 | `10` |
| `--resume` | **继续中断的评估** | 关闭 |
| `--no-incremental` | 禁用增量保存 | 关闭 |
| `--skip-ragas` | 跳过 RAGAS 评估 | 关闭 |
| `--answer-model` | 答案生成模型 | 配置文件 |
| `--eval-model` | 评估模型 | 配置文件 |

### 数据集选项

- `all` - 所有数据集
- `computational` - 计算相关
- `characterization` - 表征相关
- `stability` - 稳定性相关
- `materials` - 材料相关
- `device` - 器件相关
- `structure` - 结构相关
- `processing` - 加工相关
- `performance` - 性能相关

### 模式选项

- `all` - 运行所有三种模式
- `agentic_with_kg` - Agentic RAG + 知识图谱
- `agentic_no_kg` - Agentic RAG (仅向量)
- `direct_llm` - 直接 LLM (无检索)

---

## 中断恢复功能

### 情况1: 评估中断后继续

如果评估过程中断（Ctrl+C 或错误），可以使用 `--resume` 继续：

```bash
# 继续之前中断的评估（跳过已完成的问题）
python evaluate_rag.py --dataset all --num_questions 10 \
    --answer-model gpt-4o-mini --eval-model gemini-3-flash-preview \
    --resume
```

### 情况2: 重新开始评估

删除已有结果文件后重新运行：

```bash
# 删除特定数据集的结果
rm -rf results/all/agentic_with_kg/results.json
rm -rf results/all/agentic_no_kg/results.json
rm -rf results/all/direct_llm/results.json

# 重新评估
python evaluate_rag.py --dataset all --num_questions 10 \
    --answer-model gpt-4o-mini --eval-model gemini-3-flash-preview
```

### 情况3: 清空所有结果重新评估

```bash
# 删除所有结果
rm -rf results/

# 重新评估
python evaluate_rag.py --dataset all --num_questions all \
    --answer-model gpt-4o-mini --eval-model gemini-3-flash-preview
```

---

## 常用命令示例

### 快速测试（1题）

```bash
python evaluate_rag.py --dataset all --num_questions 1 \
    --answer-model gpt-4o-mini --eval-model gemini-3-flash-preview
```

### 标准评估（10题）

```bash
python evaluate_rag.py --dataset all --num_questions 10 \
    --answer-model gpt-4o-mini --eval-model gemini-3-flash-preview
```

### 全量评估

```bash
python evaluate_rag.py --dataset all --num_questions all \
    --answer-model gpt-4o-mini --eval-model gemini-3-flash-preview
```

### 只评估特定模式

```bash
# 只评估 Agentic RAG + KG
python evaluate_rag.py --dataset all --num_questions 10 \
    --mode agentic_with_kg \
    --answer-model gpt-4o-mini --eval-model gemini-3-flash-preview
```

---

## 输出文件结构

```
results/
└── <dataset>/
    ├── agentic_with_kg/
    │   └── results.json
    ├── agentic_no_kg/
    │   └── results.json
    ├── direct_llm/
    │   └── results.json
    └── evaluation_summary.json
```

---

## 指标详细说明

### context_precision (上下文精确度)
- **计算方式**: 最相关上下文的排名位置
- **取值**: 1, 2, 3... (数字越小越好)
- **平均计算**: 使用倒数 1/rank (MRR 风格)

### context_recall (上下文召回率)
- **计算方式**: RAGAS 
- **取值**: 0.0 - 1.0

### answer_similarity (答案相似度)
- **计算方式**: RAGAS embedding 相似度
- **取值**: 0.0 - 1.0

### answer_relevancy (答案相关性)
- **计算方式**: 语义相似度 40% + LLM 评估 60%
- **取值**: 0.0 - 1.0

### faithfulness (忠实度)
- **计算方式**: LLM 评估答案是否有上下文支持
- **取值**: 0.0 - 1.0

### answer_correctness (答案正确性)
- **计算方式**: LLM 评估事实准确性 + 语义等价性
- **取值**: 0.0 - 1.0
