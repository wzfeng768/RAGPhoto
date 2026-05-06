#!/usr/bin/env python3
"""
跨模型对比可视化脚本
生成多模型之间的对比图表:
1. 多模型雷达图对比
2. 分组柱状图 - 按模型和模式对比所有指标
3. 热力图 - 指标 x 模型交叉对比
4. 模型排名图 - 各指标下模型的排名
5. 综合评分对比

使用方法:
    # 对比单个类别
    python compare_models.py --category characterization
    python compare_models.py --mode agentic_with_kg
    python compare_models.py --category characterization --mode agentic_with_kg
    
    # 使用所有类别汇总对比
    python compare_models.py --all
    python compare_models.py --all --mode agentic_with_kg
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
import argparse

matplotlib.use('Agg')
import matplotlib.ticker as mticker

_Y_FMT = mticker.FormatStrFormatter('%.2f')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型颜色配置
MODEL_COLORS = {
    'Gpt-4o-mini': '#ADC0D1',
    'Qwen3-235B': '#9BC3C6',
    'DeepSeek-V3.2-Thinking': '#F4E5BE',
    'GLM-4.7': '#D4D2D2',
    'MiniMax-M2.1': '#B8D4C8',
    'Llama-3.3-70B': '#E8C8B8',
}

# 模式颜色
MODE_COLORS = {
    'direct_llm': '#ADC0D1',
    'agentic_no_kg': '#9BC3C6',
    'agentic_with_kg': '#F4E5BE',
}

METRICS = ['answer_correctness', 'answer_similarity', 'answer_relevancy',
           'faithfulness', 'context_recall', 'context_precision']

METRIC_LABELS = {
    'answer_correctness': 'Correctness',
    'answer_similarity': 'Similarity',
    'answer_relevancy': 'Relevancy',
    'faithfulness': 'Faithfulness',
    'context_recall': 'Recall',
    'context_precision': 'Precision'
}
EXPORT_DPI = 300


# 非模型的子目录名称（需要跳过）
NON_MODEL_DIRS = {'model_comparison', 'visualize_results'}


def load_all_summaries(visualize_dir: Path) -> dict:
    """加载所有模型的 summary_data.json"""
    all_data = {}
    
    for model_dir in sorted(visualize_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        # 跳过非模型目录（如 model_comparison）
        if model_dir.name in NON_MODEL_DIRS:
            continue
        
        summary_file = model_dir / 'summary_data.json'
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                all_data[model_dir.name] = json.load(f)
            print(f"  ✓ 已加载: {model_dir.name}")
        else:
            print(f"  ⚠️  缺少 summary_data.json: {model_dir.name}")
    
    return all_data


def get_metric_value(data: dict, category: str, mode: str, metric: str) -> float:
    """从嵌套数据中获取指标值"""
    try:
        categories = data.get('categories', {})
        cat_data = categories.get(category, {})
        modes = cat_data.get('modes', {})
        mode_data = modes.get(mode, {})
        metrics = mode_data.get('ragas_metrics', {})
        value = metrics.get(metric)
        
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0
        return value
    except Exception:
        return 0.0


def has_category_data(data: dict, category: str, mode: str = None) -> bool:
    """检查模型是否有指定类别的数据"""
    try:
        categories = data.get('categories', {})
        if category not in categories:
            return False
        cat_data = categories.get(category, {})
        modes = cat_data.get('modes', {})
        
        if mode:
            # 检查特定模式
            if mode not in modes:
                return False
            mode_data = modes.get(mode, {})
            metrics = mode_data.get('ragas_metrics', {})
            # 至少有一个非零指标
            return any(v and v > 0 for v in metrics.values() if isinstance(v, (int, float)))
        else:
            # 检查任何模式
            return len(modes) > 0
    except Exception:
        return False


def filter_models_with_data(all_data: dict, category: str, mode: str) -> dict:
    """过滤出有指定类别数据的模型"""
    return {
        model_name: model_data 
        for model_name, model_data in all_data.items() 
        if has_category_data(model_data, category, mode)
    }


def aggregate_all_categories(all_data: dict, mode: str) -> dict:
    """汇总所有类别的指标，计算加权平均（按样本数量加权）"""
    aggregated = {}
    
    for model_name, model_data in all_data.items():
        categories = model_data.get('categories', {})
        
        # 收集各类别的指标值和样本数
        metric_sums = {m: 0.0 for m in METRICS}
        metric_counts = {m: 0 for m in METRICS}
        
        for cat_name, cat_data in categories.items():
            modes = cat_data.get('modes', {})
            mode_data = modes.get(mode, {})
            metrics = mode_data.get('ragas_metrics', {})
            num_questions = mode_data.get('num_questions', 1)
            
            for metric in METRICS:
                value = metrics.get(metric)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    metric_sums[metric] += value * num_questions
                    metric_counts[metric] += num_questions
        
        # 计算加权平均
        avg_metrics = {}
        for metric in METRICS:
            if metric_counts[metric] > 0:
                avg_metrics[metric] = metric_sums[metric] / metric_counts[metric]
            else:
                avg_metrics[metric] = 0.0
        
        # 构建与原始数据结构兼容的格式
        aggregated[model_name] = {
            'categories': {
                'all': {
                    'modes': {
                        mode: {
                            'ragas_metrics': avg_metrics,
                            'num_questions': sum(metric_counts.values()) // len(METRICS)
                        }
                    }
                }
            }
        }
    
    return aggregated


def get_all_categories(all_data: dict) -> set:
    """获取所有可用的类别"""
    available_categories = set()
    for model_data in all_data.values():
        available_categories.update(model_data.get('categories', {}).keys())
    return available_categories


def aggregate_all_categories_all_modes(all_data: dict) -> dict:
    """汇总所有类别、所有模式的指标"""
    modes = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    aggregated = {}
    
    for model_name, model_data in all_data.items():
        categories = model_data.get('categories', {})
        
        mode_aggregated = {}
        for mode in modes:
            metric_sums = {m: 0.0 for m in METRICS}
            metric_counts = {m: 0 for m in METRICS}
            
            for cat_name, cat_data in categories.items():
                modes_data = cat_data.get('modes', {})
                mode_data = modes_data.get(mode, {})
                metrics = mode_data.get('ragas_metrics', {})
                num_questions = mode_data.get('num_questions', 1)
                
                for metric in METRICS:
                    value = metrics.get(metric)
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        metric_sums[metric] += value * num_questions
                        metric_counts[metric] += num_questions
            
            avg_metrics = {}
            for metric in METRICS:
                if metric_counts[metric] > 0:
                    avg_metrics[metric] = metric_sums[metric] / metric_counts[metric]
                else:
                    avg_metrics[metric] = 0.0
            
            mode_aggregated[mode] = {
                'ragas_metrics': avg_metrics,
                'num_questions': sum(metric_counts.values()) // len(METRICS) if metric_counts else 0
            }
        
        aggregated[model_name] = {
            'categories': {
                'all': {
                    'modes': mode_aggregated
                }
            }
        }
    
    return aggregated


# ============================================================================
# 共同双条件QA对比函数
# ============================================================================

def load_common_qa_ids(common_qa_file: Path) -> set:
    """从 common_qa_across_models.txt 加载所有模型共同满足双条件的 QA ID"""
    common_qa_ids = set()
    
    if not common_qa_file.exists():
        print(f"⚠️  共同QA文件不存在: {common_qa_file}")
        return common_qa_ids
    
    in_dual_condition_section = False
    
    with open(common_qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 进入双条件section
            if '所有模型都满足双条件的QA' in line:
                in_dual_condition_section = True
                continue
            
            # 离开双条件section（进入其他section）
            if line.startswith('##') and in_dual_condition_section:
                break
            
            # 提取 QA ID
            if in_dual_condition_section and line.startswith('qa_'):
                common_qa_ids.add(line)
    
    return common_qa_ids


def load_dual_condition_qa_by_model_count(results_dir: Path, min_models: int = 5) -> tuple:
    """从各模型的 dual_condition_report.txt 加载满足双条件的 QA ID
    
    Args:
        results_dir: 结果目录路径
        min_models: 至少要有多少个模型满足双条件
        
    Returns:
        (qa_ids: set, qa_model_count: dict, total_models: int)
    """
    # 收集每个模型满足双条件的 QA
    model_qa_sets = {}  # {model_name: set(qa_ids)}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in ['visualize_results', 'model_comparison']:
            continue
        
        model_name = model_dir.name
        report_file = model_dir / 'dual_condition_report.txt'
        
        if not report_file.exists():
            continue
        
        qa_ids = set()
        in_dual_section = False
        
        with open(report_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 进入满足双条件区域
                if '满足双条件的QA' in line and 'with_kg >= no_kg' in line:
                    in_dual_section = True
                    continue
                
                # 离开该区域 (进入其他区域)
                if in_dual_section and ('仅满足条件' in line or '高分但no_kg更优' in line):
                    break
                
                # 提取 QA ID
                if in_dual_section and line.startswith('qa_'):
                    qa_ids.add(line)
        
        model_qa_sets[model_name] = qa_ids
    
    total_models = len(model_qa_sets)
    
    if total_models == 0:
        return set(), {}, 0
    
    # 统计每个 QA 出现在多少个模型中
    qa_model_count = {}  # {qa_id: count}
    all_qa_ids = set()
    for qa_set in model_qa_sets.values():
        all_qa_ids.update(qa_set)
    
    for qa_id in all_qa_ids:
        count = sum(1 for model_qas in model_qa_sets.values() if qa_id in model_qas)
        qa_model_count[qa_id] = count
    
    # 筛选满足最小模型数的 QA
    filtered_qa_ids = {qa_id for qa_id, count in qa_model_count.items() if count >= min_models}
    
    return filtered_qa_ids, qa_model_count, total_models


def load_qa_category_map(results_dir: Path) -> dict:
    """从各模型的 results.json 中加载 QA ID 到 category 的映射"""
    qa_category_map = {}  # {qa_id: category}
    
    # 遍历任意一个模型的目录
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in ['visualize_results', 'model_comparison']:
            continue
        
        for category_dir in model_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            
            # 优先使用 agentic_with_kg 的结果
            results_file = category_dir / 'agentic_with_kg' / 'results.json'
            if not results_file.exists():
                results_file = category_dir / 'agentic_no_kg' / 'results.json'
            if not results_file.exists():
                results_file = category_dir / 'direct_llm' / 'results.json'
            
            if results_file.exists():
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for qa_result in data.get('qa_results', []):
                        qa_id = qa_result.get('question_id')
                        if qa_id and qa_id not in qa_category_map:
                            qa_category_map[qa_id] = category_name
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # 只需要处理一个模型就够了
        if qa_category_map:
            break
    
    return qa_category_map


def aggregate_metrics_from_common_qa(results_dir: Path, common_qa_ids: set, 
                                     qa_category_map: dict, category_filter: str = None) -> dict:
    """从原始 results.json 中读取 QA 级别数据，仅使用共同 QA 计算指标均值"""
    modes = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    aggregated = {}
    
    # 如果指定了类别过滤，筛选 QA
    if category_filter:
        filtered_qa_ids = {qa_id for qa_id in common_qa_ids 
                          if qa_category_map.get(qa_id) == category_filter}
    else:
        filtered_qa_ids = common_qa_ids
    
    if not filtered_qa_ids:
        print(f"⚠️  没有符合条件的共同QA")
        return aggregated
    
    print(f"📊 使用 {len(filtered_qa_ids)} 个共同双条件QA计算指标")
    if category_filter:
        print(f"   类别过滤: {category_filter}")
    
    # 遍历每个模型
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ['visualize_results', 'model_comparison']:
            continue
        
        model_name = model_dir.name
        mode_aggregated = {}
        
        for mode in modes:
            metric_sums = {m: 0.0 for m in METRICS}
            metric_counts = {m: 0 for m in METRICS}
            
            # 遍历所有类别目录
            for category_dir in model_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                
                # 如果指定了类别，只处理该类别
                if category_filter and category_dir.name != category_filter:
                    continue
                
                results_file = category_dir / mode / 'results.json'
                if not results_file.exists():
                    continue
                
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for qa_result in data.get('qa_results', []):
                        qa_id = qa_result.get('question_id')
                        
                        # 只处理共同QA
                        if qa_id not in filtered_qa_ids:
                            continue
                        
                        ragas_scores = qa_result.get('ragas_scores', {})
                        
                        for metric in METRICS:
                            value = ragas_scores.get(metric)
                            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                                metric_sums[metric] += value
                                metric_counts[metric] += 1
                                
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # 计算均值
            avg_metrics = {}
            for metric in METRICS:
                if metric_counts[metric] > 0:
                    avg_metrics[metric] = metric_sums[metric] / metric_counts[metric]
                else:
                    avg_metrics[metric] = 0.0
            
            mode_aggregated[mode] = {
                'ragas_metrics': avg_metrics,
                'num_questions': metric_counts.get('answer_correctness', 0)
            }
        
        aggregated[model_name] = {
            'categories': {
                'common_qa': {
                    'modes': mode_aggregated
                }
            }
        }
    
    return aggregated


def create_multi_model_line_chart(all_data: dict, category: str, output_path: str, category_label: str = None):
    """创建多模型折线图 - 横坐标为模式，纵坐标为 answer_correctness"""
    modes = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    mode_labels = ['Direct LLM', 'Agentic (No KG)', 'Agentic (With KG)']
    
    # 过滤有数据的模型
    filtered_data = filter_models_with_data(all_data, category, modes[0])
    if not filtered_data:
        # 尝试其他模式
        for mode in modes[1:]:
            filtered_data = filter_models_with_data(all_data, category, mode)
            if filtered_data:
                break
    
    if not filtered_data:
        print(f"⚠️  没有模型有 {category} 的数据，跳过折线图")
        return
    
    # 按 agentic_with_kg 的 correctness 从高到低排序模型
    model_scores = []
    for model_name in filtered_data.keys():
        correctness = get_metric_value(filtered_data[model_name], category, 'agentic_with_kg', 'answer_correctness')
        model_scores.append((model_name, correctness))
    model_scores.sort(key=lambda x: x[1], reverse=True)
    models = [m[0] for m in model_scores]
    mode_suffix_map = {
        'direct_llm': 'direct',
        'agentic_no_kg': 'no kg',
        'agentic_with_kg': 'with kg',
    }
    display_models = [f"{model} ({mode_suffix_map.get(mode, mode.replace('_', ' '))})" for model in models]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(modes))
    
    # 动态计算偏移量使得各模型的线在 x 轴上稍微错开避免完全重合
    width = 0.3
    offsets = np.linspace(-width/2, width/2, len(models)) if len(models) > 1 else [0]
    
    # 线条样式
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    for idx, model_name in enumerate(models):
        model_data = filtered_data[model_name]
        correctness_values = []
        
        for mode in modes:
            value = get_metric_value(model_data, category, mode, 'answer_correctness')
            correctness_values.append(value)
        
        color = MODEL_COLORS.get(model_name, f'C{idx}')
        marker = markers[idx % len(markers)]
        
        # 应用偏移量
        x_shifted = x + offsets[idx]
        
        ax.plot(x_shifted, correctness_values, '-', linewidth=2.5, markersize=10,
               label=model_name, color=color, marker=marker, alpha=0.85)
        
        # 添加数值标签 (交错高度避免重叠)
        for i, val in enumerate(correctness_values):
            if val > 0:
                y_offset = 10 if idx % 2 == 0 else -15
                va = 'bottom' if idx % 2 == 0 else 'top'
                ax.annotate(f'{val:.2f}', (x_shifted[i], val), 
                           textcoords="offset points", xytext=(0, y_offset),
                           ha='center', va=va, fontsize=9, color=color, fontweight='bold', alpha=0.9)
    
    ax.set_ylabel('Answer Correctness', fontsize=12)
    ax.set_xlabel('Mode', fontsize=12)
    
    title = 'Multi-Model Comparison - Answer Correctness by Mode'
    if category_label:
        title += f'\n({category_label})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=11)
    ax.set_ylim(0.2, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Multi-model line chart saved: {output_path}")


def create_multi_model_radar(all_data: dict, category: str, mode: str, output_path: str):
    """创建多模型雷达图对比"""
    # 过滤有数据的模型
    filtered_data = filter_models_with_data(all_data, category, mode)
    if not filtered_data:
        print(f"⚠️  没有模型有 {category}/{mode} 的数据，跳过雷达图")
        return

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(polar=True))

    num_metrics = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    labels = [METRIC_LABELS.get(m, m) for m in METRICS]

    # 按 correctness 从高到低排序模型
    model_scores = []
    for model_name in filtered_data.keys():
        correctness = get_metric_value(filtered_data[model_name], category, mode, 'answer_correctness')
        model_scores.append((model_name, correctness))
    model_scores.sort(key=lambda x: x[1], reverse=True)
    models = [m[0] for m in model_scores]
    mode_suffix_map = {
        'direct_llm': 'Direct',
        'agentic_no_kg': 'No KG',
        'agentic_with_kg': 'With KG',
    }
    legend_labels = [f"{model} ({mode_suffix_map.get(mode, mode.replace('_', ' '))})" for model in models]
    colors = [MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(models)]

    for idx, model_name in enumerate(models):
        model_data = filtered_data[model_name]

        values = []
        for metric in METRICS:
            v = get_metric_value(model_data, category, mode, metric)
            values.append(v)

        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=1.8, markersize=4.5, color=colors[idx])
        ax.fill(angles, values, alpha=0.12, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.tick_params(axis='x', pad=14)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.20', '0.40', '0.60', '0.80', '1.00'], size=9)

    ax.legend(
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        ncol=min(3, len(legend_labels)),
        fontsize=10,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.5,
        borderaxespad=0,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()

    print(f"✅ Multi-model radar chart saved: {output_path}")


def create_grouped_bar_chart(all_data: dict, category: str, mode: str, output_path: str):
    """创建分组柱状图 - 按指标分组，对比各模型"""
    # 过滤有数据的模型
    filtered_data = filter_models_with_data(all_data, category, mode)
    if not filtered_data:
        print(f"⚠️  没有模型有 {category}/{mode} 的数据，跳过柱状图")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 按 correctness 从高到低排序模型
    model_scores = []
    for model_name in filtered_data.keys():
        correctness = get_metric_value(filtered_data[model_name], category, mode, 'answer_correctness')
        model_scores.append((model_name, correctness))
    model_scores.sort(key=lambda x: x[1], reverse=True)
    models = [m[0] for m in model_scores]
    mode_suffix_map = {
        'direct_llm': 'direct',
        'agentic_no_kg': 'no kg',
        'agentic_with_kg': 'with kg',
    }
    display_models = [f"{model} ({mode_suffix_map.get(mode, mode.replace('_', ' '))})" for model in models]
    
    n_models = len(models)
    n_metrics = len(METRICS)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    for idx, model_name in enumerate(models):
        model_data = filtered_data[model_name]
        
        values = [get_metric_value(model_data, category, mode, m) for m in METRICS]
        
        offset = (idx - n_models/2 + 0.5) * width
        color = MODEL_COLORS.get(model_name, f'C{idx}')
        bars = ax.bar(x + offset, values, width, label=model_name, color=color, alpha=0.85)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_title(f'Model Comparison by Metrics - {category} ({mode})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in METRICS], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grouped bar chart saved: {output_path}")


def create_heatmap(all_data: dict, category: str, mode: str, output_path: str):
    """创建热力图 - 模型 x 指标"""
    # 过滤有数据的模型
    filtered_data = filter_models_with_data(all_data, category, mode)
    if not filtered_data:
        print(f"⚠️  没有模型有 {category}/{mode} 的数据，跳过热力图")
        return
    
    # 按 correctness 从高到低排序模型
    model_scores = []
    for model_name in filtered_data.keys():
        correctness = get_metric_value(filtered_data[model_name], category, mode, 'answer_correctness')
        model_scores.append((model_name, correctness))
    model_scores.sort(key=lambda x: x[1], reverse=True)
    models = [m[0] for m in model_scores]
    mode_suffix_map = {
        'direct_llm': 'direct',
        'agentic_no_kg': 'no kg',
        'agentic_with_kg': 'with kg',
    }
    display_models = [f"{model} ({mode_suffix_map.get(mode, mode.replace('_', ' '))})" for model in models]
    
    n_models = len(models)
    n_metrics = len(METRICS)
    
    # 构建数据矩阵
    matrix = np.zeros((n_models, n_metrics))
    
    for i, model_name in enumerate(models):
        for j, metric in enumerate(METRICS):
            matrix[i, j] = get_metric_value(filtered_data[model_name], category, mode, metric)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    from matplotlib.colors import LinearSegmentedColormap
    _heatmap_cmap = LinearSegmentedColormap.from_list('WhiteGn', ['#ffffff', '#9BC3C6', '#ADC0D1'])
    im = ax.imshow(matrix, cmap=_heatmap_cmap, aspect='auto', vmin=0, vmax=1)
    
    # 设置刻度
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in METRICS], fontsize=11)
    ax.set_yticklabels(display_models, fontsize=11)
    
    # 添加数值标注
    for i in range(n_models):
        for j in range(n_metrics):
            val = matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Heatmap saved: {output_path}")


def create_mode_comparison_chart(all_data: dict, category: str, output_path: str):
    """创建模式对比图 - 对比 direct_llm, agentic_no_kg, agentic_with_kg"""
    modes = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    
    # 过滤至少有一个模式有数据的模型
    filtered_data = {}
    for model_name, model_data in all_data.items():
        if has_category_data(model_data, category):
            filtered_data[model_name] = model_data
    
    if not filtered_data:
        print(f"⚠️  没有模型有 {category} 的数据，跳过模式对比图")
        return
    
    # 按 agentic_with_kg 的 correctness 从高到低排序模型
    model_scores = []
    for model_name in filtered_data.keys():
        correctness = get_metric_value(filtered_data[model_name], category, 'agentic_with_kg', 'answer_correctness')
        model_scores.append((model_name, correctness))
    model_scores.sort(key=lambda x: x[1], reverse=True)
    models = [m[0] for m in model_scores]
    mode_suffix_map = {
        'direct_llm': 'direct',
        'agentic_no_kg': 'no kg',
        'agentic_with_kg': 'with kg',
    }
    display_models = [f"{model} ({mode_suffix_map.get(mode, mode.replace('_', ' '))})" for model in models]
    
    # 使用 answer_correctness 作为主要指标
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: Answer Correctness 对比
    ax1 = axes[0]
    x = np.arange(len(modes))
    width = 0.8 / len(models)
    
    for idx, model_name in enumerate(models):
        model_data = filtered_data[model_name]
        values = [get_metric_value(model_data, category, mode, 'answer_correctness') for mode in modes]
        
        offset = (idx - len(models)/2 + 0.5) * width
        color = MODEL_COLORS.get(model_name, f'C{idx}')
        bars = ax1.bar(x + offset, values, width, label=model_name, color=color, alpha=0.85)
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('Answer Correctness', fontsize=12)
    ax1.set_title('Answer Correctness by Mode', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Direct LLM', 'Agentic (No KG)', 'Agentic (With KG)'], fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(_Y_FMT)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图: 平均得分对比
    ax2 = axes[1]
    
    for idx, model_name in enumerate(models):
        model_data = filtered_data[model_name]
        avg_scores = []
        
        for mode in modes:
            values = [get_metric_value(model_data, category, mode, m) for m in METRICS]
            # 只计算非零值的平均
            valid_values = [v for v in values if v > 0]
            avg = np.mean(valid_values) if valid_values else 0
            avg_scores.append(avg)
        
        offset = (idx - len(models)/2 + 0.5) * width
        color = MODEL_COLORS.get(model_name, f'C{idx}')
        bars = ax2.bar(x + offset, avg_scores, width, label=model_name, color=color, alpha=0.85)
        
        for bar, val in zip(bars, avg_scores):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('Average Score', fontsize=12)
    ax2.set_title('Overall Average Score by Mode', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Direct LLM', 'Agentic (No KG)', 'Agentic (With KG)'], fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.set_major_formatter(_Y_FMT)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Mode Comparison - {category}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Mode comparison chart saved: {output_path}")


def create_improvement_chart(all_data: dict, category: str, output_path: str):
    """创建提升效果图 - 对比使用 KG 前后的提升"""
    # 过滤有数据的模型
    filtered_data = {}
    for model_name, model_data in all_data.items():
        if has_category_data(model_data, category):
            filtered_data[model_name] = model_data
    
    if not filtered_data:
        print(f"⚠️  没有模型有 {category} 的数据，跳过提升效果图")
        return
    
    # 按 agentic_with_kg 的 correctness 从高到低排序模型
    model_scores = []
    for model_name in filtered_data.keys():
        model_data = filtered_data[model_name]
        with_kg = get_metric_value(model_data, category, 'agentic_with_kg', 'answer_correctness')
        model_scores.append((model_name, with_kg))
    model_scores.sort(key=lambda x: x[1], reverse=True)
    models = [m[0] for m in model_scores]
    mode_suffix_map = {
        'direct_llm': 'direct',
        'agentic_no_kg': 'no kg',
        'agentic_with_kg': 'with kg',
    }
    display_models = [f"{model} ({mode_suffix_map.get(mode, mode.replace('_', ' '))})" for model in models]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # 计算各模型的提升幅度
    no_kg_scores = []
    with_kg_scores = []
    improvements = []
    
    for model_name in models:
        model_data = filtered_data[model_name]
        
        no_kg = get_metric_value(model_data, category, 'agentic_no_kg', 'answer_correctness')
        with_kg = get_metric_value(model_data, category, 'agentic_with_kg', 'answer_correctness')
        
        no_kg_scores.append(no_kg)
        with_kg_scores.append(with_kg)
        improvements.append((with_kg - no_kg) * 100)  # 百分比
    
    # 绘制
    bars1 = ax.bar(x - width/2, no_kg_scores, width, label='Agentic (No KG)', color='#9BC3C6', alpha=0.85)
    bars2 = ax.bar(x + width/2, with_kg_scores, width, label='Agentic (With KG)', color='#F4E5BE', alpha=0.85)
    
    # 添加数值和提升箭头
    for i, (b1, b2, imp) in enumerate(zip(bars1, bars2, improvements)):
        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.02,
               f'{no_kg_scores[i]:.2f}', ha='center', va='bottom', fontsize=10)
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.02,
               f'{with_kg_scores[i]:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 添加提升指示
        color = 'green' if imp > 0 else 'red'
        symbol = '↑' if imp > 0 else '↓'
        ax.annotate(f'{symbol}{abs(imp):.1f}%', 
                   xy=(x[i], max(no_kg_scores[i], with_kg_scores[i]) + 0.08),
                   ha='center', fontsize=11, color=color, fontweight='bold')
    
    ax.set_ylabel('Answer Correctness', fontsize=12)
    ax.set_title(f'KG Enhancement Effect - {category}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Improvement chart saved: {output_path}")


def create_agentic_vs_direct_improvement_chart(all_data: dict, category: str, output_path: str):
    """创建 Agentic 相对于 Direct LLM 的提升效果图"""
    # 过滤有数据的模型
    filtered_data = {}
    for model_name, model_data in all_data.items():
        if has_category_data(model_data, category):
            filtered_data[model_name] = model_data
    
    if not filtered_data:
        print(f"⚠️  没有模型有 {category} 的数据，跳过 Agentic vs Direct 提升效果图")
        return
    
    # 按 agentic_with_kg 的 correctness 从高到低排序模型
    model_scores = []
    for model_name in filtered_data.keys():
        model_data = filtered_data[model_name]
        with_kg = get_metric_value(model_data, category, 'agentic_with_kg', 'answer_correctness')
        model_scores.append((model_name, with_kg))
    model_scores.sort(key=lambda x: x[1], reverse=True)
    models = [m[0] for m in model_scores]
    mode_suffix_map = {
        'direct_llm': 'direct',
        'agentic_no_kg': 'no kg',
        'agentic_with_kg': 'with kg',
    }
    display_models = [f"{model} ({mode_suffix_map.get(mode, mode.replace('_', ' '))})" for model in models]
    
    # 创建两个子图：一个 No KG vs Direct，一个 With KG vs Direct
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 收集数据
    direct_scores = []
    no_kg_scores = []
    with_kg_scores = []
    
    for model_name in models:
        model_data = filtered_data[model_name]
        
        direct = get_metric_value(model_data, category, 'direct_llm', 'answer_correctness')
        no_kg = get_metric_value(model_data, category, 'agentic_no_kg', 'answer_correctness')
        with_kg = get_metric_value(model_data, category, 'agentic_with_kg', 'answer_correctness')
        
        direct_scores.append(direct)
        no_kg_scores.append(no_kg)
        with_kg_scores.append(with_kg)
    
    x = np.arange(len(models))
    width = 0.35
    
    # === 子图1: Agentic (No KG) vs Direct LLM ===
    ax1 = axes[0]
    improvements_nokg = [(no_kg - direct) * 100 for direct, no_kg in zip(direct_scores, no_kg_scores)]
    
    bars1 = ax1.bar(x - width/2, direct_scores, width, label='Direct LLM', color='#ADC0D1', alpha=0.85)
    bars2 = ax1.bar(x + width/2, no_kg_scores, width, label='Agentic (No KG)', color='#9BC3C6', alpha=0.85)
    
    for i, (b1, b2, imp) in enumerate(zip(bars1, bars2, improvements_nokg)):
        ax1.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.02,
               f'{direct_scores[i]:.2f}', ha='center', va='bottom', fontsize=9)
        ax1.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.02,
               f'{no_kg_scores[i]:.2f}', ha='center', va='bottom', fontsize=9)
        
        color = 'green' if imp > 0 else 'red'
        symbol = '↑' if imp > 0 else '↓'
        ax1.annotate(f'{symbol}{abs(imp):.1f}%', 
                   xy=(x[i], max(direct_scores[i], no_kg_scores[i]) + 0.08),
                   ha='center', fontsize=10, color=color, fontweight='bold')
    
    ax1.set_ylabel('Answer Correctness', fontsize=12)
    ax1.set_title('Agentic (No KG) vs Direct LLM', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9, rotation=15, ha='right')
    ax1.set_ylim(0, 1.15)
    ax1.yaxis.set_major_formatter(_Y_FMT)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # === 子图2: Agentic (With KG) vs Direct LLM ===
    ax2 = axes[1]
    improvements_withkg = [(with_kg - direct) * 100 for direct, with_kg in zip(direct_scores, with_kg_scores)]
    
    bars3 = ax2.bar(x - width/2, direct_scores, width, label='Direct LLM', color='#ADC0D1', alpha=0.85)
    bars4 = ax2.bar(x + width/2, with_kg_scores, width, label='Agentic (With KG)', color='#F4E5BE', alpha=0.85)
    
    for i, (b3, b4, imp) in enumerate(zip(bars3, bars4, improvements_withkg)):
        ax2.text(b3.get_x() + b3.get_width()/2, b3.get_height() + 0.02,
               f'{direct_scores[i]:.2f}', ha='center', va='bottom', fontsize=9)
        ax2.text(b4.get_x() + b4.get_width()/2, b4.get_height() + 0.02,
               f'{with_kg_scores[i]:.2f}', ha='center', va='bottom', fontsize=9)
        
        color = 'green' if imp > 0 else 'red'
        symbol = '↑' if imp > 0 else '↓'
        ax2.annotate(f'{symbol}{abs(imp):.1f}%', 
                   xy=(x[i], max(direct_scores[i], with_kg_scores[i]) + 0.08),
                   ha='center', fontsize=10, color=color, fontweight='bold')
    
    ax2.set_ylabel('Answer Correctness', fontsize=12)
    ax2.set_title('Agentic (With KG) vs Direct LLM', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9, rotation=15, ha='right')
    ax2.set_ylim(0, 1.15)
    ax2.yaxis.set_major_formatter(_Y_FMT)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Agentic RAG Enhancement over Direct LLM - {category}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Agentic vs Direct improvement chart saved: {output_path}")

def create_ranking_table(all_data: dict, category: str, mode: str, output_path: str):
    """创建模型排名表 - 按 correctness 排序"""
    # 过滤有数据的模型
    filtered_data = filter_models_with_data(all_data, category, mode)
    if not filtered_data:
        print(f"⚠️  没有模型有 {category}/{mode} 的数据，跳过排名表")
        return
    
    models = list(filtered_data.keys())
    
    # 计算各模型的得分
    scores = {}
    for model_name in models:
        model_data = filtered_data[model_name]
        correctness = get_metric_value(model_data, category, mode, 'answer_correctness')
        values = [get_metric_value(model_data, category, mode, m) for m in METRICS]
        valid_values = [v for v in values if v > 0]
        scores[model_name] = {
            'correctness': correctness,
            'avg': np.mean(valid_values) if valid_values else 0,
            'metrics': {m: get_metric_value(model_data, category, mode, m) for m in METRICS}
        }
    
    # 按 correctness 排序
    ranked = sorted(scores.items(), key=lambda x: x[1]['correctness'], reverse=True)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 4 + len(models) * 0.5))
    ax.axis('off')
    
    # 创建表格数据
    headers = ['Rank', 'Model', 'Correctness', 'Avg Score'] + [METRIC_LABELS.get(m, m) for m in METRICS]
    table_data = []
    
    for rank, (model_name, data) in enumerate(ranked, 1):
        row = [str(rank), model_name, f'{data["correctness"]:.2f}', f'{data["avg"]:.2f}']
        for m in METRICS:
            row.append(f'{data["metrics"][m]:.2f}')
        table_data.append(row)
    
    # 绘制表格
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#E8E8E8'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 高亮第一名
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('#B8D4C8')
    
    ax.set_title(f'Model Ranking by Correctness - {category} ({mode})', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Ranking table saved: {output_path}")


def create_ranking_table_all_modes(all_data: dict, category: str, output_path: str):
    """创建模型排名表 - 每行显示 Model + Type + Correctness，按 Correctness 排序"""
    modes = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    mode_labels = {
        'direct_llm': 'Direct LLM',
        'agentic_no_kg': 'No KG',
        'agentic_with_kg': 'With KG'
    }
    
    # 过滤有数据的模型 (使用 agentic_with_kg 作为基准)
    filtered_data = filter_models_with_data(all_data, category, 'agentic_with_kg')
    if not filtered_data:
        for mode in modes:
            filtered_data = filter_models_with_data(all_data, category, mode)
            if filtered_data:
                break
    
    if not filtered_data:
        print(f"⚠️  没有模型有 {category} 的数据，跳过排名表")
        return
    
    models = list(filtered_data.keys())
    
    # 收集所有 Model + Mode 组合的得分
    all_scores = []
    for model_name in models:
        model_data = filtered_data[model_name]
        for mode in modes:
            correctness = get_metric_value(model_data, category, mode, 'answer_correctness')
            if correctness > 0:
                all_scores.append({
                    'model': model_name,
                    'mode': mode,
                    'mode_label': mode_labels[mode],
                    'correctness': correctness
                })
    
    # 按 correctness 排序
    all_scores.sort(key=lambda x: x['correctness'], reverse=True)
    
    # 创建图表
    num_rows = len(all_scores)
    fig, ax = plt.subplots(figsize=(10, 3 + num_rows * 0.4))
    ax.axis('off')
    
    # 创建表格数据
    headers = ['Rank', 'Model', 'Type', 'Correctness']
    table_data = []
    
    for rank, item in enumerate(all_scores, 1):
        row = [str(rank), item['model'], item['mode_label'], f'{item["correctness"]:.2f}']
        table_data.append(row)
    
    # 绘制表格
    col_colors = ['#E8E8E8'] * len(headers)
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=col_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # 根据类型设置颜色
    type_colors = {
        'Direct LLM': '#ADC0D1',
        'No KG': '#9BC3C6',
        'With KG': '#F4E5BE',
    }
    
    for i, item in enumerate(all_scores):
        mode_label = item['mode_label']
        color = type_colors.get(mode_label, '#FFFFFF')
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(color)
    
    # 高亮第一名（更深的颜色）
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('#B8D4C8')
    
    ax.set_title(f'Model Ranking by Correctness (All Modes) - {category}', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Ranking table (all modes) saved: {output_path}")



# 单模型跨类别对比函数
# ============================================================================

CATEGORY_COLORS = {
    'computational': '#ADC0D1',
    'characterization': '#9BC3C6',
    'materials': '#F4E5BE',
    'processing': '#D4D2D2',
    'device': '#B8D4C8',
    'synthesis': '#E8C8B8',
}


def create_single_model_radar(model_data: dict, model_name: str, output_path: str):
    """创建单模型雷达图 (包含所有类别的合并指标，仅按模式分类)"""
    modes = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    mode_labels = {'direct_llm': 'Direct LLM', 
                   'agentic_no_kg': 'Agentic (No KG)', 
                   'agentic_with_kg': 'Agentic (With KG)'}
    
    # 汇总该模型所有类别的各项模式指标
    agg_dict = aggregate_all_categories_all_modes({model_name: model_data})
    agg_model_data = agg_dict[model_name]
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    num_metrics = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    labels = [METRIC_LABELS.get(m, m) for m in METRICS]
    
    for idx, mode in enumerate(modes):
        values = [get_metric_value(agg_model_data, 'all', mode, m) for m in METRICS]
        values += values[:1]
        
        color = MODE_COLORS.get(mode, f'C{idx}')
        # 图注上注明模型名称
        label_name = f"{model_name} ({mode_labels.get(mode, mode)})"
        
        ax.plot(angles, values, 'o-', linewidth=2, label=label_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.tick_params(axis='x', pad=15)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.20', '0.40', '0.60', '0.80', '1.00'], size=9)
    
    ax.set_title(f'{model_name} - All Modes Comparison Radar', 
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Single model radar chart saved: {output_path}")


def create_single_model_bar_chart(model_data: dict, model_name: str, mode: str, output_path: str):
    """创建单模型跨类别柱状图 - 只显示 correctness"""
    categories = list(model_data.get('categories', {}).keys())
    if not categories:
        return
    
    # 按 correctness 从高到低排序类别
    cat_scores = []
    for cat in categories:
        correctness = get_metric_value(model_data, cat, mode, 'answer_correctness')
        cat_scores.append((cat, correctness))
    cat_scores.sort(key=lambda x: x[1], reverse=True)
    
    categories = [c[0] for c in cat_scores]
    values = [c[1] for c in cat_scores]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(categories))
    
    # 创建渐变颜色 (从高分绿色到低分橙色)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(categories)))[::-1]
    
    bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # 在柱子上添加数值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Answer Correctness', fontsize=12)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_title(f'{model_name} - Correctness by Category ({mode})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加均值线
    avg = np.mean(values)
    ax.axhline(y=avg, color='#ADC0D1', linestyle='--', linewidth=2, label=f'Average: {avg:.2f}')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Single model bar chart saved: {output_path}")


def create_single_model_heatmap(model_data: dict, model_name: str, mode: str, output_path: str):
    """创建单模型热力图 - 类别 x 指标"""
    categories = list(model_data.get('categories', {}).keys())
    if not categories:
        return
    
    # 按 correctness 从高到低排序类别
    cat_scores = []
    for cat in categories:
        correctness = get_metric_value(model_data, cat, mode, 'answer_correctness')
        cat_scores.append((cat, correctness))
    cat_scores.sort(key=lambda x: x[1], reverse=True)
    categories = [c[0] for c in cat_scores]
    
    n_categories = len(categories)
    n_metrics = len(METRICS)
    
    matrix = np.zeros((n_categories, n_metrics))
    
    for i, cat in enumerate(categories):
        for j, metric in enumerate(METRICS):
            matrix[i, j] = get_metric_value(model_data, cat, mode, metric)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_categories))
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in METRICS], fontsize=11)
    ax.set_yticklabels([c.capitalize() for c in categories], fontsize=11)
    
    for i in range(n_categories):
        for j in range(n_metrics):
            val = matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)
    
    ax.set_title(f'{model_name} - Category x Metric Heatmap ({mode})', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Single model heatmap saved: {output_path}")


def create_single_model_summary_table(model_data: dict, model_name: str, mode: str, output_path: str):
    """创建单模型跨类别汇总表"""
    categories = list(model_data.get('categories', {}).keys())
    if not categories:
        return
    
    # 计算各类别的得分
    scores = {}
    for cat in categories:
        correctness = get_metric_value(model_data, cat, mode, 'answer_correctness')
        values = [get_metric_value(model_data, cat, mode, m) for m in METRICS]
        valid_values = [v for v in values if v > 0]
        scores[cat] = {
            'correctness': correctness,
            'avg': np.mean(valid_values) if valid_values else 0,
            'metrics': {m: get_metric_value(model_data, cat, mode, m) for m in METRICS}
        }
    
    # 按 correctness 排序
    ranked = sorted(scores.items(), key=lambda x: x[1]['correctness'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, 4 + len(categories) * 0.5))
    ax.axis('off')
    
    headers = ['Rank', 'Category', 'Correctness', 'Avg Score'] + [METRIC_LABELS.get(m, m) for m in METRICS]
    table_data = []
    
    for rank, (cat, data) in enumerate(ranked, 1):
        row = [str(rank), cat.capitalize(), f'{data["correctness"]:.2f}', f'{data["avg"]:.2f}']
        for m in METRICS:
            row.append(f'{data["metrics"][m]:.2f}')
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#E8E8E8'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('#B8D4C8')
    
    ax.set_title(f'{model_name} - Category Performance Summary ({mode})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Single model summary table saved: {output_path}")


def create_single_model_mode_comparison(model_data: dict, model_name: str, output_path: str):
    """创建单模型跨模式对比图"""
    categories = list(model_data.get('categories', {}).keys())
    modes = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    
    if not categories:
        return
    
    # 按 agentic_with_kg 的 correctness 排序
    cat_scores = []
    for cat in categories:
        correctness = get_metric_value(model_data, cat, 'agentic_with_kg', 'answer_correctness')
        cat_scores.append((cat, correctness))
    cat_scores.sort(key=lambda x: x[1], reverse=True)
    categories = [c[0] for c in cat_scores]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(categories))
    width = 0.25
    
    for idx, mode in enumerate(modes):
        values = [get_metric_value(model_data, cat, mode, 'answer_correctness') for cat in categories]
        
        offset = (idx - 1) * width
        color = MODE_COLORS.get(mode, f'C{idx}')
        mode_label = {'direct_llm': 'Direct LLM', 
                      'agentic_no_kg': 'Agentic (No KG)', 
                      'agentic_with_kg': 'Agentic (With KG)'}.get(mode, mode)
        bars = ax.bar(x + offset, values, width, label=mode_label, color=color, alpha=0.85)
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Answer Correctness', fontsize=12)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_title(f'{model_name} - Mode Comparison Across Categories', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Single model mode comparison saved: {output_path}")


def run_single_model_analysis(all_data: dict, model_name: str, mode: str, output_dir: Path):
    """运行单模型跨类别分析"""
    if model_name not in all_data:
        print(f"⚠️  模型 '{model_name}' 不存在")
        print(f"   可用模型: {list(all_data.keys())}")
        return
    
    model_data = all_data[model_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n生成 {model_name} 跨类别对比图表 (模式: {mode})...")
    
    create_single_model_radar(model_data, model_name, 
                              str(output_dir / 'category_radar.png'))
    
    create_single_model_bar_chart(model_data, model_name, mode,
                                  str(output_dir / 'category_bar.png'))
    
    create_single_model_heatmap(model_data, model_name, mode,
                                str(output_dir / 'category_heatmap.png'))
    
    create_single_model_summary_table(model_data, model_name, mode,
                                      str(output_dir / 'category_summary.png'))
    
    create_single_model_mode_comparison(model_data, model_name,
                                        str(output_dir / 'mode_comparison.png'))


def main():
    parser = argparse.ArgumentParser(
        description='模型对比可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 使用所有模型共同双条件QA对比 (推荐)
  python compare_models.py --common-qa
  python compare_models.py --common-qa --category characterization
  
  # 跨模型对比 (默认: characterization 类别)
  python compare_models.py --category computational
  
  # 汇总所有类别进行跨模型对比
  python compare_models.py --all
  
  # 单模型跨类别分析
  python compare_models.py --model Gpt-4o-mini
  python compare_models.py --model DeepSeek-V3.2-Thinking --mode agentic_no_kg
        '''
    )
    parser.add_argument('--category', type=str, default=None,
                       help='要对比的类别 (跨模型对比时使用)')
    parser.add_argument('--mode', type=str, default='agentic_with_kg',
                       help='要对比的模式 (默认: agentic_with_kg)')
    parser.add_argument('--all', action='store_true',
                       help='汇总所有类别进行跨模型对比')
    parser.add_argument('--model', type=str, default=None,
                       help='指定单个模型进行跨类别分析')
    parser.add_argument('--common-qa', action='store_true', dest='common_qa',
                       help='使用所有模型共同双条件QA计算均值 (需先运行 compare_reports.py)')
    parser.add_argument('--min-models', type=int, default=None, dest='min_models',
                       help='至少有多少个模型满足双条件 (默认: 5=所有模型)')
    args = parser.parse_args()
    
    # 自动启用 common_qa 模式（如果用户指定了 min_models 则隐含此意图）
    if args.min_models is not None:
        args.common_qa = True
        
    base_dir = Path('/home/wzfeng/RAGPhoto/Data_Agentic_RAG/evaluation')
    results_dir = base_dir / 'results'
    visualize_dir = results_dir / 'visualize_results'
    
    print("=" * 70)
    print("📊 模型对比可视化工具")
    print("=" * 70)
    
    # ========================
    # 共同双条件QA对比模式
    # ========================
    if args.common_qa:
        print(f"\n🎯 共同双条件QA对比模式")
        
        # 根据 min_models 参数选择加载方式
        if args.min_models is not None:
            # 使用灵活的模型数量筛选
            common_qa_ids, qa_model_count, total_models = load_dual_condition_qa_by_model_count(
                results_dir, min_models=args.min_models
            )
            
            if not common_qa_ids:
                print(f"❌ 没有QA在至少 {args.min_models} 个模型中满足双条件")
                return
            
            print(f"   找到 {len(common_qa_ids)} 个QA在至少 {args.min_models}/{total_models} 个模型中满足双条件")
            min_models_label = args.min_models
        else:
            # 使用原有方式：加载所有模型共同的QA
            common_qa_file = results_dir / 'common_qa_across_models.txt'
            common_qa_ids = load_common_qa_ids(common_qa_file)
            
            if not common_qa_ids:
                print("❌ 未找到共同双条件QA，请先运行: python compare_reports.py")
                return
            
            print(f"   加载了 {len(common_qa_ids)} 个共同双条件QA (所有模型)")
            min_models_label = 'all'
        
        # 加载QA到类别的映射
        qa_category_map = load_qa_category_map(results_dir)
        print(f"   加载了 {len(qa_category_map)} 个QA的类别映射")
        
        # 聚合指标
        aggregated_data = aggregate_metrics_from_common_qa(
            results_dir, common_qa_ids, qa_category_map, args.category
        )
        
        if not aggregated_data:
            print("❌ 无法聚合数据")
            return
        
        # 兼容 '--model' 单模型分析功能
        if args.model:
            if args.model not in aggregated_data:
                print(f"❌ '{args.model}' 没有在这些共同QA上找到数据")
                return
            
            print(f"\n🔍 单模型公共QA分析模式: {args.model}")
            output_dir = visualize_dir / args.model / f'common_qa_min{min_models_label}'
            run_single_model_analysis(aggregated_data, args.model, args.mode, output_dir)
            
            print(f"\n{'=' * 70}")
            print(f"✅ 单模型公共QA图表已保存到: {output_dir}")
            print("=" * 70)
            return
        
        # 设置输出目录
        if args.category:
            output_dir = visualize_dir / 'model_comparison' / f'common_qa_{args.category}_min{min_models_label}'
            category_label = f'Dual-Condition QA (≥{min_models_label} models) - {args.category}'
        else:
            output_dir = visualize_dir / 'model_comparison' / f'common_qa_all_min{min_models_label}'
            category_label = f'Dual-Condition QA (≥{min_models_label} models) - All Categories'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n生成共同双条件QA对比图表...")
        
        # 生成折线图
        create_multi_model_line_chart(aggregated_data, 'common_qa', 
                                      str(output_dir / 'correctness_line_chart.png'),
                                      category_label)
        
        # 生成其他图表
        create_multi_model_radar(aggregated_data, 'common_qa', args.mode, 
                                str(output_dir / 'radar_comparison.png'))
        
        create_grouped_bar_chart(aggregated_data, 'common_qa', args.mode,
                                str(output_dir / 'metrics_comparison.png'))
        
        create_heatmap(aggregated_data, 'common_qa', args.mode,
                      str(output_dir / 'heatmap.png'))
        
        create_mode_comparison_chart(aggregated_data, 'common_qa',
                                    str(output_dir / 'mode_comparison.png'))
        
        create_improvement_chart(aggregated_data, 'common_qa',
                                str(output_dir / 'kg_improvement.png'))
        
        create_agentic_vs_direct_improvement_chart(aggregated_data, 'common_qa',
                                                   str(output_dir / 'agentic_vs_direct.png'))
        
        create_ranking_table_all_modes(aggregated_data, 'common_qa',
                                       str(output_dir / 'ranking.png'))
        
        print(f"\n{'=' * 70}")
        print(f"✅ 所有图表已保存到: {output_dir}")
        print("=" * 70)
        return
    
    # ========================
    # 原有模式
    # ========================
    
    # 加载所有模型数据
    all_data = load_all_summaries(visualize_dir)
    print(f"\n加载了 {len(all_data)} 个模型的数据: {list(all_data.keys())}")
    
    # 获取可用类别
    available_categories = get_all_categories(all_data)
    print(f"可用类别: {available_categories}")
    
    # 单模型跨类别分析模式
    if args.model:
        print(f"\n🔍 单模型跨类别分析模式: {args.model}")
        output_dir = visualize_dir / args.model / 'cross_category'
        run_single_model_analysis(all_data, args.model, args.mode, output_dir)
        
        print(f"\n{'=' * 70}")
        print(f"✅ 所有图表已保存到: {output_dir}")
        print("=" * 70)
        return
    
    # 跨模型对比模式
    if args.all:
        # 汇总所有类别
        print(f"\n📊 汇总所有类别进行跨模型对比...")
        aggregated_data = aggregate_all_categories_all_modes(all_data)
        category = 'all'
        display_category = 'All Categories (Aggregated)'
        output_dir = visualize_dir / 'model_comparison' / 'all_categories'
    else:
        if args.category is None:
            args.category = 'characterization'  # 默认值
        
        if args.category not in available_categories:
            print(f"\n⚠️  类别 '{args.category}' 不存在于任何模型中")
            print(f"   可用类别: {available_categories}")
            print(f"   或使用 --all 汇总所有类别")
            return
        
        aggregated_data = all_data
        category = args.category
        display_category = args.category
        output_dir = visualize_dir / 'model_comparison' / args.category
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n生成跨模型对比图表 (类别: {display_category}, 模式: {args.mode})...")
    
    # 生成各种对比图表
    create_multi_model_radar(aggregated_data, category, args.mode, 
                            str(output_dir / 'radar_comparison.png'))
    
    create_grouped_bar_chart(aggregated_data, category, args.mode,
                            str(output_dir / 'metrics_comparison.png'))
    
    create_heatmap(aggregated_data, category, args.mode,
                  str(output_dir / 'heatmap.png'))
    
    create_mode_comparison_chart(aggregated_data, category,
                                str(output_dir / 'mode_comparison.png'))
    
    create_improvement_chart(aggregated_data, category,
                            str(output_dir / 'kg_improvement.png'))
    
    create_agentic_vs_direct_improvement_chart(aggregated_data, category,
                                               str(output_dir / 'agentic_vs_direct.png'))
    
    create_ranking_table_all_modes(aggregated_data, category,
                                   str(output_dir / 'ranking.png'))
    
    print(f"\n{'=' * 70}")
    print(f"✅ 所有图表已保存到: {output_dir}")
    if args.all:
        print(f"📊 分析范围: 所有类别汇总 ({len(available_categories)} 个类别)")
    print("=" * 70)


if __name__ == '__main__':
    main()
