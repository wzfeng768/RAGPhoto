#!/usr/bin/env python3
"""
Visualization Script for Evaluation Results
Creates:
1. Radar chart comparing 6 metrics across modes
2. Boxplot for answer_correctness distribution (with outliers)
3. Simple correctness comparison chart
4. Bar chart for all 6 metrics comparison
5. Non-zero question count line chart (real count from data)
6. JSON summary data for multi-category comparison

运行命令:
python3 visualize_results.py --dir results/gpt-4o-mini--/computational

输出目录结构:
results/visualize_results/
└── {model}/
    ├── summary_data.json       # 汇总数据(累积多个类型)
    └── {category}/
        ├── radar_chart.png
        ├── boxplot.png
        ├── correctness_simple.png
        ├── metrics_bar.png
        └── nonzero_count.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import matplotlib
import os
matplotlib.use('Agg')  # Non-interactive backend

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def extract_model_and_category(dir_path: str) -> tuple:
    """
    从路径中解析模型名和类别名
    
    Args:
        dir_path: 如 results/gpt-4o-mini--/computational
        
    Returns:
        (model, category) 元组
    """
    parts = Path(dir_path).parts
    # 找到 results 后的两级: model/category
    model = parts[-2]  # gpt-4o-mini--
    category = parts[-1]  # computational
    return model, category


def load_results(results_dir: str):
    """Load results from all modes."""
    results_path = Path(results_dir)
    data = {}
    
    for subdir in results_path.iterdir():
        if subdir.is_dir():
            mode = subdir.name
            
            # For agentic modes, use non-zero results
            if mode.startswith('agentic'):
                non_zero_file = subdir / 'results_non_zero_scores.json'
                if non_zero_file.exists():
                    with open(non_zero_file, 'r') as f:
                        data[f"{mode} (non-zero)"] = json.load(f)
                else:
                    # Fallback to main results
                    results_file = subdir / 'results.json'
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            data[mode] = json.load(f)
            
            # For direct_llm, use non-zero results if available
            elif mode == 'direct_llm':
                non_zero_file = subdir / 'results_non_zero_scores.json'
                if non_zero_file.exists():
                    with open(non_zero_file, 'r') as f:
                        data[f"{mode} (non-zero)"] = json.load(f)
                else:
                    results_file = subdir / 'results.json'
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            data[mode] = json.load(f)
    
    return data


def count_real_nonzero_questions(qa_results: list) -> int:
    """
    真实计算非零问题数量：遍历每个 QA 检查是否有任何指标为 0
    
    Args:
        qa_results: QA 结果列表
        
    Returns:
        真实非零问题的数量
    """
    metrics_to_check = ['faithfulness', 'answer_correctness', 'context_recall',
                        'context_precision', 'answer_similarity', 'answer_relevancy']
    
    nonzero_count = 0
    for qa in qa_results:
        scores = qa.get('ragas_scores', {})
        has_zero = False
        
        for metric in metrics_to_check:
            v = scores.get(metric)
            if v is not None and v == 0.0:
                has_zero = True
                break
        
        if not has_zero:
            nonzero_count += 1
    
    return nonzero_count


def save_summary_data(data: dict, model: str, category: str, model_dir: Path):
    """
    保存汇总数据到 JSON 文件，支持多类型累积
    
    Args:
        data: 当前类别的评估数据
        model: 模型名
        category: 类别名
        model_dir: 模型目录路径 (results/visualize_results/{model}/)
    """
    summary_file = model_dir / 'summary_data.json'
    
    # 读取已有数据
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    else:
        summary = {
            "model": model,
            "last_updated": "",
            "categories": {}
        }
    
    # 构建当前类别的数据
    category_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "modes": {}
    }
    
    for mode_name, mode_data in data.items():
        qa_results = mode_data.get('qa_results', [])
        ragas_metrics = mode_data.get('ragas_metrics', {})
        
        total_qas = len(qa_results)
        real_nonzero_qas = count_real_nonzero_questions(qa_results)
        nonzero_rate = (real_nonzero_qas / total_qas * 100) if total_qas > 0 else 0
        
        # 清理模式名（去掉 non-zero 后缀用于存储）
        clean_mode_name = mode_name.replace(' (non-zero)', '')
        
        mode_summary = {
            "total_qas": total_qas,
            "real_nonzero_qas": real_nonzero_qas,
            "nonzero_rate": round(nonzero_rate, 2),
            "ragas_metrics": {}
        }
        
        # 保存所有指标
        metrics_list = ['answer_correctness', 'answer_similarity', 'answer_relevancy',
                        'faithfulness', 'context_recall', 'context_precision']
        
        for metric in metrics_list:
            v = ragas_metrics.get(metric)
            if v is None or (isinstance(v, float) and v != v):  # None or NaN
                mode_summary["ragas_metrics"][metric] = None
            else:
                mode_summary["ragas_metrics"][metric] = round(v, 4)
        
        category_data["modes"][clean_mode_name] = mode_summary
    
    # 更新/添加当前类别
    summary["categories"][category] = category_data
    summary["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 写回文件
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Summary data saved: {summary_file}")


def save_detailed_data(data: dict, model: str, category: str, model_dir: Path):
    """
    保存详细版数据到 JSON 文件，包括：
    1. 有0指标的问题的详细情况
    2. 零值统计摘要（按LLM、问题类型、方法、指标）
    
    Args:
        data: 当前类别的评估数据
        model: 模型名
        category: 类别名
        model_dir: 模型目录路径 (results/visualize_results/{model}/)
    """
    detailed_file = model_dir / 'detailed_data.json'
    
    # 读取已有数据
    if detailed_file.exists():
        with open(detailed_file, 'r', encoding='utf-8') as f:
            detailed = json.load(f)
    else:
        detailed = {
            "model": model,
            "last_updated": "",
            "categories": {},
            "zero_metrics_summary": {
                "by_mode": {},
                "by_category": {},
                "by_metric": {}
            }
        }
    
    metrics_list = ['answer_correctness', 'answer_similarity', 'answer_relevancy',
                    'faithfulness', 'context_recall', 'context_precision']
    
    # 构建当前类别的详细数据
    category_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "modes": {}
    }
    
    # 更新零值统计
    if "zero_metrics_summary" not in detailed:
        detailed["zero_metrics_summary"] = {
            "by_mode": {},
            "by_category": {},
            "by_metric": {}
        }
    
    for mode_name, mode_data in data.items():
        qa_results = mode_data.get('qa_results', [])
        
        # 清理模式名（去掉 non-zero 后缀用于存储）
        clean_mode_name = mode_name.replace(' (non-zero)', '')
        
        # 收集有零值指标的问题
        questions_with_zeros = []
        mode_zero_counts = {metric: 0 for metric in metrics_list}
        
        for idx, qa in enumerate(qa_results):
            scores = qa.get('ragas_scores', {})
            zero_metrics = []
            
            for metric in metrics_list:
                v = scores.get(metric)
                if v is not None and v == 0.0:
                    zero_metrics.append(metric)
                    mode_zero_counts[metric] += 1
            
            if zero_metrics:
                question_detail = {
                    "index": idx,
                    "question": qa.get('question', 'N/A')[:200],  # 截断过长的问题
                    "zero_metrics": zero_metrics,
                    "all_scores": {
                        metric: scores.get(metric) for metric in metrics_list
                    }
                }
                # 添加答案预览（如果有）
                answer = qa.get('answer', qa.get('generated_answer', ''))
                if answer:
                    question_detail["answer_preview"] = answer[:200]
                
                questions_with_zeros.append(question_detail)
        
        total_qas = len(qa_results)
        questions_with_zero_count = len(questions_with_zeros)
        
        mode_detail = {
            "total_qas": total_qas,
            "questions_with_zero_metrics": questions_with_zero_count,
            "questions_without_zero_metrics": total_qas - questions_with_zero_count,
            "zero_count_by_metric": mode_zero_counts,
            "questions_details": questions_with_zeros
        }
        
        category_data["modes"][clean_mode_name] = mode_detail
        
        # 更新全局统计 - by_mode
        if clean_mode_name not in detailed["zero_metrics_summary"]["by_mode"]:
            detailed["zero_metrics_summary"]["by_mode"][clean_mode_name] = {
                "total_questions": 0,
                "questions_with_zeros": 0,
                "zero_counts": {metric: 0 for metric in metrics_list}
            }
        
        detailed["zero_metrics_summary"]["by_mode"][clean_mode_name]["total_questions"] += total_qas
        detailed["zero_metrics_summary"]["by_mode"][clean_mode_name]["questions_with_zeros"] += questions_with_zero_count
        for metric in metrics_list:
            detailed["zero_metrics_summary"]["by_mode"][clean_mode_name]["zero_counts"][metric] += mode_zero_counts[metric]
    
    # 更新 by_category 统计
    category_total = 0
    category_zeros = 0
    category_zero_counts = {metric: 0 for metric in metrics_list}
    
    for mode_name, mode_detail in category_data["modes"].items():
        category_total += mode_detail["total_qas"]
        category_zeros += mode_detail["questions_with_zero_metrics"]
        for metric in metrics_list:
            category_zero_counts[metric] += mode_detail["zero_count_by_metric"][metric]
    
    detailed["zero_metrics_summary"]["by_category"][category] = {
        "total_questions": category_total,
        "questions_with_zeros": category_zeros,
        "zero_counts": category_zero_counts
    }
    
    # 更新 by_metric 汇总统计
    for metric in metrics_list:
        if metric not in detailed["zero_metrics_summary"]["by_metric"]:
            detailed["zero_metrics_summary"]["by_metric"][metric] = {
                "total_zero_count": 0,
                "by_mode": {},
                "by_category": {}
            }
        
        # 累加该指标在各模式下的零值数
        for mode_name, mode_detail in category_data["modes"].items():
            if mode_name not in detailed["zero_metrics_summary"]["by_metric"][metric]["by_mode"]:
                detailed["zero_metrics_summary"]["by_metric"][metric]["by_mode"][mode_name] = 0
            detailed["zero_metrics_summary"]["by_metric"][metric]["by_mode"][mode_name] += mode_detail["zero_count_by_metric"][metric]
        
        # 累加该指标在当前类别的零值数
        detailed["zero_metrics_summary"]["by_metric"][metric]["by_category"][category] = category_zero_counts[metric]
        
        # 更新总计
        detailed["zero_metrics_summary"]["by_metric"][metric]["total_zero_count"] = sum(
            detailed["zero_metrics_summary"]["by_metric"][metric]["by_mode"].values()
        )
    
    # 更新/添加当前类别
    detailed["categories"][category] = category_data
    detailed["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 写回文件
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    
    # 打印零值摘要
    print(f"\n📊 零值指标摘要 ({model}/{category}):")
    for mode_name, mode_detail in category_data["modes"].items():
        zero_metrics_info = []
        for metric, count in mode_detail["zero_count_by_metric"].items():
            if count > 0:
                zero_metrics_info.append(f"{metric}: {count}")
        if zero_metrics_info:
            print(f"   {mode_name}: {', '.join(zero_metrics_info)}")
        else:
            print(f"   {mode_name}: 无零值指标")
    
    print(f"✅ Detailed data saved: {detailed_file}")


def create_radar_chart(data: dict, output_path: str):
    """Create radar chart comparing 6 metrics across modes."""
    
    # Metrics to plot (6 metrics)
    metrics = ['answer_correctness', 'answer_similarity', 'answer_relevancy', 
               'faithfulness', 'context_recall', 'context_precision']
    
    # Labels for display
    labels = ['Answer\nCorrectness', 'Answer\nSimilarity', 'Answer\nRelevancy',
              'Faithfulness', 'Context\nRecall', 'Context\nPrecision']
    
    # Number of metrics
    num_metrics = len(metrics)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Colors for different modes (more distinct)
    colors = ['#E63946', '#2A9D8F', '#264653']
    
    # Plot each mode
    for idx, (mode_name, mode_data) in enumerate(data.items()):
        ragas_metrics = mode_data.get('ragas_metrics', {})
        
        values = []
        for metric in metrics:
            v = ragas_metrics.get(metric)
            if v is None or (isinstance(v, float) and v != v):  # Handle None and NaN
                v = 0
            # Normalize context_precision (it can be > 1)
            if metric == 'context_precision' and v > 1:
                v = 1.0 / v  # Use reciprocal for display
            values.append(v)
        
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=mode_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    # Set labels with padding to move them outward
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    # Move labels outward to avoid overlap with radar
    ax.tick_params(axis='x', pad=15)

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)

    # Legend only (no title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Radar chart saved: {output_path}")


def create_boxplot_chart(data: dict, output_path: str):
    """Create boxplot for answer_correctness distribution across modes with outliers marked."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#E63946', '#2A9D8F', '#264653']
    
    box_data = []
    labels = []
    
    for mode_name, mode_data in data.items():
        qa_results = mode_data.get('qa_results', [])
        
        scores = []
        for qa in qa_results:
            v = qa.get('ragas_scores', {}).get('answer_correctness')
            if v is not None and not (isinstance(v, float) and v != v):
                scores.append(v)
            else:
                scores.append(0)
        
        if scores:
            box_data.append(scores)
            # Shorten label for display
            short_label = mode_name.replace(' (non-zero)', '\n(non-zero)')
            labels.append(short_label)
    
    if box_data:
        # showfliers=True 显示离群点，flierprops 设置离群点样式
        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                        showfliers=True,
                        flierprops=dict(marker='o', markerfacecolor='red', 
                                       markersize=6, markeredgecolor='darkred',
                                       alpha=0.6))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Style the medians
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # Add mean markers
        means = [np.mean(d) for d in box_data]
        ax.scatter(range(1, len(means) + 1), means, color='yellow', marker='D', 
                   s=80, zorder=5, edgecolors='black', linewidths=1, label='Mean')
        
        # Add mean value annotations
        for i, mean in enumerate(means):
            ax.annotate(f'{mean:.3f}', (i + 1, mean), textcoords="offset points",
                       xytext=(25, 0), ha='left', fontsize=10, fontweight='bold')
        
        # Count and annotate outliers for each mode
        for i, scores in enumerate(box_data):
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [s for s in scores if s < lower_bound or s > upper_bound]
            if outliers:
                ax.annotate(f'{len(outliers)} outliers', (i + 1, min(scores) - 0.05),
                           ha='center', fontsize=8, color='red')
    
    ax.set_ylabel('Answer Correctness Score', fontsize=12)
    ax.set_title('Answer Correctness Distribution by Mode\n(with Outliers)', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.15, 1.15)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Boxplot chart saved: {output_path}")


def create_simple_correctness_chart(data: dict, output_path: str):
    """Create simple line chart for answer_correctness with 3 points."""

    fig, ax = plt.subplots(figsize=(8, 6))

    # Define mode order - collect values properly
    x_labels = ['RAGPhoto_LLM', 'RAGPhoto_no_kg', 'RAGPhoto_kg']
    values = []

    # Direct LLM
    direct_value = 0
    for key in ['direct_llm (non-zero)', 'direct_llm']:
        if key in data:
            v = data[key].get('ragas_metrics', {}).get('answer_correctness')
            if v is not None and not (isinstance(v, float) and v != v):
                direct_value = v
            break
    values.append(direct_value)

    # Agentic No KG
    no_kg_value = 0
    for key in ['agentic_no_kg (non-zero)', 'agentic_no_kg']:
        if key in data:
            v = data[key].get('ragas_metrics', {}).get('answer_correctness')
            if v is not None and not (isinstance(v, float) and v != v):
                no_kg_value = v
            break
    values.append(no_kg_value)

    # Agentic With KG
    with_kg_value = 0
    for key in ['agentic_with_kg (non-zero)', 'agentic_with_kg']:
        if key in data:
            v = data[key].get('ragas_metrics', {}).get('answer_correctness')
            if v is not None and not (isinstance(v, float) and v != v):
                with_kg_value = v
            break
    values.append(with_kg_value)

    x = range(len(x_labels))

    # Plot line with markers
    colors = ['#E63946', '#2A9D8F', '#264653']
    ax.plot(x, values, marker='o', markersize=12, linewidth=2.5, color='#457B9D')

    # Add scatter points with different colors
    for i, (xi, vi) in enumerate(zip(x, values)):
        ax.scatter(xi, vi, s=150, c=colors[i], zorder=5, edgecolors='white', linewidths=2)
        ax.annotate(f'{vi:.4f}', (xi, vi), textcoords="offset points",
                   xytext=(0, 15), ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylabel('Answer Correctness', fontsize=12)
    ax.set_title('Answer Correctness Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Simple correctness chart saved: {output_path}")


def create_bar_chart(data: dict, output_path: str):
    """Create bar chart comparing all 6 metrics in a single plot."""

    # All 6 metrics
    metrics = ['answer_correctness', 'answer_similarity', 'answer_relevancy',
               'faithfulness', 'context_recall', 'context_precision']

    labels = ['Correctness', 'Similarity', 'Relevancy', 'Faithfulness', 'Recall', 'Precision']

    # Create single figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(metrics))
    width = 0.25

    colors = ['#E63946', '#2A9D8F', '#264653']

    # Plot all metrics
    for idx, (mode_name, mode_data) in enumerate(data.items()):
        ragas_metrics = mode_data.get('ragas_metrics', {})

        values = []
        for metric in metrics:
            v = ragas_metrics.get(metric)
            if v is None or (isinstance(v, float) and v != v):
                v = 0
            # Cap context_precision at 1 for display
            if metric == 'context_precision' and v > 1:
                v = min(v, 1.0)
            values.append(v)

        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width, label=mode_name, color=colors[idx % len(colors)])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Bar chart saved: {output_path}")


def create_nonzero_count_chart(data: dict, output_path: str):
    """Create line chart showing REAL non-zero question counts across modes."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_labels = ['RAGPhoto_LLM', 'RAGPhoto_no_kg', 'RAGPhoto_kg']
    counts = []
    total_counts = []

    # Direct LLM - 真实计算非零问题数
    for key in ['direct_llm (non-zero)', 'direct_llm']:
        if key in data:
            qa_results = data[key].get('qa_results', [])
            real_nonzero = count_real_nonzero_questions(qa_results)
            counts.append(real_nonzero)
            total_counts.append(len(qa_results))
            break
    else:
        counts.append(0)
        total_counts.append(0)
    
    # Agentic No KG - 真实计算非零问题数
    for key in ['agentic_no_kg (non-zero)', 'agentic_no_kg']:
        if key in data:
            qa_results = data[key].get('qa_results', [])
            real_nonzero = count_real_nonzero_questions(qa_results)
            counts.append(real_nonzero)
            total_counts.append(len(qa_results))
            break
    else:
        counts.append(0)
        total_counts.append(0)
    
    # Agentic With KG - 真实计算非零问题数
    for key in ['agentic_with_kg (non-zero)', 'agentic_with_kg']:
        if key in data:
            qa_results = data[key].get('qa_results', [])
            real_nonzero = count_real_nonzero_questions(qa_results)
            counts.append(real_nonzero)
            total_counts.append(len(qa_results))
            break
    else:
        counts.append(0)
        total_counts.append(0)
    
    x = range(len(x_labels))
    colors = ['#E63946', '#2A9D8F', '#264653']
    
    # Plot line
    ax.plot(x, counts, marker='o', markersize=12, linewidth=2.5, color='#457B9D')
    
    # Add scatter points with different colors
    for i, (xi, ci, ti) in enumerate(zip(x, counts, total_counts)):
        ax.scatter(xi, ci, s=150, c=colors[i], zorder=5, edgecolors='white', linewidths=2)
        # 显示 真实非零数 / 总数
        ax.annotate(f'{ci}/{ti}', (xi, ci), textcoords="offset points",
                   xytext=(0, 15), ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel('Real Non-Zero Question Count', fontsize=12)
    ax.set_title('Real Non-Zero Questions by Mode\n(Questions without any zero-score metrics)', 
                 fontsize=14, fontweight='bold')
    
    # Set y-axis to start from 0
    max_count = max(total_counts) if total_counts else 10
    ax.set_ylim(0, max_count * 1.2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Non-zero count chart saved: {output_path}")


def create_cross_category_charts(summary_path: str, output_dir: Path):
    """
    Generate cross-category comparison charts from summary_data.json
    
    Args:
        summary_path: Path to summary_data.json
        output_dir: Directory to save charts
    """
    if not os.path.exists(summary_path):
        print(f"❌ Summary file not found: {summary_path}")
        return

    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = data.get('categories', {})
    if not categories:
        print("❌ No category data found in summary file")
        return

    # Prepare data structure for plotting
    category_names = sorted(categories.keys())
    modes = set()
    for cat_data in categories.values():
        modes.update(cat_data.get('modes', {}).keys())
    
    modes = sorted(list(modes))
    # Ensure specific order if possible
    mode_order = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    sorted_modes = [m for m in mode_order if m in modes] + [m for m in modes if m not in mode_order]
    
    colors = ['#E63946', '#2A9D8F', '#264653']  # Match previous colors

    # 1. Answer Correctness Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(category_names))
    width = 0.8 / len(sorted_modes)
    
    for i, mode in enumerate(sorted_modes):
        values = []
        for cat in category_names:
            mode_data = categories[cat]['modes'].get(mode, {})
            # Use 'ragas_metrics' -> 'answer_correctness'
            val = mode_data.get('ragas_metrics', {}).get('answer_correctness', 0)
            if val is None: val = 0
            values.append(val)
        
        offset = (i - len(sorted_modes)/2 + 0.5) * width
        rects = ax.bar(x + offset, values, width, label=mode, color=colors[i % len(colors)])
        
        # Add value labels
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylabel('Answer Correctness Score')
    ax.set_title(f'Answer Correctness Across Categories ({data.get("model", "Unknown")})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(category_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    output_path = output_dir / 'cross_category_correctness.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Cross-category correctness chart saved: {output_path}")

    # 2. Non-Zero Rate Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, mode in enumerate(sorted_modes):
        values = []
        for cat in category_names:
            mode_data = categories[cat]['modes'].get(mode, {})
            val = mode_data.get('nonzero_rate', 0)
            values.append(val)
        
        offset = (i - len(sorted_modes)/2 + 0.5) * width
        rects = ax.bar(x + offset, values, width, label=mode, color=colors[i % len(colors)])
        
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylabel('Non-Zero Rate (%)')
    ax.set_title(f'Non-Zero Answer Rate Across Categories ({data.get("model", "Unknown")})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(category_names, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 115)  # Make room for labels

    output_path = output_dir / 'cross_category_nonzero_rate.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Cross-category non-zero rate chart saved: {output_path}")


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--dir', default='results/gpt-4o-mini--/computational', 
                        help='Directory containing results OR model directory for summary charts')
    parser.add_argument('--summary-charts', action='store_true',
                       help='Generate cross-category summary charts (requires --dir to be model directory)')
    args = parser.parse_args()
    
    print("="*60)
    print("Visualizing Evaluation Results")
    print("="*60)
    
    if args.summary_charts:
        # Cross-category summary mode
        model_dir = Path(args.dir)
        summary_file = model_dir / 'summary_data.json'
        
        print(f"Mode: Cross-Category Summary")
        print(f"Model Directory: {model_dir}")
        print(f"Summary File: {summary_file}")
        
        create_cross_category_charts(str(summary_file), model_dir)
        print("="*60)
        return

    # Extract model and category from path
    model, category = extract_model_and_category(args.dir)
    print(f"\nModel: {model}")
    print(f"Category: {category}")
    
    # Load data
    data = load_results(args.dir)
    print(f"Loaded {len(data)} modes: {list(data.keys())}")
    
    # Output directory structure: results/visualize_results/{model}/{category}/
    base_output_dir = Path('results/visualize_results')
    model_dir = base_output_dir / model
    category_dir = model_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # Create charts (no prefix needed, in category subfolder)
    create_radar_chart(data, str(category_dir / "radar_chart.png"))
    create_boxplot_chart(data, str(category_dir / "boxplot.png"))
    create_simple_correctness_chart(data, str(category_dir / "correctness_simple.png"))
    create_bar_chart(data, str(category_dir / "metrics_bar.png"))
    create_nonzero_count_chart(data, str(category_dir / "nonzero_count.png"))
    
    # Save summary data to model folder (accumulates across categories)
    save_summary_data(data, model, category, model_dir)
    
    # Save detailed data with zero metrics information
    save_detailed_data(data, model, category, model_dir)
    
    print(f"\n{'='*60}")
    print(f"Done! Charts saved to: {category_dir}")
    print(f"Summary data saved to: {model_dir / 'summary_data.json'}")
    print(f"Detailed data saved to: {model_dir / 'detailed_data.json'}")
    print("="*60)


if __name__ == '__main__':
    main()
