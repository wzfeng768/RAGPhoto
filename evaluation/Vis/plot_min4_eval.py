#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.transforms import Bbox

_Y_FMT = mticker.FormatStrFormatter('%.2f')

plt.style.use('seaborn-v0_8-whitegrid')

# Set matplotlib font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#D0D7DE'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#DCE3EA'
plt.rcParams['grid.alpha'] = 0.65
plt.rcParams['grid.linewidth'] = 0.9
plt.rcParams['legend.frameon'] = False

# Constants
METRICS = [
    'answer_correctness', 'answer_similarity', 'answer_relevancy',
    'faithfulness', 'context_recall', 'context_precision'
]
METRIC_LABELS = {
    'answer_correctness': 'Correctness',
    'answer_similarity': 'Similarity',
    'answer_relevancy': 'Relevancy',
    'faithfulness': 'Faithfulness',
    'context_recall': 'Recall',
    'context_precision': 'Precision'
}
MODEL_COLORS = {
    'Gpt-4o-mini': '#ADC0D1',
    'Qwen3-235B': '#9BC3C6',
    'DeepSeek-V3.2-Thinking': '#F4E5BE',
    'GLM-4.7': '#D4D2D2',
    'MiniMax-M2.1': '#B8D4C8',
    'Llama-3.3-70B': '#E8C8B8',
}
# Deeper/vivid variant used exclusively for thin-line charts
# (correctness_simple, category/difficulty/reasoning breakdown)
MODEL_COLORS_LINE = {
    'Gpt-4o-mini': '#5B8FB9',
    'Qwen3-235B': '#2E8B8B',
    'DeepSeek-V3.2-Thinking': '#EC9B6C',
    'GLM-4.7': '#8C8C8C',
    'MiniMax-M2.1': '#4DA680',
    'Llama-3.3-70B': '#C75B3F',
}
MODE_COLORS = {
    'direct_llm': '#ADC0D1',
    'agentic_no_kg': '#9BC3C6',
    'agentic_with_kg': '#F4E5BE',
}
MODE_DISPLAY_NAMES = {
    'direct_llm': 'Direct',
    'agentic_no_kg': 'No KG',
    'agentic_with_kg': 'With KG',
}
MODE_SHORT_NAMES = {
    'direct_llm': 'direct',
    'agentic_no_kg': 'no_kg',
    'agentic_with_kg': 'with_kg',
}
MODE_ALIASES = {
    'direct': 'direct_llm',
    'direct_llm': 'direct_llm',
    'no_kg': 'agentic_no_kg',
    'agentic_no_kg': 'agentic_no_kg',
    'with_kg': 'agentic_with_kg',
    'agentic_with_kg': 'agentic_with_kg',
}
BREAKDOWN_DISPLAY_NAMES = {
    'category': 'Category',
    'difficulty': 'Difficulty',
    'reasoning': 'Reasoning Type',
}
BREAKDOWN_ALIASES = {
    'category': 'category',
    'categories': 'category',
    'difficulty': 'difficulty',
    'difficulties': 'difficulty',
    'reasoning': 'reasoning',
    'reasonings': 'reasoning',
}
ANALYSIS_METRIC_LABELS = {
    'avg_correctness': 'Answer Correctness',
    'correct_rate': 'Correct Rate',
}
NON_MODEL_DIRS = {'visualize', 'visualize_results', 'model_comparison', 'eval_visualize'}
CATEGORY_MAP = {
    'materials': 'Material design',
    'processing': 'Material design',
    'device': 'Device engineering',
    'performance': 'Device engineering',
    'stability': 'Device engineering',
    'characterization': 'Characterization',
    'computational': 'Mechanism discovery',
    'structure': 'Mechanism discovery',
}
CATEGORY_ORDER = ['Material design', 'Device engineering', 'Characterization', 'Mechanism discovery']
DIFFICULTY_ORDER = ['easy', 'medium', 'hard']
REASONING_ORDER = ['single-hop', 'multi-hop', 'cross-doc']
SERIES_MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h']
OVERVIEW_FIGSIZE = (12, 6)
CORRECTNESS_FIGSIZE = (12, 6)
RADAR_FIGSIZE = OVERVIEW_FIGSIZE
OVERVIEW_EXPORT_SIZE = (11.88, 5.933333333333334)
CORRECTNESS_EXPORT_SIZE = (9.88, 5.933333333333334)
EXPORT_DPI = 300
SIMPLE_CORRECTNESS_STYLE = {
    'ylabel_fontsize': 17,
    'xtick_fontsize': 15,
    'ytick_fontsize': 14,
    'legend_fontsize': 14,
    'line_width': 2.6,
    'marker_size': 11.0,
    'scatter_size': 120,
    'legend_anchor_y': 1.16,
    'offset_max_span': 0.24,
}
BREAKDOWN_CHART_STYLE = {
    'ylabel_fontsize': 17,
    'xtick_fontsize': 14,
    'ytick_fontsize': 14,
    'legend_fontsize': 14,
    'line_width': 2.6,
    'marker_size': 10.5,
    'scatter_size': 115,
    'annotation_fontsize': 12,
    'legend_anchor_y': 1.15,
    'offset_max_span': 0.26,
}
METRICS_BAR_STYLE = {
    'ylabel_fontsize': 15,
    'xtick_fontsize': 13,
    'ytick_fontsize': 13,
    'legend_fontsize': 13,
    'value_fontsize': 10,
    'legend_anchor_y': 1.14,
}
AGENTIC_VS_DIRECT_STYLE = {
    'ylabel_fontsize': 15,
    'xtick_fontsize': 12,
    'ytick_fontsize': 13,
    'legend_fontsize': 13,
    'value_fontsize': 12,
    'improvement_fontsize': 13,
    'legend_anchor_y': 1.14,
}


def get_metric_value(model_data, category, mode, metric):
    return model_data.get('categories', {}).get(category, {}).get('modes', {}).get(mode, {}).get('ragas_metrics', {}).get(metric, 0.0)


def safe_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric):
        return None
    return numeric


def slugify(text: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', str(text).strip()).strip('_').lower()
    return slug or 'item'


def wrap_label(text: str, width: int = 18) -> str:
    wrapped = textwrap.wrap(str(text), width=width)
    return '\n'.join(wrapped) if wrapped else str(text)


def get_series_offsets(series_count: int, max_span: float) -> np.ndarray:
    if series_count <= 1:
        return np.array([0.0])

    span = min(max_span, 0.12 + 0.025 * (series_count - 1))
    return np.linspace(-span, span, series_count)


def save_overview_figure(fig, output_path, dpi: int = EXPORT_DPI, target_size=None) -> None:
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    target_width, target_height = target_size or OVERVIEW_EXPORT_SIZE

    if bbox.width < target_width:
        delta_x = (target_width - bbox.width) / 2
        bbox = Bbox.from_extents(bbox.x0 - delta_x, bbox.y0, bbox.x1 + delta_x, bbox.y1)

    if bbox.height < target_height:
        delta_y = (target_height - bbox.height) / 2
        bbox = Bbox.from_extents(bbox.x0, bbox.y0 - delta_y, bbox.x1, bbox.y1 + delta_y)

    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox)


def get_display_mode_name(mode: str) -> str:
    return MODE_DISPLAY_NAMES.get(mode, mode)


def get_mode_legend_suffix(mode: str) -> str:
    suffix_map = {
        'direct_llm': 'direct',
        'agentic_no_kg': 'no kg',
        'agentic_with_kg': 'with kg',
    }
    return suffix_map.get(mode, str(mode).replace('_', ' '))


def get_model_dirs(min4_dir: Path):
    for model_dir in sorted(min4_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in NON_MODEL_DIRS:
            continue
        yield model_dir


def resolve_models(requested_models, available_models):
    available_models = sorted(available_models)
    if not requested_models or any(model.lower() == 'all' for model in requested_models):
        return available_models

    model_lookup = {model.lower(): model for model in available_models}
    resolved_models = []
    unknown_models = []

    for model in requested_models:
        matched_model = model_lookup.get(model.lower())
        if matched_model is None:
            unknown_models.append(model)
            continue
        if matched_model not in resolved_models:
            resolved_models.append(matched_model)

    if unknown_models:
        available_text = ', '.join(available_models)
        raise ValueError(f"Unknown model(s): {', '.join(unknown_models)}. Available models: {available_text}")

    return resolved_models


def resolve_methods(requested_methods):
    default_methods = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    if not requested_methods or any(method.lower() == 'all' for method in requested_methods):
        return default_methods

    resolved_methods = []
    unknown_methods = []
    for method in requested_methods:
        normalized = MODE_ALIASES.get(method.lower())
        if normalized is None:
            unknown_methods.append(method)
            continue
        if normalized not in resolved_methods:
            resolved_methods.append(normalized)

    if unknown_methods:
        raise ValueError("Unknown method(s): " + ', '.join(unknown_methods) + ". Use: direct, no_kg, with_kg")

    return resolved_methods


def resolve_breakdowns(requested_breakdowns):
    default_breakdowns = ['category', 'difficulty', 'reasoning']
    if not requested_breakdowns or any(item.lower() == 'all' for item in requested_breakdowns):
        return default_breakdowns

    resolved_breakdowns = []
    unknown_breakdowns = []
    for item in requested_breakdowns:
        normalized = BREAKDOWN_ALIASES.get(item.lower())
        if normalized is None:
            unknown_breakdowns.append(item)
            continue
        if normalized not in resolved_breakdowns:
            resolved_breakdowns.append(normalized)

    if unknown_breakdowns:
        raise ValueError(
            "Unknown breakdown(s): " + ', '.join(unknown_breakdowns) + ". Use: category, difficulty, reasoning"
        )

    return resolved_breakdowns


def load_min4_data(min4_dir: Path) -> dict:
    all_data = {}
    for model_dir in get_model_dirs(min4_dir):
        model_name = model_dir.name
        model_categories = {}

        for category_dir in sorted(model_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            mapped_cat = CATEGORY_MAP.get(category_dir.name.lower(), category_dir.name)
            if mapped_cat not in model_categories:
                model_categories[mapped_cat] = {'modes': {}}

            for mode_dir in sorted(category_dir.iterdir()):
                if not mode_dir.is_dir():
                    continue
                mode_name = mode_dir.name
                results_file = mode_dir / 'results.json'

                if not results_file.exists():
                    continue

                try:
                    with open(results_file, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                except Exception:
                    continue

                summary = data.get('summary', {})
                ragas_metrics = summary.get('ragas_metrics', {})
                num_questions = summary.get('num_questions_completed', 0)
                if num_questions <= 0 or not ragas_metrics:
                    continue

                weighted_metrics = {}
                for metric_name, metric_value in ragas_metrics.items():
                    numeric_value = safe_float(metric_value)
                    if numeric_value is not None:
                        weighted_metrics[metric_name] = numeric_value * num_questions
                if not weighted_metrics:
                    continue

                modes_dict = model_categories[mapped_cat]['modes']
                if mode_name not in modes_dict:
                    modes_dict[mode_name] = {
                        'ragas_metrics': weighted_metrics,
                        'num_questions': num_questions,
                    }
                else:
                    for metric_name, metric_value in weighted_metrics.items():
                        modes_dict[mode_name]['ragas_metrics'][metric_name] = (
                            modes_dict[mode_name]['ragas_metrics'].get(metric_name, 0.0) + metric_value
                        )
                    modes_dict[mode_name]['num_questions'] += num_questions

        for category_data in model_categories.values():
            for mode_data in category_data['modes'].values():
                total_questions = mode_data['num_questions']
                if total_questions <= 0:
                    continue
                for metric_name in list(mode_data['ragas_metrics'].keys()):
                    mode_data['ragas_metrics'][metric_name] /= total_questions

        all_data[model_name] = {'categories': model_categories}

    return all_data


def load_min4_question_records(min4_dir: Path):
    records = []
    for model_dir in get_model_dirs(min4_dir):
        model_name = model_dir.name
        for category_dir in sorted(model_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            mapped_category = CATEGORY_MAP.get(category_dir.name.lower(), category_dir.name)

            for mode_dir in sorted(category_dir.iterdir()):
                if not mode_dir.is_dir():
                    continue
                mode_name = mode_dir.name
                results_file = mode_dir / 'results.json'
                if not results_file.exists():
                    continue

                try:
                    with open(results_file, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                except Exception:
                    continue

                for qa_result in data.get('qa_results', []):
                    score = safe_float(qa_result.get('ragas_scores', {}).get('answer_correctness'))
                    if score is None:
                        continue

                    difficulty = str(qa_result.get('difficulty', 'unknown')).strip().lower() or 'unknown'
                    reasoning = str(qa_result.get('reasoning_type', 'unknown')).strip().lower() or 'unknown'
                    records.append({
                        'model': model_name,
                        'mode': mode_name,
                        'category': mapped_category,
                        'difficulty': difficulty,
                        'reasoning': reasoning,
                        'answer_correctness': score,
                        'question_id': qa_result.get('question_id'),
                    })

    return records


def aggregate_all_categories(all_data: dict) -> dict:
    aggregated = {}
    modes_list = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']

    for model_name, model_data in all_data.items():
        categories = model_data.get('categories', {})
        mode_aggregated = {}

        for mode in modes_list:
            metric_sums = {metric: 0.0 for metric in METRICS}
            metric_counts = {metric: 0 for metric in METRICS}

            for category_data in categories.values():
                mode_data = category_data.get('modes', {}).get(mode, {})
                metrics = mode_data.get('ragas_metrics', {})
                num_questions = mode_data.get('num_questions', 0)

                if num_questions <= 0:
                    continue

                for metric in METRICS:
                    value = safe_float(metrics.get(metric))
                    if value is not None:
                        metric_sums[metric] += value * num_questions
                        metric_counts[metric] += num_questions

            avg_metrics = {
                metric: (metric_sums[metric] / metric_counts[metric] if metric_counts[metric] > 0 else 0.0)
                for metric in METRICS
            }
            mode_aggregated[mode] = {
                'ragas_metrics': avg_metrics,
                'num_questions': sum(metric_counts.values()) // max(1, len(METRICS)),
            }

        aggregated[model_name] = {'categories': {'all': {'modes': mode_aggregated}}}

    return aggregated


def create_grouped_bar_chart(all_data, category, mode, output_path, show_values=True):
    models = sorted(all_data.keys(), key=lambda model: get_metric_value(all_data[model], category, mode, 'answer_correctness'), reverse=True)
    if not models:
        return

    fig, ax = plt.subplots(figsize=OVERVIEW_FIGSIZE)
    n_models = len(models)
    x = np.arange(len(METRICS))
    width = 0.8 / n_models

    for idx, model_name in enumerate(models):
        values = [get_metric_value(all_data[model_name], category, mode, metric) for metric in METRICS]
        offset = (idx - n_models / 2 + 0.5) * width
        color = MODEL_COLORS.get(model_name, f'C{idx}')
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=f"{model_name} ({get_display_mode_name(mode)})",
            color=color,
            alpha=0.88,
            edgecolor='white',
            linewidth=0.8,
        )

        if show_values:
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f'{value:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=METRICS_BAR_STYLE['value_fontsize'],
                        rotation=45,
                    )

    ax.set_ylabel('Score', fontsize=METRICS_BAR_STYLE['ylabel_fontsize'])
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(metric, metric) for metric in METRICS], fontsize=METRICS_BAR_STYLE['xtick_fontsize'])
    ax.tick_params(axis='y', labelsize=METRICS_BAR_STYLE['ytick_fontsize'])
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, METRICS_BAR_STYLE['legend_anchor_y']),
        ncol=min(3, len(models)),
        fontsize=METRICS_BAR_STYLE['legend_fontsize'],
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.5,
        borderaxespad=0,
    )
    ax.grid(True, alpha=0.45, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_overview_figure(fig, output_path)
    plt.close()


def create_improvement_charts(all_data, category, out_dir):
    models = sorted(all_data.keys(), key=lambda model: get_metric_value(all_data[model], category, 'agentic_with_kg', 'answer_correctness'), reverse=True)
    if not models:
        return

    # Print original colors before overriding
    print(f'[kg_improvement] original colors: No KG={MODE_COLORS["agentic_no_kg"]}  With KG={MODE_COLORS["agentic_with_kg"]}')

    # Use same palette as agentic_no_kg_vs_direct charts
    KG_COLOR_NO_KG = '#9BC3C6'
    KG_COLOR_WITH_KG = '#F4E5BE'

    no_kg_scores = []
    with_kg_scores = []
    improvements = []
    for model_name in models:
        no_kg_score = get_metric_value(all_data[model_name], category, 'agentic_no_kg', 'answer_correctness')
        with_kg_score = get_metric_value(all_data[model_name], category, 'agentic_with_kg', 'answer_correctness')
        no_kg_scores.append(no_kg_score)
        with_kg_scores.append(with_kg_score)
        improvements.append((with_kg_score - no_kg_score) * 100)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=OVERVIEW_FIGSIZE)
    bars1 = ax.bar(x - width / 2, no_kg_scores, width, label='Agentic (No KG)', color=KG_COLOR_NO_KG, alpha=0.85)
    bars2 = ax.bar(x + width / 2, with_kg_scores, width, label='Agentic (With KG)', color=KG_COLOR_WITH_KG, alpha=0.85)
    for i, (bar1, bar2, improvement) in enumerate(zip(bars1, bars2, improvements)):
        ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() + 0.01, f'{no_kg_scores[i]:.2f}', ha='center', va='bottom', fontsize=10)
        ax.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height() + 0.01, f'{with_kg_scores[i]:.2f}', ha='center', va='bottom', fontsize=10)
        symbol = '↑' if improvement > 0 else '↓'
        color = 'green' if improvement > 0 else 'red'
        ax.annotate(f'{symbol}{abs(improvement):.1f}%', xy=(x[i], max(no_kg_scores[i], with_kg_scores[i]) + 0.04), ha='center', fontsize=11, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_overview_figure(fig, out_dir / 'kg_improvement_annotated.png')
    plt.close()

    fig, ax = plt.subplots(figsize=OVERVIEW_FIGSIZE)
    ax.bar(x - width / 2, no_kg_scores, width, label='Agentic (No KG)', color=KG_COLOR_NO_KG, alpha=0.85)
    ax.bar(x + width / 2, with_kg_scores, width, label='Agentic (With KG)', color=KG_COLOR_WITH_KG, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_overview_figure(fig, out_dir / 'kg_improvement.png')
    plt.close()


def create_agentic_vs_direct_charts(all_data, category, out_dir):
    models = sorted(all_data.keys(), key=lambda model: get_metric_value(all_data[model], category, 'agentic_with_kg', 'answer_correctness'), reverse=True)
    if not models:
        return

    x = np.arange(len(models))
    width = 0.35
    direct = [get_metric_value(all_data[model], category, 'direct_llm', 'answer_correctness') for model in models]
    no_kg = [get_metric_value(all_data[model], category, 'agentic_no_kg', 'answer_correctness') for model in models]
    with_kg = [get_metric_value(all_data[model], category, 'agentic_with_kg', 'answer_correctness') for model in models]

    fig, ax = plt.subplots(figsize=OVERVIEW_FIGSIZE)
    bars1 = ax.bar(x - width / 2, direct, width, label='Direct LLM', color='#F4E5BE', alpha=0.85)
    bars2 = ax.bar(x + width / 2, no_kg, width, label='Agentic (No KG)', color='#9BC3C6', alpha=0.85)

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        improvement = (no_kg[i] - direct[i]) * 100
        ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() + 0.01, f'{direct[i]:.2f}', ha='center', va='bottom', fontsize=AGENTIC_VS_DIRECT_STYLE['value_fontsize'])
        ax.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height() + 0.01, f'{no_kg[i]:.2f}', ha='center', va='bottom', fontsize=AGENTIC_VS_DIRECT_STYLE['value_fontsize'])
        symbol = '↑' if improvement > 0 else '↓'
        color = 'green' if improvement > 0 else 'red'
        ax.annotate(f'{symbol}{abs(improvement):.1f}%', xy=(x[i], max(direct[i], no_kg[i]) + 0.04), ha='center', fontsize=AGENTIC_VS_DIRECT_STYLE['improvement_fontsize'], color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=AGENTIC_VS_DIRECT_STYLE['xtick_fontsize'])
    ax.set_ylabel('Answer Correctness', fontsize=AGENTIC_VS_DIRECT_STYLE['ylabel_fontsize'])
    ax.tick_params(axis='y', labelsize=AGENTIC_VS_DIRECT_STYLE['ytick_fontsize'])
    ax.set_ylim(0.2, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, AGENTIC_VS_DIRECT_STYLE['legend_anchor_y']), ncol=2, fontsize=AGENTIC_VS_DIRECT_STYLE['legend_fontsize'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_overview_figure(fig, out_dir / 'agentic_no_kg_vs_direct_annotated.png')
    plt.close()

    fig, ax = plt.subplots(figsize=OVERVIEW_FIGSIZE)
    bars1 = ax.bar(x - width / 2, direct, width, label='Direct LLM', color='#F4E5BE', alpha=0.85)
    bars2 = ax.bar(x + width / 2, no_kg, width, label='Agentic (No KG)', color='#9BC3C6', alpha=0.85)

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() + 0.01, f'{direct[i]:.2f}', ha='center', va='bottom', fontsize=AGENTIC_VS_DIRECT_STYLE['value_fontsize'])
        ax.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height() + 0.01, f'{no_kg[i]:.2f}', ha='center', va='bottom', fontsize=AGENTIC_VS_DIRECT_STYLE['value_fontsize'])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=AGENTIC_VS_DIRECT_STYLE['xtick_fontsize'])
    ax.set_ylabel('Answer Correctness', fontsize=AGENTIC_VS_DIRECT_STYLE['ylabel_fontsize'])
    ax.tick_params(axis='y', labelsize=AGENTIC_VS_DIRECT_STYLE['ytick_fontsize'])
    ax.set_ylim(0.2, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, AGENTIC_VS_DIRECT_STYLE['legend_anchor_y']), ncol=2, fontsize=AGENTIC_VS_DIRECT_STYLE['legend_fontsize'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_overview_figure(fig, out_dir / 'agentic_no_kg_vs_direct.png')
    plt.close()

    fig, ax = plt.subplots(figsize=OVERVIEW_FIGSIZE)
    bars1 = ax.bar(x - width / 2, direct, width, label='Direct LLM', color='#F4E5BE', alpha=0.85)
    bars2 = ax.bar(x + width / 2, with_kg, width, label='Agentic (With KG)', color='#9BC3C6', alpha=0.85)

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        improvement = (with_kg[i] - direct[i]) * 100
        ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() + 0.01, f'{direct[i]:.2f}', ha='center', va='bottom', fontsize=AGENTIC_VS_DIRECT_STYLE['value_fontsize'])
        ax.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height() + 0.01, f'{with_kg[i]:.2f}', ha='center', va='bottom', fontsize=AGENTIC_VS_DIRECT_STYLE['value_fontsize'])
        symbol = '↑' if improvement > 0 else '↓'
        color = 'green' if improvement > 0 else 'red'
        ax.annotate(f'{symbol}{abs(improvement):.1f}%', xy=(x[i], max(direct[i], with_kg[i]) + 0.04), ha='center', fontsize=AGENTIC_VS_DIRECT_STYLE['improvement_fontsize'], color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=AGENTIC_VS_DIRECT_STYLE['xtick_fontsize'])
    ax.set_ylabel('Answer Correctness', fontsize=AGENTIC_VS_DIRECT_STYLE['ylabel_fontsize'])
    ax.tick_params(axis='y', labelsize=AGENTIC_VS_DIRECT_STYLE['ytick_fontsize'])
    ax.set_ylim(0.2, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, AGENTIC_VS_DIRECT_STYLE['legend_anchor_y']), ncol=2, fontsize=AGENTIC_VS_DIRECT_STYLE['legend_fontsize'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_overview_figure(fig, out_dir / 'agentic_with_kg_vs_direct_annotated.png')
    plt.close()

    fig, ax = plt.subplots(figsize=OVERVIEW_FIGSIZE)
    bars1 = ax.bar(x - width / 2, direct, width, label='Direct LLM', color='#F4E5BE', alpha=0.85)
    bars2 = ax.bar(x + width / 2, with_kg, width, label='Agentic (With KG)', color='#9BC3C6', alpha=0.85)

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() + 0.01, f'{direct[i]:.2f}', ha='center', va='bottom', fontsize=AGENTIC_VS_DIRECT_STYLE['value_fontsize'])
        ax.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height() + 0.01, f'{with_kg[i]:.2f}', ha='center', va='bottom', fontsize=AGENTIC_VS_DIRECT_STYLE['value_fontsize'])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=AGENTIC_VS_DIRECT_STYLE['xtick_fontsize'])
    ax.set_ylabel('Answer Correctness', fontsize=AGENTIC_VS_DIRECT_STYLE['ylabel_fontsize'])
    ax.tick_params(axis='y', labelsize=AGENTIC_VS_DIRECT_STYLE['ytick_fontsize'])
    ax.set_ylim(0.2, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, AGENTIC_VS_DIRECT_STYLE['legend_anchor_y']), ncol=2, fontsize=AGENTIC_VS_DIRECT_STYLE['legend_fontsize'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_overview_figure(fig, out_dir / 'agentic_with_kg_vs_direct.png')
    plt.close()


def create_simple_correctness_chart(all_data, category, output_path):
    models = sorted(all_data.keys(), key=lambda model: get_metric_value(all_data[model], category, 'agentic_with_kg', 'answer_correctness'), reverse=True)
    if not models:
        return

    fig, ax = plt.subplots(figsize=CORRECTNESS_FIGSIZE)
    modes = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    mode_labels = ['Direct LLM', 'Agentic (No KG)', 'Agentic (With KG)']
    x = np.arange(len(modes))

    offsets = get_series_offsets(len(models), SIMPLE_CORRECTNESS_STYLE['offset_max_span'])

    for idx, model_name in enumerate(models):
        values = [get_metric_value(all_data[model_name], category, mode, 'answer_correctness') for mode in modes]
        color = MODEL_COLORS_LINE.get(model_name, f'C{idx}')
        marker = SERIES_MARKERS[idx % len(SERIES_MARKERS)]
        x_shifted = x + offsets[idx]

        ax.plot(
            x_shifted,
            values,
            '-',
            linewidth=SIMPLE_CORRECTNESS_STYLE['line_width'],
            label=model_name,
            color=color,
            marker=marker,
            markersize=SIMPLE_CORRECTNESS_STYLE['marker_size'],
            markeredgecolor='white',
            markeredgewidth=1.2,
            alpha=0.96,
            zorder=2,
        )
        ax.scatter(
            x_shifted,
            values,
            s=SIMPLE_CORRECTNESS_STYLE['scatter_size'],
            color=color,
            marker=marker,
            edgecolor='white',
            linewidth=1.0,
            zorder=3,
        )

    ax.set_ylabel('Answer Correctness', fontsize=SIMPLE_CORRECTNESS_STYLE['ylabel_fontsize'])
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=SIMPLE_CORRECTNESS_STYLE['xtick_fontsize'])
    ax.tick_params(axis='y', labelsize=SIMPLE_CORRECTNESS_STYLE['ytick_fontsize'])
    ax.set_ylim(0.2, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, SIMPLE_CORRECTNESS_STYLE['legend_anchor_y']),
        ncol=min(3, len(models)),
        fontsize=SIMPLE_CORRECTNESS_STYLE['legend_fontsize'],
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.5,
        borderaxespad=0,
        markerscale=1.08,
    )
    ax.grid(True, alpha=0.55, axis='y', linestyle='--')
    ax.grid(False, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#C7D0D9')
    ax.spines['bottom'].set_color('#C7D0D9')
    ax.margins(x=0.06)

    plt.tight_layout()
    save_overview_figure(fig, output_path, target_size=CORRECTNESS_EXPORT_SIZE)
    plt.close()


def create_mode_comparison_chart_correctness_only(all_data, category, output_path):
    models = sorted(all_data.keys(), key=lambda model: get_metric_value(all_data[model], category, 'agentic_with_kg', 'answer_correctness'), reverse=True)
    if not models:
        return

    modes = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(modes))
    width = 0.8 / len(models)

    for idx, model_name in enumerate(models):
        values = [get_metric_value(all_data[model_name], category, mode, 'answer_correctness') for mode in modes]
        offset = (idx - len(models) / 2 + 0.5) * width
        color = MODEL_COLORS.get(model_name, f'C{idx}')
        bars = ax.bar(x + offset, values, width, label=model_name, color=color, alpha=0.85)

        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=9, rotation=45)

    ax.set_ylabel('Answer Correctness', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Direct LLM', 'Agentic (No KG)', 'Agentic (With KG)'], fontsize=11)
    ax.set_ylim(0.3, 1.0)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close()


def create_per_model_radars(all_data, category, out_dir):
    colors = ['#ADC0D1', '#9BC3C6', '#F4E5BE']
    modes = ['direct_llm', 'agentic_no_kg', 'agentic_with_kg']

    for model_name in all_data:
        fig, ax = plt.subplots(figsize=RADAR_FIGSIZE, subplot_kw=dict(polar=True))
        num_metrics = len(METRICS)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]
        labels = [METRIC_LABELS.get(metric, metric) for metric in METRICS]

        has_data = False
        legend_labels = []
        for idx, mode in enumerate(modes):
            values = []
            for metric in METRICS:
                value = get_metric_value(all_data[model_name], category, mode, metric)
                if metric == 'context_precision' and value > 1:
                    value = 1.0 / value
                values.append(value)

            if sum(values) == 0:
                continue

            has_data = True
            values += values[:1]
            suffix = 'No KG' if 'no_kg' in mode else ('With KG' if 'with_kg' in mode else 'Direct')
            legend_labels.append(f'{model_name} ({suffix})')
            ax.plot(angles, values, 'o-', linewidth=1.8, markersize=4.5, color=colors[idx])
            ax.fill(angles, values, alpha=0.12, color=colors[idx])

        if not has_data:
            plt.close()
            continue

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
        plt.savefig(out_dir / f'{model_name}_radar.png', dpi=EXPORT_DPI, bbox_inches='tight')
        plt.close()


def get_breakdown_value(record, dimension):
    if dimension == 'category':
        return record['category']
    if dimension == 'difficulty':
        return record['difficulty']
    if dimension == 'reasoning':
        return record['reasoning']
    raise ValueError(f'Unsupported breakdown dimension: {dimension}')


def get_breakdown_order(dimension, values):
    values = list(values)
    if dimension == 'category':
        preferred_order = CATEGORY_ORDER
    elif dimension == 'difficulty':
        preferred_order = DIFFICULTY_ORDER
    else:
        preferred_order = REASONING_ORDER

    ordered_values = [value for value in preferred_order if value in values]
    ordered_values.extend(sorted(value for value in values if value not in preferred_order))
    return ordered_values


def format_breakdown_label(dimension, value):
    if dimension == 'difficulty':
        return str(value).capitalize()
    if dimension == 'reasoning':
        return str(value).replace('-', ' ').title()
    return str(value)


def summarize_correctness(records, dimension, correct_threshold):
    grouped = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'correct_count': 0})
    for record in records:
        group_key = get_breakdown_value(record, dimension)
        grouped[group_key]['sum'] += record['answer_correctness']
        grouped[group_key]['count'] += 1
        if record['answer_correctness'] >= correct_threshold:
            grouped[group_key]['correct_count'] += 1

    summary = {}
    for group_key, info in grouped.items():
        count = info['count']
        summary[group_key] = {
            'avg_correctness': info['sum'] / count if count else 0.0,
            'correct_rate': info['correct_count'] / count if count else 0.0,
            'count': count,
            'correct_count': info['correct_count'],
        }
    return summary


def format_metric_annotation(value, analysis_metric):
    if analysis_metric == 'correct_rate':
        return f'{value * 100:.1f}%'
    return f'{value:.2f}'


def get_series_values(series, ordered_groups, analysis_metric):
    values = []
    counts = []
    for group in ordered_groups:
        group_summary = series['summary'].get(group, {})
        count = group_summary.get('count', 0)
        value = safe_float(group_summary.get(analysis_metric))
        counts.append(count)
        values.append(value if count > 0 and value is not None else np.nan)
    return np.array(values, dtype=float), counts


def compute_breakdown_ylim(series_list, ordered_groups, analysis_metric):
    all_values = []
    for series in series_list:
        values, _ = get_series_values(series, ordered_groups, analysis_metric)
        all_values.extend(value for value in values if not np.isnan(value))

    if not all_values:
        return 0.0, 1.0

    min_value = min(all_values)
    max_value = max(all_values)
    span = max(max_value - min_value, 0.05)
    lower = max(0.0, min_value - span * 0.45)
    upper = min(1.02, max_value + span * 0.55)

    if upper - lower < 0.08:
        midpoint = (upper + lower) / 2
        lower = max(0.0, midpoint - 0.04)
        upper = min(1.02, midpoint + 0.04)

    return lower, upper


def get_series_rank_score(series, ordered_groups, analysis_metric):
    values, _ = get_series_values(series, ordered_groups, analysis_metric)
    valid_values = values[~np.isnan(values)]
    return float(np.mean(valid_values)) if valid_values.size else float('-inf')


def style_breakdown_axis(ax, x, ordered_groups, dimension, analysis_metric, y_limits):
    ax.set_ylabel(ANALYSIS_METRIC_LABELS[analysis_metric], fontsize=BREAKDOWN_CHART_STYLE['ylabel_fontsize'])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [wrap_label(format_breakdown_label(dimension, group), 16 if dimension == 'category' else 12) for group in ordered_groups],
        fontsize=BREAKDOWN_CHART_STYLE['xtick_fontsize'],
    )
    ax.tick_params(axis='y', labelsize=BREAKDOWN_CHART_STYLE['ytick_fontsize'])
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_formatter(_Y_FMT)
    ax.grid(True, alpha=0.55, axis='y', linestyle='--')
    ax.grid(False, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#C7D0D9')
    ax.spines['bottom'].set_color('#C7D0D9')
    ax.margins(x=0.06)


def plot_single_series_trend(ax, x, ordered_groups, series, analysis_metric, y_limits):
    values, counts = get_series_values(series, ordered_groups, analysis_metric)
    valid_mask = ~np.isnan(values)
    if not valid_mask.any():
        return

    x_valid = x[valid_mask]
    values_valid = values[valid_mask]
    baseline = np.full_like(values_valid, y_limits[0], dtype=float)
    color = series['color']

    ax.fill_between(x_valid, baseline, values_valid, color=color, alpha=0.12, zorder=1)
    ax.vlines(x_valid, y_limits[0], values_valid, color=color, alpha=0.22, linewidth=3.0, zorder=2)
    ax.plot(
        x_valid,
        values_valid,
        color=color,
        linewidth=BREAKDOWN_CHART_STYLE['line_width'],
        marker='o',
        markersize=BREAKDOWN_CHART_STYLE['marker_size'],
        markerfacecolor='white',
        markeredgewidth=2.0,
        zorder=3,
    )

    for x_val, value, count in zip(x_valid, values_valid, np.array(counts)[valid_mask]):
        if count == 0:
            continue
        ax.annotate(
            format_metric_annotation(value, analysis_metric),
            (x_val, value),
            textcoords='offset points',
            xytext=(0, 10),
            ha='center',
            va='bottom',
            fontsize=BREAKDOWN_CHART_STYLE['annotation_fontsize'],
            fontweight='bold',
            color=color,
            bbox={
                'boxstyle': 'round,pad=0.25',
                'facecolor': 'white',
                'edgecolor': color,
                'linewidth': 1,
                'alpha': 0.96,
            },
            zorder=4,
        )


def plot_multi_series_trend(ax, x, ordered_groups, series_list, analysis_metric):
    n_series = len(series_list)
    offsets = get_series_offsets(n_series, BREAKDOWN_CHART_STYLE['offset_max_span'])

    for idx, series in enumerate(series_list):
        values, _ = get_series_values(series, ordered_groups, analysis_metric)
        color = series['color']
        marker = SERIES_MARKERS[idx % len(SERIES_MARKERS)]
        x_shifted = x + offsets[idx]

        ax.plot(
            x_shifted,
            values,
            label=series['label'],
            color=color,
            linewidth=BREAKDOWN_CHART_STYLE['line_width'],
            marker=marker,
            markersize=BREAKDOWN_CHART_STYLE['marker_size'],
            markeredgecolor='white',
            markeredgewidth=1.2,
            alpha=0.96,
            zorder=2,
        )
        ax.scatter(
            x_shifted,
            values,
            s=BREAKDOWN_CHART_STYLE['scatter_size'],
            color=color,
            marker=marker,
            edgecolor='white',
            linewidth=1.0,
            zorder=3,
        )


def prepare_breakdown_chart_data(series_list, dimension, analysis_metric):
    active_series = [series for series in series_list if series.get('summary')]
    if not active_series:
        return None, None

    group_values = set()
    for series in active_series:
        group_values.update(series['summary'].keys())
    if not group_values:
        return None, None

    ordered_groups = get_breakdown_order(dimension, group_values)
    active_series = sorted(
        active_series,
        key=lambda series: get_series_rank_score(series, ordered_groups, analysis_metric),
        reverse=True,
    )
    return active_series, ordered_groups


def build_correctness_breakdown_export(series_list, dimension, analysis_metric):
    active_series, ordered_groups = prepare_breakdown_chart_data(series_list, dimension, analysis_metric)
    if not active_series:
        return None

    payload = {
        'dimension': dimension,
        'analysis_metric': analysis_metric,
        'ordered_groups': ordered_groups,
        'series': [],
    }

    rows = []
    for series in active_series:
        group_payload = {}
        for group in ordered_groups:
            group_summary = series['summary'].get(group)
            if not group_summary:
                continue

            value = safe_float(group_summary.get(analysis_metric))
            group_payload[group] = {
                'value': value,
                'count': int(group_summary.get('count', 0)),
                'correct_count': int(group_summary.get('correct_count', 0)),
                'correct_rate': safe_float(group_summary.get('correct_rate')),
                'avg_correctness': safe_float(group_summary.get('avg_correctness')),
            }
            rows.append({
                'series_label': series['label'],
                'series_color': series.get('color'),
                'group': group,
                'value': value,
                'count': group_payload[group]['count'],
                'correct_count': group_payload[group]['correct_count'],
                'correct_rate': group_payload[group]['correct_rate'],
                'avg_correctness': group_payload[group]['avg_correctness'],
            })

        payload['series'].append({
            'label': series['label'],
            'color': series.get('color'),
            'groups': group_payload,
        })

    return payload, rows


def export_correctness_breakdown_data(series_list, dimension, analysis_metric, output_path):
    export_bundle = build_correctness_breakdown_export(series_list, dimension, analysis_metric)
    if not export_bundle:
        return [], []

    payload, rows = export_bundle
    json_path = output_path.with_suffix('.json')
    csv_path = output_path.with_suffix('.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    with open(csv_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'series_label',
                'series_color',
                'group',
                'value',
                'count',
                'correct_count',
                'correct_rate',
                'avg_correctness',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return [json_path, csv_path], rows


def export_by_method_master_table(output_dir, analysis_metric, selected_methods, selected_breakdowns, rows):
    if not rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f'master_breakdown_table_{analysis_metric}.json'
    csv_path = output_dir / f'master_breakdown_table_{analysis_metric}.csv'

    payload = {
        'analysis_metric': analysis_metric,
        'methods': [MODE_SHORT_NAMES.get(method, method) for method in selected_methods],
        'dimensions': list(selected_breakdowns),
        'rows': rows,
    }

    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    with open(csv_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'method',
                'dimension',
                'series_label',
                'series_color',
                'group',
                'value',
                'count',
                'correct_count',
                'correct_rate',
                'avg_correctness',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return [json_path, csv_path]


def create_correctness_breakdown_chart(series_list, dimension, analysis_metric, output_path, title):
    active_series, ordered_groups = prepare_breakdown_chart_data(series_list, dimension, analysis_metric)
    if not active_series:
        return False

    x = np.arange(len(ordered_groups))
    n_series = len(active_series)
    y_limits = compute_breakdown_ylim(active_series, ordered_groups, analysis_metric)

    fig, ax = plt.subplots(figsize=CORRECTNESS_FIGSIZE)

    if n_series == 1:
        plot_single_series_trend(ax, x, ordered_groups, active_series[0], analysis_metric, y_limits)
    else:
        plot_multi_series_trend(ax, x, ordered_groups, active_series, analysis_metric)

    style_breakdown_axis(ax, x, ordered_groups, dimension, analysis_metric, y_limits)

    if n_series > 1:
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, BREAKDOWN_CHART_STYLE['legend_anchor_y']),
            ncol=min(3, n_series),
            fontsize=BREAKDOWN_CHART_STYLE['legend_fontsize'],
            columnspacing=1.2,
            handlelength=2.2,
            handletextpad=0.5,
            borderaxespad=0,
            markerscale=1.05,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    save_overview_figure(fig, output_path, target_size=CORRECTNESS_EXPORT_SIZE)
    plt.close()
    return True


def filter_records(records, model_names=None, modes=None):
    return [
        record for record in records
        if (model_names is None or record['model'] in model_names) and (modes is None or record['mode'] in modes)
    ]


def generate_correctness_breakdown_plots(records, out_dir, selected_models, selected_methods, selected_breakdowns, analysis_metric, correct_threshold):
    filtered_records = filter_records(records, model_names=selected_models, modes=selected_methods)
    if not filtered_records:
        print('No QA-level correctness records matched the selected filters.')
        return

    breakdown_dir = out_dir / 'correctness_breakdown'
    breakdown_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    by_method_master_rows = []

    for model_name in selected_models:
        model_records = filter_records(filtered_records, model_names=[model_name])
        if not model_records:
            continue

        for dimension in selected_breakdowns:
            if len(selected_methods) == 1:
                mode_name = selected_methods[0]
                summary = summarize_correctness(filter_records(model_records, modes=[mode_name]), dimension, correct_threshold)
                series_list = [{
                    'label': f'{model_name} · {get_display_mode_name(mode_name)}',
                    'color': MODE_COLORS.get(mode_name, '#ADC0D1'),
                    'summary': summary,
                }]
                output_path = breakdown_dir / 'by_model' / slugify(model_name) / MODE_SHORT_NAMES[mode_name] / f'{dimension}_{analysis_metric}.png'
                title = f'{model_name} · {get_display_mode_name(mode_name)} · {BREAKDOWN_DISPLAY_NAMES[dimension]}'
            else:
                series_list = []
                for method_name in selected_methods:
                    summary = summarize_correctness(filter_records(model_records, modes=[method_name]), dimension, correct_threshold)
                    if summary:
                        series_list.append({
                            'label': get_display_mode_name(method_name),
                            'color': MODE_COLORS.get(method_name, '#ADC0D1'),
                            'summary': summary,
                        })

                output_path = breakdown_dir / 'by_model' / slugify(model_name) / f'{dimension}_{analysis_metric}.png'
                title = f'{model_name} · Method comparison by {BREAKDOWN_DISPLAY_NAMES[dimension]}'

            if create_correctness_breakdown_chart(series_list, dimension, analysis_metric, output_path, title):
                generated_files.append(str(output_path.relative_to(out_dir)))
                sidecar_paths, _ = export_correctness_breakdown_data(series_list, dimension, analysis_metric, output_path)
                for sidecar_path in sidecar_paths:
                    generated_files.append(str(sidecar_path.relative_to(out_dir)))

    if len(selected_models) > 1:
        for mode_name in selected_methods:
            mode_records = filter_records(filtered_records, modes=[mode_name])
            if not mode_records:
                continue

            for dimension in selected_breakdowns:
                series_list = []
                for model_name in selected_models:
                    summary = summarize_correctness(filter_records(mode_records, model_names=[model_name]), dimension, correct_threshold)
                    if summary:
                        series_list.append({
                            'label': f'{model_name} ({get_mode_legend_suffix(mode_name)})',
                            'color': MODEL_COLORS_LINE.get(model_name, '#ADC0D1'),
                            'summary': summary,
                        })

                output_path = breakdown_dir / 'by_method' / MODE_SHORT_NAMES[mode_name] / f'{dimension}_{analysis_metric}.png'
                title = f'{get_display_mode_name(mode_name)} · Model comparison by {BREAKDOWN_DISPLAY_NAMES[dimension]}'
                if create_correctness_breakdown_chart(series_list, dimension, analysis_metric, output_path, title):
                    generated_files.append(str(output_path.relative_to(out_dir)))
                    sidecar_paths, sidecar_rows = export_correctness_breakdown_data(series_list, dimension, analysis_metric, output_path)
                    for sidecar_path in sidecar_paths:
                        generated_files.append(str(sidecar_path.relative_to(out_dir)))
                    for row in sidecar_rows:
                        by_method_master_rows.append({
                            'method': MODE_SHORT_NAMES.get(mode_name, mode_name),
                            'dimension': dimension,
                            **row,
                        })

    for master_path in export_by_method_master_table(
        breakdown_dir / 'by_method',
        analysis_metric,
        selected_methods,
        selected_breakdowns,
        by_method_master_rows,
    ):
        generated_files.append(str(master_path.relative_to(out_dir)))

    manifest = {
        'selected_models': selected_models,
        'selected_methods': selected_methods,
        'selected_breakdowns': selected_breakdowns,
        'analysis_metric': analysis_metric,
        'correct_threshold': correct_threshold,
        'generated_files': generated_files,
    }
    manifest_path = breakdown_dir / 'analysis_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f'Generated {len(generated_files)} correctness breakdown plot(s) in {breakdown_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radar', action='store_true', help='Generate per-model radar charts')
    parser.add_argument('--models', nargs='+', default=['all'], help='Models to analyze, e.g. Gpt-4o-mini Qwen3-235B or all')
    parser.add_argument('--methods', nargs='+', default=['all'], help='Methods to analyze: direct no_kg with_kg or all')
    parser.add_argument('--breakdowns', nargs='+', default=['all'], help='Breakdown dimensions: category difficulty reasoning or all')
    parser.add_argument('--analysis-metric', choices=['avg_correctness', 'correct_rate'], default='avg_correctness', help='Use mean answer_correctness or threshold-based correct rate')
    parser.add_argument('--correct-threshold', type=float, default=0.8, help='Threshold used when analysis-metric=correct_rate')
    parser.add_argument('--skip-breakdown', action='store_true', help='Skip category/difficulty/reasoning correctness analysis plots')
    parser.add_argument('--breakdown-only', action='store_true', help='Generate only category/difficulty/reasoning correctness analysis plots')
    args = parser.parse_args()

    if args.skip_breakdown and args.breakdown_only:
        parser.error('--skip-breakdown and --breakdown-only cannot be used together')

    base_dir = Path(__file__).parent.parent
    min4_dir = base_dir / 'results_min4'
    out_dir = min4_dir / 'eval_visualize'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading data from {min4_dir}...')
    all_data = load_min4_data(min4_dir)
    print(f'Loaded {len(all_data)} models.')
    if not all_data:
        return

    try:
        selected_models = resolve_models(args.models, all_data.keys())
        selected_methods = resolve_methods(args.methods)
        selected_breakdowns = resolve_breakdowns(args.breakdowns)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Selected models: {', '.join(selected_models)}")
    print(f"Selected methods: {', '.join(get_display_mode_name(method) for method in selected_methods)}")
    print(f"Selected breakdowns: {', '.join(selected_breakdowns)}")

    if not args.breakdown_only:
        print('Aggregating categories...')
        aggregated_data = aggregate_all_categories(all_data)
        category_label = 'all'

        try:
            sys.path.insert(0, str(base_dir))
            from compare_models import create_heatmap, create_multi_model_radar, create_ranking_table_all_modes

            create_multi_model_radar(aggregated_data, category_label, 'agentic_with_kg', str(out_dir / 'radar_comparison.png'))
            create_heatmap(aggregated_data, category_label, 'agentic_with_kg', str(out_dir / 'heatmap.png'))
            create_ranking_table_all_modes(aggregated_data, category_label, str(out_dir / 'ranking.png'))
        except Exception as exc:
            print(f'Could not use original compare_models generators for basic charts: {exc}')

        print('Generating overview plots...')
        create_grouped_bar_chart(aggregated_data, category_label, 'agentic_with_kg', str(out_dir / 'metrics_comparison.png'))
        create_grouped_bar_chart(aggregated_data, category_label, 'agentic_with_kg', str(out_dir / 'metrics_comparison_nolabel.png'), show_values=False)
        create_improvement_charts(aggregated_data, category_label, out_dir)
        create_agentic_vs_direct_charts(aggregated_data, category_label, out_dir)
        create_simple_correctness_chart(aggregated_data, category_label, str(out_dir / 'correctness_simple.png'))
        create_mode_comparison_chart_correctness_only(aggregated_data, category_label, str(out_dir / 'mode_comparison.png'))

        if args.radar:
            print('Generating individual radar plots...')
            create_per_model_radars(aggregated_data, category_label, out_dir)

    if not args.skip_breakdown:
        print('Loading QA-level correctness records...')
        question_records = load_min4_question_records(min4_dir)
        print(f'Loaded {len(question_records)} QA-level correctness records.')
        generate_correctness_breakdown_plots(
            records=question_records,
            out_dir=out_dir,
            selected_models=selected_models,
            selected_methods=selected_methods,
            selected_breakdowns=selected_breakdowns,
            analysis_metric=args.analysis_metric,
            correct_threshold=args.correct_threshold,
        )

    print(f'✅ All requested plots have been generated and saved to {out_dir}')


if __name__ == '__main__':
    main()
