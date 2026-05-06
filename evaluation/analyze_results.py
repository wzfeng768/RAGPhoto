#!/usr/bin/env python3
"""
结果分析脚本
正确解读和对比不同模式的评估结果
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List


def load_results(mode: str, dataset: str = None, model: str = None) -> Dict[str, Any]:
    """
    加载指定模式的评估结果
    支持三层结构: results/<模型>/<类别>/<模式>/results.json
    """
    results_dir = Path(__file__).parent / "results"

    if model and dataset:
        # 三层结构：results/<模型>/<类别>/<模式>/results.json
        result_file = results_dir / model / dataset / mode / "results.json"
    elif dataset:
        # 两层结构（旧格式）：results/<数据集>/<模式>/results.json
        result_file = results_dir / dataset / mode / "results.json"
    else:
        # 旧格式向后兼容
        result_file = results_dir / mode / "results.json"

    if not result_file.exists():
        return None

    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_available_models() -> List[str]:
    """查找所有可用的模型目录"""
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return []
    
    skip_dirs = {'visualize_results', 'model_comparison'}
    models = []
    for item in results_dir.iterdir():
        if not item.is_dir() or item.name in skip_dirs or item.name.startswith('_'):
            continue
        # 检查是否包含 <类别>/<模式>/results.json 格式
        has_results = any(
            True
            for cat_dir in item.iterdir()
            if cat_dir.is_dir()
            for mode in ['agentic_with_kg', 'agentic_no_kg', 'direct_llm']
            if (cat_dir / mode / 'results.json').exists()
        )
        if has_results:
            models.append(item.name)
    return sorted(models)


def find_available_datasets(model: str = None) -> List[str]:
    """查找指定模型（或所有模型）下的可用类别"""
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return []

    skip_dirs = {'visualize_results', 'model_comparison'}
    datasets = set()

    model_dirs = ([results_dir / model] if model else
                  [d for d in results_dir.iterdir()
                   if d.is_dir() and d.name not in skip_dirs and not d.name.startswith('_')])

    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        for cat_dir in model_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            has_results = any(
                (cat_dir / mode / 'results.json').exists()
                for mode in ['agentic_with_kg', 'agentic_no_kg', 'direct_llm']
            )
            if has_results:
                datasets.add(cat_dir.name)

    return sorted(datasets)


def _fmt(value, decimals=4):
    """安全格式化数值，None 显示为 N/A"""
    if value is None:
        return 'N/A   '
    return f"{value:.{decimals}f}"


def analyze_mode(mode: str, data: Dict[str, Any]):
    """分析单个模式的结果"""
    print(f"\n{'='*80}")
    print(f"模式: {mode.replace('_', ' ').title()}")
    print(f"{'='*80}")

    metadata = data['metadata']
    metrics = data['ragas_metrics']
    performance = data['performance']

    print(f"\n基本信息:")
    # 兼容新格式（answer_generation.model）和旧格式（metadata.model）
    model_name = (metadata.get('answer_generation', {}).get('model')
                  or metadata.get('model', 'N/A'))
    num_q = (metadata.get('num_questions_completed')
             or metadata.get('num_questions', 'N/A'))
    timestamp = metadata.get('timestamp', metadata.get('evaluation', {}).get('metrics_version', 'N/A'))
    print(f"  模型: {model_name}")
    print(f"  问题数: {num_q}")
    print(f"  评估版本/时间: {timestamp}")
    eval_model = metadata.get('evaluation', {}).get('ragas_model', '')
    if eval_model:
        print(f"  评估模型: {eval_model}")
    if 'kg_enabled' in metadata:
        print(f"  知识图谱: {'✅ 启用' if metadata['kg_enabled'] else '❌ 禁用'}")

    # 根据模式分析不同的指标
    if mode == 'direct_llm':
        print(f"\n⚠️  注意: Direct LLM模式只有答案质量指标有效")
        print(f"\n📊 答案质量指标 (有效):")
        print(f"  Answer Relevancy:    {_fmt(metrics.get('answer_relevancy'))}")
        print(f"  Answer Correctness:  {_fmt(metrics.get('answer_correctness'))}")
        print(f"  Answer Similarity:   {_fmt(metrics.get('answer_similarity'))}")

        vals = [v for v in [metrics.get('answer_relevancy'),
                             metrics.get('answer_correctness'),
                             metrics.get('answer_similarity')] if v is not None]
        if vals:
            answer_avg = sum(vals) / len(vals)
            print(f"  ─────────────────────────────────")
            print(f"  答案质量平均分:      {answer_avg:.4f}")

        print(f"\n❌ 检索质量指标 (不适用 - 无检索):")
        print(f"  Faithfulness:        {_fmt(metrics.get('faithfulness'))} (恒为无效)")
        print(f"  Context Precision:   {_fmt(metrics.get('context_precision'))} (恒为无效)")
        print(f"  Context Recall:      {_fmt(metrics.get('context_recall'))} (恒为无效)")

        fact_metrics = data.get('fact_metrics', {})
        if fact_metrics and 'error' not in fact_metrics:
            print(f"\n🎯 事实正确性指标 (重要):")
            print(f"  Factual Correctness: {_fmt(fact_metrics.get('factual_correctness', 0))}")
            print(f"  Key Facts Coverage:  {_fmt(fact_metrics.get('key_facts_coverage', 0))}")
            print(f"  Fact Precision:      {_fmt(fact_metrics.get('fact_precision', 0))}")
            print(f"  Critical Errors:     {fact_metrics.get('total_critical_errors', 0)}")

        weighted = data.get('weighted_score')
        if weighted is not None:
            print(f"\n⭐ 加权综合评分 (推荐关注):")
            print(f"  Weighted Score:      {weighted:.4f}")
            print(f"  (RAGAS 30% + Fact 70%)")

    else:
        print(f"\n📊 完整RAGAS指标:")
        print(f"\n  检索质量:")
        print(f"    Faithfulness:        {_fmt(metrics.get('faithfulness'))}")
        print(f"    Context Precision:   {_fmt(metrics.get('context_precision'))}")
        print(f"    Context Recall:      {_fmt(metrics.get('context_recall'))}")

        r_vals = [v for v in [metrics.get('faithfulness'),
                               metrics.get('context_precision'),
                               metrics.get('context_recall')] if v is not None]
        if r_vals:
            retrieval_avg = sum(r_vals) / len(r_vals)
            print(f"    检索质量平均分:      {retrieval_avg:.4f}")

        print(f"\n  答案质量:")
        print(f"    Answer Relevancy:    {_fmt(metrics.get('answer_relevancy'))}")
        print(f"    Answer Correctness:  {_fmt(metrics.get('answer_correctness'))}")
        print(f"    Answer Similarity:   {_fmt(metrics.get('answer_similarity'))}")

        a_vals = [v for v in [metrics.get('answer_relevancy'),
                               metrics.get('answer_correctness'),
                               metrics.get('answer_similarity')] if v is not None]
        if a_vals:
            answer_avg = sum(a_vals) / len(a_vals)
            print(f"    答案质量平均分:      {answer_avg:.4f}")

        print(f"\n  综合评分:")
        print(f"    RAGAS平均分:         {_fmt(metrics.get('average_score'))}")

        fact_metrics = data.get('fact_metrics', {})
        if fact_metrics and 'error' not in fact_metrics:
            print(f"\n🎯 事实正确性指标:")
            print(f"    Factual Correctness: {_fmt(fact_metrics.get('factual_correctness', 0))}")
            print(f"    Key Facts Coverage:  {_fmt(fact_metrics.get('key_facts_coverage', 0))}")
            print(f"    Fact Precision:      {_fmt(fact_metrics.get('fact_precision', 0))}")
            print(f"    Hallucination Rate:  {_fmt(fact_metrics.get('hallucination_rate', 0))}")
            print(f"    Critical Errors:     {fact_metrics.get('total_critical_errors', 0)}")

        weighted = data.get('weighted_score')
        if weighted is not None:
            print(f"\n⭐ 加权综合评分:")
            print(f"    Weighted Score:      {weighted:.4f}")
            print(f"    (RAGAS 30% + Fact 70%)")

    print(f"\n⏱️  性能统计:")
    print(f"  平均响应时间: {performance['avg_response_time']:.2f}秒")
    if 'avg_iterations' in performance:
        print(f"  平均迭代次数: {performance['avg_iterations']:.2f}")
    if 'avg_quality_score' in performance:
        print(f"  平均质量分数: {performance['avg_quality_score']:.4f}")
    print(f"  总耗时: {performance['total_time']:.2f}秒")
    print(f"  错误数: {performance['error_count']}")


def compare_modes(results: Dict[str, Dict]):
    """对比不同模式的结果"""
    print(f"\n{'='*80}")
    print("模式对比分析")
    print(f"{'='*80}")
    
    # 提取所有模式的答案质量指标（公平对比）
    print(f"\n📊 答案质量对比 (所有模式适用):")
    print(f"\n{'指标':<25} ", end="")
    modes = list(results.keys())
    for mode in modes:
        print(f"{mode.replace('_', ' ')[:20]:>20} ", end="")
    print()
    print("─" * 80)
    
    answer_metrics = ['answer_relevancy', 'answer_correctness', 'answer_similarity']
    for metric in answer_metrics:
        print(f"{metric:<25} ", end="")
        for mode in modes:
            value = results[mode]['ragas_metrics'].get(metric)
            print(f"{_fmt(value):>20} ", end="")
        print()

    # 答案质量平均分
    print("─" * 80)
    print(f"{'答案质量平均':<25} ", end="")
    for mode in modes:
        metrics = results[mode]['ragas_metrics']
        vals = [metrics.get(m) for m in answer_metrics if metrics.get(m) is not None]
        avg = sum(vals) / len(vals) if vals else float('nan')
        print(f"{_fmt(avg):>20} ", end="")
    print("\n")

    # 只对比有检索的模式
    retrieval_modes = {k: v for k, v in results.items() if k != 'direct_llm'}
    if len(retrieval_modes) > 1:
        print(f"\n📊 检索质量对比 (仅Agentic模式):")
        print(f"\n{'指标':<25} ", end="")
        for mode in retrieval_modes.keys():
            print(f"{mode.replace('_', ' ')[:20]:>20} ", end="")
        print()
        print("─" * 80)

        retrieval_metrics = ['faithfulness', 'context_precision', 'context_recall']
        for metric in retrieval_metrics:
            print(f"{metric:<25} ", end="")
            for mode in retrieval_modes.keys():
                value = results[mode]['ragas_metrics'].get(metric)
                print(f"{_fmt(value):>20} ", end="")
            print()

        # 检索质量平均分
        print("─" * 80)
        print(f"{'检索质量平均':<25} ", end="")
        for mode in retrieval_modes.keys():
            metrics = results[mode]['ragas_metrics']
            vals = [metrics.get(m) for m in retrieval_metrics if metrics.get(m) is not None]
            avg = sum(vals) / len(vals) if vals else float('nan')
            print(f"{_fmt(avg):>20} ", end="")
        print("\n")

    # 性能对比
    print(f"\n⏱️  性能对比:")
    print(f"\n{'指标':<25} ", end="")
    for mode in modes:
        print(f"{mode.replace('_', ' ')[:20]:>20} ", end="")
    print()
    print("─" * 80)

    print(f"{'平均响应时间(秒)':<25} ", end="")
    for mode in modes:
        value = results[mode]['performance']['avg_response_time']
        print(f"{value:>20.2f} ", end="")
    print()

    if all('avg_iterations' in results[m]['performance'] for m in modes if m != 'direct_llm'):
        print(f"{'平均迭代次数':<25} ", end="")
        for mode in modes:
            if mode == 'direct_llm':
                print(f"{'N/A':>20} ", end="")
            else:
                value = results[mode]['performance']['avg_iterations']
                print(f"{value:>20.2f} ", end="")
        print()



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析RAG评估结果')
    parser.add_argument('--dataset', type=str, default=None,
                       help='指定要分析的类别（如 computational / characterization 等）')
    parser.add_argument('--model', type=str, default=None,
                       help='指定要分析的模型（如 Gpt-4o-mini / GLM-4.7 等）')
    args = parser.parse_args()

    print("="*80)
    print("RAG评估结果分析工具")
    print("="*80)

    # 查找可用模型
    available_models = find_available_models()

    if not available_models:
        print("\n⚠️  未找到任何评估结果")
        print("请先运行评估: python evaluate_rag.py --dataset computational --num_questions 10")
        return

    # 确定要分析的模型列表
    if args.model:
        if args.model not in available_models:
            print(f"\n❌ 错误: 模型 '{args.model}' 不存在")
            print(f"可用模型: {', '.join(available_models)}")
            return
        models_to_analyze = [args.model]
    else:
        models_to_analyze = available_models

    print(f"\n🤖 可用模型: {', '.join(available_models)}")
    print(f"📊 分析模型: {', '.join(models_to_analyze)}")

    # 遍历每个模型
    for model_name in models_to_analyze:
        # 查找该模型下的可用类别
        available_datasets = find_available_datasets(model_name)

        # 确定要分析的类别
        if args.dataset:
            if args.dataset not in available_datasets:
                print(f"\n⚠️  模型 {model_name} 中不存在类别 '{args.dataset}'，跳过")
                continue
            datasets_to_analyze = [args.dataset]
        else:
            datasets_to_analyze = available_datasets

        print(f"\n{'#'*80}")
        print(f"🤖 模型: {model_name}")
        print(f"📁 可用类别: {', '.join(available_datasets)}")
        print(f"{'#'*80}")

        for dataset in datasets_to_analyze:
            print(f"\n{'='*60}")
            print(f"📂 类别: {dataset}")
            print(f"{'='*60}")

            modes = ['agentic_with_kg', 'agentic_no_kg', 'direct_llm']
            results = {}

            for mode in modes:
                data = load_results(mode, dataset, model_name)
                if data:
                    results[mode] = data
                    analyze_mode(mode, data)

            if not results:
                print(f"\n⚠️  {model_name}/{dataset} 没有有效评估结果")
                continue

            if len(results) > 1:
                compare_modes(results)

    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}\n")
    print("💡 提示:")
    print("  - Direct LLM的检索指标为0是正常的（无检索操作）")
    print("  - 对比时应关注答案质量指标（answer_*），而非综合平均分")
    print("  - 加权综合评分(weighted_score)才是推荐的最终评价指标")
    print("  - 详细解释请参考 METRICS_EXPLANATION.md")
    print()


if __name__ == "__main__":
    main()
