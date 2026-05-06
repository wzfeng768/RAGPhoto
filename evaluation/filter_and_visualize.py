#!/usr/bin/env python3
"""
过滤评估结果脚本 - 支持处理所有模型和类别
1. 根据 dual_condition_report.txt 中的问答对ID，仅保留指定的问答对
2. 将过滤后的结果保存到新文件夹 results_filtered/{model}/{category}/
3. 调用可视化脚本生成图表

使用方法:
    # 处理 Gpt-4o-mini 模型的所有类别（过滤+可视化）
    python filter_and_visualize.py
    
    # 仅执行过滤，不生成可视化
    python filter_and_visualize.py --filter-only
    
    # 仅执行可视化（使用已过滤的数据）
    python filter_and_visualize.py --visualize-only
    
    # 处理指定模型
    python filter_and_visualize.py --model Qwen3-235B
    
    # 处理所有模型
    python filter_and_visualize.py --model all
    
    # 只处理特定类别
    python filter_and_visualize.py --category characterization
    
    # 处理特定模型的特定类别（仅过滤）
    python filter_and_visualize.py --model Gpt-4o-mini --category characterization --filter-only
"""

import json
import os
import argparse
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import subprocess


def parse_dual_condition_report(report_file: str) -> dict:
    """
    解析 dual_condition_report.txt 文件，返回每个类别需要保留的 QA ID 集合
    只提取"满足双条件的QA"部分的数据
    
    Returns:
        dict: {category: set(qa_ids)}
    """
    keep_dict = {}
    current_category = None
    in_dual_condition_section = False
    
    # 匹配section开始：✅ 满足双条件的QA
    dual_section_start = re.compile(r'✅ 满足双条件的QA')
    # 匹配section结束：⚠️ 仅满足条件
    section_end = re.compile(r'⚠️ 仅满足条件')
    # 匹配类别行，如 "📁 类别: characterization (189 个)"
    category_pattern = re.compile(r'📁 类别: (\w+)')
    # 匹配 qa_ ID 行
    qa_pattern = re.compile(r'^\s*(qa_\d+)\s*$')
    
    with open(report_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # 检查是否进入双条件section
            if dual_section_start.search(line):
                in_dual_condition_section = True
                continue
            
            # 检查是否离开双条件section（进入其他condition section）
            if section_end.search(line):
                in_dual_condition_section = False
                break  # 可以直接退出循环
            
            # 只在双条件section中解析
            if not in_dual_condition_section:
                continue
            
            # 检查是否是类别行
            cat_match = category_pattern.search(line)
            if cat_match:
                current_category = cat_match.group(1)
                keep_dict[current_category] = set()
                continue
            
            # 检查是否是 qa_ ID 行
            qa_match = qa_pattern.match(line)
            if qa_match and current_category:
                keep_dict[current_category].add(qa_match.group(1))
    
    return keep_dict


def filter_results(results_data: dict, qa_ids_to_keep: set) -> tuple:
    """
    从结果数据中仅保留指定的 QA ID
    
    Args:
        results_data: 原始结果数据
        qa_ids_to_keep: 需要保留的 QA ID 集合
        
    Returns:
        tuple: (过滤后的结果数据, 过滤统计信息)
    """
    if 'qa_results' not in results_data:
        return results_data, {"original": 0, "filtered": 0, "removed": 0}
    
    # 过滤 qa_results - 仅保留在 qa_ids_to_keep 中的
    original_count = len(results_data['qa_results'])
    filtered_qa_results = []
    
    for qa in results_data['qa_results']:
        qa_id = qa.get('question_id', '')
        if qa_id in qa_ids_to_keep:
            filtered_qa_results.append(qa)
    
    filtered_count = len(filtered_qa_results)
    stats = {
        "original": original_count,
        "filtered": filtered_count,
        "deleted": original_count - filtered_count
    }
    
    # 创建新的结果数据
    filtered_data = results_data.copy()
    filtered_data['qa_results'] = filtered_qa_results
    
    # 重新计算 ragas_metrics（如果存在）
    if filtered_qa_results:
        metrics_names = ['faithfulness', 'answer_correctness', 'context_recall',
                        'context_precision', 'answer_similarity', 'answer_relevancy']
        
        new_metrics = {}
        for metric in metrics_names:
            values = []
            for qa in filtered_qa_results:
                v = qa.get('ragas_scores', {}).get(metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    values.append(v)
            
            if values:
                new_metrics[metric] = np.mean(values)
            else:
                new_metrics[metric] = None
        
        filtered_data['ragas_metrics'] = new_metrics
    
    return filtered_data, stats


def process_category(category_dir: Path, output_dir: Path, qa_ids_to_keep: set) -> dict:
    """
    处理单个类别目录
    
    Args:
        category_dir: 源类别目录
        output_dir: 输出目录
        qa_ids_to_keep: 需要保留的 QA ID 集合
    
    Returns:
        dict: 各模式的处理统计
    """
    mode_stats = {}
    
    # 遍历子目录（agentic_with_kg, agentic_no_kg, direct_llm）
    for subdir in sorted(category_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        mode = subdir.name
        results_file = subdir / 'results.json'
        
        if not results_file.exists():
            print(f"    跳过 {mode}: results.json 不存在")
            continue
        
        # 读取原始结果
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # 过滤结果 - 仅保留在 qa_ids_to_keep 中的
        filtered_data, stats = filter_results(results_data, qa_ids_to_keep)
        mode_stats[mode] = stats
        
        print(f"    {mode}: {stats['original']} -> {stats['filtered']} (删除 {stats['deleted']})")
        
        # 创建输出目录
        output_subdir = output_dir / mode
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # 保存过滤后的结果
        output_file = output_subdir / 'results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    return mode_stats


def run_visualization(base_dir: Path, output_dir: Path):
    """运行可视化脚本"""
    visualize_script = base_dir / 'visualize_results.py'
    if not visualize_script.exists():
        print(f"  ⚠️  可视化脚本不存在: {visualize_script}")
        return False
    
    cmd = ['python3', str(visualize_script), '--dir', str(output_dir)]
    result = subprocess.run(cmd, cwd=str(base_dir), capture_output=True, text=True)
    
    if result.returncode == 0:
        # 只打印关键输出
        for line in result.stdout.split('\n'):
            if '✅' in line or 'Done!' in line:
                print(f"    {line}")
        return True
    else:
        print(f"  ❌ 可视化失败: {result.stderr[:200]}")
        return False


def run_visualization_summary(base_dir: Path, model_dir: Path):
    """运行跨类别汇总可视化"""
    visualize_script = base_dir / 'visualize_results.py'
    if not visualize_script.exists():
        print(f"  ⚠️  可视化脚本不存在: {visualize_script}")
        return False
    
    cmd = ['python3', str(visualize_script), '--dir', str(model_dir), '--summary-charts']
    result = subprocess.run(cmd, cwd=str(base_dir), capture_output=True, text=True)
    
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if '✅' in line:
                print(f"    {line}")
        return True
    else:
        print(f"  ❌ 汇总可视化失败: {result.stderr[:200]}")
        return False


def main():
    parser = argparse.ArgumentParser(description='过滤评估结果并生成可视化 (仅保留 dual_condition_report.txt 中的 QA)')
    parser.add_argument('--model', type=str, default='Gpt-4o-mini',
                       help='指定要处理的模型名称，默认为 Gpt-4o-mini。使用 "all" 处理所有模型')
    parser.add_argument('--category', type=str, default=None,
                       help='指定要处理的类别，不指定则处理所有类别')
    parser.add_argument('--filter-only', action='store_true',
                       help='仅执行过滤，不生成可视化')
    parser.add_argument('--visualize-only', action='store_true',
                       help='仅执行可视化（使用已过滤的数据），不执行过滤')
    parser.add_argument('--no-visualize', action='store_true',
                       help='[已废弃] 跳过可视化步骤，请使用 --filter-only')
    args = parser.parse_args()
    
    # 处理参数兼容性
    if args.no_visualize:
        args.filter_only = True
    
    if args.filter_only and args.visualize_only:
        print("❌ 错误: --filter-only 和 --visualize-only 不能同时使用")
        return
    
    # 处理 --model all 的情况
    process_all_models = (args.model == 'all')
    if process_all_models:
        args.model = None
    
    # 配置路径
    base_dir = Path('/home/wzfeng/RAGPhoto/Data_Agentic_RAG/evaluation')
    results_dir = base_dir / 'results'
    output_base = base_dir / 'results_filtered'
    # 确定执行模式
    do_filter = not args.visualize_only
    do_visualize = not args.filter_only
    
    mode_str = "过滤+可视化" if (do_filter and do_visualize) else ("仅过滤" if do_filter else "仅可视化")
    print("=" * 70)
    print(f"🚀 执行模式: {mode_str}")
    print("=" * 70)
    

    
    # 获取模型目录（过滤模式用原始目录，可视化模式用已过滤目录）
    if args.visualize_only:
        # 仅可视化模式：使用已过滤的数据
        source_base = output_base
        if not source_base.exists():
            print(f"\n❌ 未找到已过滤的数据目录: {source_base}")
            print("   请先运行 --filter-only 进行过滤")
            return
    else:
        source_base = results_dir
    
    model_dirs = []
    for item in sorted(source_base.iterdir()):
        if item.is_dir() and item.name not in ['visualize_results', 'model_comparison']:
            if args.model is None or item.name == args.model:
                model_dirs.append(item)
    
    if not model_dirs:
        print(f"\n❌ 未找到模型目录" + (f" (指定: {args.model})" if args.model else ""))
        return
    
    print(f"\n📁 将处理 {len(model_dirs)} 个模型:")
    for md in model_dirs:
        print(f"     - {md.name}")
    
    # 汇总统计
    all_stats = {}
    
    # 处理每个模型
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n{'=' * 70}")
        print(f"🔧 处理模型: {model_name}")
        print("=" * 70)
        
        all_stats[model_name] = {}
        
        # 🟢 新逻辑: 为每个模型单独解析对应的 dual_condition_report.txt
        current_keep_dict = {}
        if do_filter:
            report_file = model_dir / 'dual_condition_report.txt'
            if not report_file.exists():
                print(f"  ❌ 警告: 未找到该模型的报告文件: {report_file}")
                print(f"     跳过该模型的过滤步骤")
                continue
            
            print(f"  📋 解析保留列表: {report_file.name}")
            current_keep_dict = parse_dual_condition_report(str(report_file))
            
            total_to_keep = sum(len(ids) for ids in current_keep_dict.values())
            print(f"     共 {len(current_keep_dict)} 个类别, {total_to_keep} 个问答对将保留")
        
        # 获取该模型下的所有类别目录
        category_dirs = []
        for item in sorted(model_dir.iterdir()):
            if item.is_dir():
                if args.category is None or item.name == args.category:
                    category_dirs.append(item)
        
        if not category_dirs:
            print(f"  ⚠️  未找到类别目录" + (f" (指定: {args.category})" if args.category else ""))
            continue
        
        # 处理每个类别
        for category_dir in category_dirs:
            category_name = category_dir.name
            qa_ids_to_keep = current_keep_dict.get(category_name, set())
            
            # 输出目录
            output_dir = output_base / model_name / category_name
            
            if do_filter:
                # 如果 keep_dict 中没有这个类别，但我们要过滤，说明这个类别没有QA要保留 (或者报告里没提到它)
                # 这种情况下应该保留 0 个，或者视为空列表
                
                print(f"\n  📂 {category_name} (保留 {len(qa_ids_to_keep)} 个问答对)")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 执行过滤
                mode_stats = process_category(category_dir, output_dir, qa_ids_to_keep)
                all_stats[model_name][category_name] = mode_stats
            else:
                print(f"\n  📂 {category_name}")
                mode_stats = {'skipped': True}
                all_stats[model_name][category_name] = mode_stats
            
            # 运行可视化
            if do_visualize:
                # 可视化使用已过滤的数据目录
                vis_source = output_dir if output_dir.exists() else category_dir
                print(f"  📊 生成可视化 (数据源: {vis_source.name})...")
                run_visualization(base_dir, vis_source)
        
        # 🟢 在每个模型的所有类别处理完后，运行跨类别汇总可视化
        if do_visualize:
            # 汇总数据保存在 visualize_results/{model_name} 下
            summary_model_dir = results_dir / 'visualize_results' / model_name
            if summary_model_dir.exists():
                print(f"\n  📈 生成跨类别对比汇总图表...")
                run_visualization_summary(base_dir, summary_model_dir)

    
    # 打印汇总
    print(f"\n{'=' * 70}")
    print("📊 处理完成汇总")
    print("=" * 70)
    
    if do_filter:
        for model_name, categories in all_stats.items():
            print(f"\n{model_name}:")
            for category_name, modes in categories.items():
                if isinstance(modes, dict) and 'skipped' not in modes:
                    total_removed = sum(s.get('deleted', 0) for s in modes.values())
                    total_filtered = sum(s.get('filtered', 0) for s in modes.values())
                    print(f"  └─ {category_name}: 保留 {total_filtered // len(modes) if modes else 0} 条/模式, 移除 {total_removed // len(modes) if modes else 0} 条")
        print(f"\n✅ 过滤后的结果保存在: {output_base}")
    
    if do_visualize:
        print(f"📈 可视化结果保存在: {results_dir / 'visualize_results'}")


if __name__ == '__main__':
    main()

