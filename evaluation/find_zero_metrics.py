#!/usr/bin/env python
"""
查找 Gpt-4o-mini/*/agentic_with_kg/results.json 中:
1. answer_correctness > 0.5 的QA对
2. agentic_with_kg 与 agentic_no_kg 的 answer_correctness 差值 > 0.2 的QA对

输出格式: 按满足条件的情况分组打印问题类型、问题编号和评估指标

使用方法:
    # 使用默认模型 (Qwen3-235B)
    python find_zero_metrics.py
    
    # 指定模型
    python find_zero_metrics.py --model Gpt-4o-mini
    python find_zero_metrics.py --model Qwen3-235B DeepSeek-V3.2-Thinking
"""

import json
import math
import argparse
from pathlib import Path



# 筛选阈值
CORRECTNESS_THRESHOLD = 0.4  # 条件1: answer_correctness > 0.5

# 定义要显示的评估指标
METRICS_TO_DISPLAY = [
    "answer_correctness",
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "context_relevancy"
]


def is_null_or_nan(value):
    """检查值是否为 null/None/NaN"""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def load_results_file(file_path):
    """加载结果文件并返回 {question_id: qa_data} 映射"""
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 {file_path}: {e}")
        return None
    
    qa_results = data.get("qa_results", [])
    return {qa.get("question_id"): qa for qa in qa_results}


def get_metric_value(qa_data, metric_name):
    """获取指标值，如果为null则返回None"""
    if not qa_data:
        return None
    ragas_scores = qa_data.get("ragas_scores", {})
    value = ragas_scores.get(metric_name)
    if is_null_or_nan(value):
        return None
    return value


def format_metrics(qa_data):
    """格式化显示所有评估指标"""
    if not qa_data:
        return "无数据"
    
    ragas_scores = qa_data.get("ragas_scores", {})
    parts = []
    for metric in METRICS_TO_DISPLAY:
        value = ragas_scores.get(metric)
        if is_null_or_nan(value):
            parts.append(f"{metric}=null")
        else:
            parts.append(f"{metric}={value:.4f}")
    return ", ".join(parts)


def compare_metrics(base_dir):
    """比较 agentic_with_kg 和 agentic_no_kg 的评估指标"""
    
    # 存储分类结果
    both_conditions = []  # 两个条件都满足
    only_condition1 = []  # 仅满足条件1 (correctness < 0.6)
    only_condition2 = []  # 仅满足条件2 (差值 > 0.2)
    
    # 遍历所有子文件夹（类别）
    for dataset_dir in sorted(base_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        category_name = dataset_dir.name
        
        # 加载两个模式的结果
        with_kg_file = dataset_dir / "agentic_with_kg" / "results.json"
        no_kg_file = dataset_dir / "agentic_no_kg" / "results.json"
        
        with_kg_data = load_results_file(with_kg_file)
        no_kg_data = load_results_file(no_kg_file)
        
        if with_kg_data is None:
            print(f"警告: {with_kg_file} 不存在或无法解析，跳过")
            continue
        
        if no_kg_data is None:
            print(f"警告: {no_kg_file} 不存在或无法解析，跳过 {category_name}")
            continue
        
        # 遍历 with_kg 中的所有 QA
        for qa_id, with_kg_qa in with_kg_data.items():
            no_kg_qa = no_kg_data.get(qa_id)
            
            # 获取 answer_correctness 值
            with_kg_correctness = get_metric_value(with_kg_qa, "answer_correctness")
            no_kg_correctness = get_metric_value(no_kg_qa, "answer_correctness")
            
            # 跳过缺失数据的情况
            if with_kg_correctness is None:
                continue
            
            # 条件1: with_kg 的 correctness >= no_kg 的 correctness
            if no_kg_correctness is not None:
                diff = with_kg_correctness - no_kg_correctness  # 正值表示with_kg更好
                condition1 = with_kg_correctness >= no_kg_correctness
            else:
                diff = None
                condition1 = False
            
            # 条件2: with_kg 的 correctness > 0.6
            condition2 = with_kg_correctness > CORRECTNESS_THRESHOLD
            
            # 记录结果
            result_item = {
                "category": category_name,
                "qa_id": qa_id,
                "with_kg_metrics": format_metrics(with_kg_qa),
                "no_kg_metrics": format_metrics(no_kg_qa),
                "with_kg_correctness": with_kg_correctness,
                "no_kg_correctness": no_kg_correctness,
                "diff": diff
            }
            
            if condition1 and condition2:
                both_conditions.append(result_item)
            elif condition1:
                only_condition1.append(result_item)
            elif condition2:
                only_condition2.append(result_item)
    
    return both_conditions, only_condition1, only_condition2


def print_results(both_conditions, only_condition1, only_condition2, output_file):
    """打印和保存结果"""
    
    output_lines = []
    
    def add_line(text=""):
        print(text)
        output_lines.append(text)
    
    # ========== 开头汇总统计 ==========
    add_line("=" * 100)
    add_line("📊 各类别数量汇总")
    add_line("=" * 100)
    
    # 统计各类别在各条件下的数量
    all_categories = set()
    cat_stats = {}
    
    for item in both_conditions:
        cat = item["category"]
        all_categories.add(cat)
        if cat not in cat_stats:
            cat_stats[cat] = {"both": 0, "cond1": 0, "cond2": 0}
        cat_stats[cat]["both"] += 1
    
    for item in only_condition1:
        cat = item["category"]
        all_categories.add(cat)
        if cat not in cat_stats:
            cat_stats[cat] = {"both": 0, "cond1": 0, "cond2": 0}
        cat_stats[cat]["cond1"] += 1
    
    for item in only_condition2:
        cat = item["category"]
        all_categories.add(cat)
        if cat not in cat_stats:
            cat_stats[cat] = {"both": 0, "cond1": 0, "cond2": 0}
        cat_stats[cat]["cond2"] += 1
    
    add_line("")
    add_line(f"{'类别':<20} {'双条件✅':>10} {'仅条件1⚠️':>12} {'仅条件2⚠️':>12} {'合计':>8}")
    add_line("-" * 70)
    
    for category in sorted(all_categories):
        stats = cat_stats[category]
        total = stats["both"] + stats["cond1"] + stats["cond2"]
        add_line(f"{category:<20} {stats['both']:>10} {stats['cond1']:>12} {stats['cond2']:>12} {total:>8}")
    
    total_both = len(both_conditions)
    total_cond1 = len(only_condition1)
    total_cond2 = len(only_condition2)
    add_line("-" * 70)
    add_line(f"{'总计':<20} {total_both:>10} {total_cond1:>12} {total_cond2:>12} {total_both + total_cond1 + total_cond2:>8}")
    add_line("")
    
    # ========== 满足双条件 ==========
    add_line("=" * 100)
    add_line(f"✅ 满足双条件的QA (with_kg >= no_kg 且 with_kg>{CORRECTNESS_THRESHOLD}) - 共 {len(both_conditions)} 个")
    add_line("=" * 100)
    
    if both_conditions:
        # 按类别分组
        by_category = {}
        for item in both_conditions:
            cat = item["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)
        
        for category in sorted(by_category.keys()):
            items = by_category[category]
            add_line(f"\n📁 类别: {category} ({len(items)} 个)")
            add_line("-" * 80)
            
            for item in items:
                add_line(f"  {item['qa_id']}")
                add_line(f"    [with_kg] {item['with_kg_metrics']}")
                add_line(f"    [no_kg]   {item['no_kg_metrics']}")
                diff_str = f"{item['diff']:.4f}" if item['diff'] is not None else "N/A"
                add_line(f"    差值: {diff_str}")
    else:
        add_line("  (无)")
    
    # ========== 仅满足条件1 ==========
    add_line("\n" + "=" * 100)
    add_line(f"⚠️ 仅满足条件1 (with_kg >= no_kg，但with_kg<={CORRECTNESS_THRESHOLD}) - 共 {len(only_condition1)} 个")
    add_line("=" * 100)
    
    if only_condition1:
        by_category = {}
        for item in only_condition1:
            cat = item["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)
        
        for category in sorted(by_category.keys()):
            items = by_category[category]
            add_line(f"\n📁 类别: {category} ({len(items)} 个)")
            add_line("-" * 80)
            
            for item in items:
                add_line(f"  {item['qa_id']}")
                add_line(f"    [with_kg] {item['with_kg_metrics']}")
                add_line(f"    [no_kg]   {item['no_kg_metrics']}")
                diff_str = f"{item['diff']:.4f}" if item['diff'] is not None else "N/A"
                add_line(f"    差值: {diff_str}")
    else:
        add_line("  (无)")
    
    # ========== 仅满足条件2 ==========
    add_line("\n" + "=" * 100)
    add_line(f"⚠️ 仅满足条件2 (with_kg>{CORRECTNESS_THRESHOLD}，但with_kg < no_kg) - 共 {len(only_condition2)} 个")
    add_line("=" * 100)
    
    if only_condition2:
        by_category = {}
        for item in only_condition2:
            cat = item["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)
        
        for category in sorted(by_category.keys()):
            items = by_category[category]
            add_line(f"\n📁 类别: {category} ({len(items)} 个)")
            add_line("-" * 80)
            
            for item in items:
                add_line(f"  {item['qa_id']}")
                add_line(f"    [with_kg] {item['with_kg_metrics']}")
                add_line(f"    [no_kg]   {item['no_kg_metrics']}")
                diff_str = f"{item['diff']:.4f}" if item['diff'] is not None else "N/A"
                add_line(f"    差值: {diff_str}")
    else:
        add_line("  (无)")
    
    # ========== 统计摘要 ==========
    add_line("\n" + "=" * 100)
    add_line("📊 统计摘要")
    add_line("=" * 100)
    add_line(f"  满足双条件: {len(both_conditions)} 个")
    add_line(f"  仅满足条件1 (with_kg略优但低分): {len(only_condition1)} 个")
    add_line(f"  仅满足条件2 (高分但no_kg更优): {len(only_condition2)} 个")
    add_line(f"  总计需关注: {len(both_conditions) + len(only_condition1) + len(only_condition2)} 个")
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    
    # ========== 额外输出: 高分QA合并文件 ==========
    high_score_file = output_file.parent / "high_score_qa_report.txt"
    high_score_lines = []
    
    # 合并 both_conditions 和 only_condition2 (都是 with_kg > 0.5 的高分QA)
    combined_high_score = both_conditions + only_condition2
    
    high_score_lines.append("=" * 100)
    high_score_lines.append(f"📊 高分QA汇总 (with_kg correctness > {CORRECTNESS_THRESHOLD}) - 共 {len(combined_high_score)} 个")
    high_score_lines.append("=" * 100)
    high_score_lines.append(f"  ✅ with_kg > no_kg: {len(both_conditions)} 个")
    high_score_lines.append(f"  ⚠️ with_kg <= no_kg: {len(only_condition2)} 个")
    high_score_lines.append("")
    
    # 按类别分组
    by_category = {}
    for item in combined_high_score:
        cat = item["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)
    
    for category in sorted(by_category.keys()):
        items = by_category[category]
        high_score_lines.append(f"\n📁 类别: {category} ({len(items)} 个)")
        high_score_lines.append("-" * 80)
        
        for item in items:
            # 标记 with_kg 是否更优
            if item['diff'] is not None and item['diff'] > 0:
                marker = "✅"
            else:
                marker = "⚠️"
            high_score_lines.append(f"  {marker} {item['qa_id']}")
            high_score_lines.append(f"      [with_kg] {item['with_kg_metrics']}")
            high_score_lines.append(f"      [no_kg]   {item['no_kg_metrics']}")
            diff_str = f"{item['diff']:.4f}" if item['diff'] is not None else "N/A"
            high_score_lines.append(f"      差值: {diff_str}")
    
    with open(high_score_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(high_score_lines))
    
    print(f"\n高分QA汇总报告已保存到: {high_score_file}")
    
    print(f"\n报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析评估指标')
    parser.add_argument('--model', type=str, default='Qwen3-235B', help='指定模型名称，默认为 Qwen3-235B')
    args = parser.parse_args()

    model_name = args.model
    
    # 动态定义路径
    base_dir = Path(f"/data/wzfeng/RAGPhoto/Data_Agentic_RAG/evaluation/results/{model_name}")
    output_file = base_dir / "dual_condition_report.txt"
    
    if not base_dir.exists():
        print(f"❌ 错误: 找不到模型目录: {base_dir}")
        return

    print(f"开始分析评估指标 (模型: {model_name})...")
    print(f"路径: {base_dir}")
    print(f"条件1: agentic_with_kg 的 answer_correctness >= agentic_no_kg")
    print(f"条件2: agentic_with_kg 的 answer_correctness > {CORRECTNESS_THRESHOLD}")
    print("-" * 80)
    
    both_conditions, only_condition1, only_condition2 = compare_metrics(base_dir)
    print_results(both_conditions, only_condition1, only_condition2, output_file)


if __name__ == "__main__":
    main()
