#!/usr/bin/env python
"""
对比所有模型的 dual_condition_report.txt，找出重合的问题

使用方法:
    python compare_reports.py
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations


def load_qa_metadata(base_dir):
    """从results.json文件中加载所有QA的difficulty和reasoning_type元数据"""
    qa_metadata = {}  # {qa_id: {'difficulty': '...', 'reasoning_type': '...'}}
    
    base_path = Path(base_dir)
    
    # 查找任意一个模型的results.json文件来获取元数据
    # 因为同一个QA在不同模型中的difficulty和reasoning_type应该是一样的
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        for category_dir in model_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            # 优先使用agentic_with_kg的结果
            results_file = category_dir / "agentic_with_kg" / "results.json"
            if not results_file.exists():
                results_file = category_dir / "agentic_no_kg" / "results.json"
            
            if results_file.exists():
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for qa_result in data.get('qa_results', []):
                        qa_id = qa_result.get('question_id')
                        if qa_id and qa_id not in qa_metadata:
                            qa_metadata[qa_id] = {
                                'difficulty': qa_result.get('difficulty', 'unknown'),
                                'reasoning_type': qa_result.get('reasoning_type', 'unknown'),
                                'category': category_dir.name
                            }
                except (json.JSONDecodeError, KeyError):
                    continue
    
    return qa_metadata


def extract_qa_ids_from_report(file_path):
    """从报告文件中提取所有QA的ID，按条件类型分组，同时提取类别信息"""
    qa_by_condition = {
        'both_conditions': set(),  # 双条件
        'only_condition1': set(),  # 仅条件1
        'only_condition2': set()   # 仅条件2
    }
    
    # 同时记录每个QA的类别
    qa_category_map = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    current_section = None
    current_category = None
    
    for i, line in enumerate(lines):
        # 识别当前章节
        if '✅ 满足双条件的QA' in line:
            current_section = 'both_conditions'
        elif '⚠️ 仅满足条件1' in line:
            current_section = 'only_condition1'
        elif '⚠️ 仅满足条件2' in line:
            current_section = 'only_condition2'
        elif '📊 统计摘要' in line:
            break  # 到统计部分就停止
        
        # 识别当前类别 - 匹配 "📁 类别: xxx (N 个)"
        category_match = re.match(r'📁 类别: (\w+)', line)
        if category_match:
            current_category = category_match.group(1)
        
        # 提取 QA ID（以两个空格开头，后面跟问题ID）
        if current_section and line.strip() and not line.startswith('    '):
            # 匹配像 "  qa_0016" 这样的行
            match = re.match(r'\s{2}(qa_\d+)\s*$', line)
            if match:
                qa_id = match.group(1)
                qa_by_condition[current_section].add(qa_id)
                if current_category:
                    qa_category_map[qa_id] = current_category
    
    return qa_by_condition, qa_category_map


def discover_reports(base_dir):
    """自动发现所有模型的报告文件"""
    reports = {}
    base_path = Path(base_dir)
    
    for report_file in base_path.glob("*/dual_condition_report.txt"):
        model_name = report_file.parent.name
        reports[model_name] = report_file
    
    return reports


def compare_two_models(name1, qa1, name2, qa2):
    """对比两个模型的结果"""
    results = {}
    
    # 获取所有QA集合
    all1 = qa1['both_conditions'] | qa1['only_condition1'] | qa1['only_condition2']
    all2 = qa2['both_conditions'] | qa2['only_condition1'] | qa2['only_condition2']
    
    # 总体重合
    results['overlap_total'] = all1 & all2
    results['only_in_1'] = all1 - all2
    results['only_in_2'] = all2 - all1
    
    # 双条件重合
    results['overlap_both'] = qa1['both_conditions'] & qa2['both_conditions']
    
    # 仅条件1重合
    results['overlap_cond1'] = qa1['only_condition1'] & qa2['only_condition1']
    
    # 仅条件2重合
    results['overlap_cond2'] = qa1['only_condition2'] & qa2['only_condition2']
    
    return results


def compare_all_models(reports_data):
    """对比所有模型的结果"""
    model_names = list(reports_data.keys())
    pairwise_results = {}
    
    # 两两对比
    for name1, name2 in combinations(model_names, 2):
        key = f"{name1} vs {name2}"
        pairwise_results[key] = compare_two_models(
            name1, reports_data[name1],
            name2, reports_data[name2]
        )
    
    return pairwise_results


def find_common_qa_across_all(reports_data, condition='both_conditions'):
    """找出所有模型中共同出现的QA"""
    model_names = list(reports_data.keys())
    if not model_names:
        return set()
    
    # 从第一个模型开始
    common = reports_data[model_names[0]][condition].copy()
    
    # 与其他模型求交集
    for name in model_names[1:]:
        common &= reports_data[name][condition]
    
    return common


def print_model_summary(reports_data):
    """打印各模型的统计摘要"""
    print("=" * 100)
    print("📊 各模型QA数量统计")
    print("=" * 100)
    print()
    
    # 表头
    print(f"{'模型名称':<25} {'双条件✅':>12} {'仅条件1⚠️':>12} {'仅条件2⚠️':>12} {'合计':>10}")
    print("-" * 75)
    
    for model_name, qa_data in sorted(reports_data.items()):
        both = len(qa_data['both_conditions'])
        cond1 = len(qa_data['only_condition1'])
        cond2 = len(qa_data['only_condition2'])
        total = both + cond1 + cond2
        print(f"{model_name:<25} {both:>12} {cond1:>12} {cond2:>12} {total:>10}")
    
    print()


def print_pairwise_comparison(pairwise_results, reports_data):
    """打印两两对比结果"""
    print("=" * 100)
    print("🔄 两两模型对比分析")
    print("=" * 100)
    print()
    
    for pair_name, results in sorted(pairwise_results.items()):
        names = pair_name.split(' vs ')
        name1, name2 = names[0], names[1]
        
        all1 = (reports_data[name1]['both_conditions'] | 
                reports_data[name1]['only_condition1'] | 
                reports_data[name1]['only_condition2'])
        all2 = (reports_data[name2]['both_conditions'] | 
                reports_data[name2]['only_condition1'] | 
                reports_data[name2]['only_condition2'])
        
        print(f"📌 {pair_name}")
        print("-" * 60)
        print(f"  总体重合: {len(results['overlap_total'])} 个")
        if len(all1) > 0:
            print(f"  重合率 (基于{name1}): {len(results['overlap_total'])/len(all1)*100:.1f}%")
        if len(all2) > 0:
            print(f"  重合率 (基于{name2}): {len(results['overlap_total'])/len(all2)*100:.1f}%")
        print(f"  {name1}独有: {len(results['only_in_1'])} 个")
        print(f"  {name2}独有: {len(results['only_in_2'])} 个")
        print()
        
        print(f"  双条件重合: {len(results['overlap_both'])} 个")
        print(f"  仅条件1重合: {len(results['overlap_cond1'])} 个")
        print(f"  仅条件2重合: {len(results['overlap_cond2'])} 个")
        print()


def print_common_across_all(reports_data):
    """打印所有模型共同的QA"""
    model_names = list(reports_data.keys())
    n_models = len(model_names)
    
    print("=" * 100)
    print(f"🎯 所有 {n_models} 个模型的共同分析")
    print("=" * 100)
    print()
    
    # 所有模型共同的双条件QA
    common_both = find_common_qa_across_all(reports_data, 'both_conditions')
    print(f"✅ 所有模型都满足双条件的QA: {len(common_both)} 个")
    
    # 所有模型共同的仅条件1 QA
    common_cond1 = find_common_qa_across_all(reports_data, 'only_condition1')
    print(f"⚠️ 所有模型都仅满足条件1的QA: {len(common_cond1)} 个")
    
    # 所有模型共同的仅条件2 QA
    common_cond2 = find_common_qa_across_all(reports_data, 'only_condition2')
    print(f"⚠️ 所有模型都仅满足条件2的QA: {len(common_cond2)} 个")
    
    # 合并所有类型
    all_common = set()
    for model_name in model_names:
        model_all = (reports_data[model_name]['both_conditions'] | 
                     reports_data[model_name]['only_condition1'] | 
                     reports_data[model_name]['only_condition2'])
        if len(all_common) == 0:
            all_common = model_all.copy()
        else:
            all_common &= model_all
    
    print(f"\n📊 所有模型都出现的QA (任意条件): {len(all_common)} 个")
    
    print()
    return {
        'common_both': common_both,
        'common_cond1': common_cond1,
        'common_cond2': common_cond2,
        'all_common': all_common
    }


def analyze_qa_appearance(reports_data):
    """分析每个QA在多少个模型中出现"""
    model_names = list(reports_data.keys())
    qa_appearance = defaultdict(lambda: {'models': set(), 'conditions': defaultdict(set)})
    
    for model_name in model_names:
        for condition in ['both_conditions', 'only_condition1', 'only_condition2']:
            for qa_id in reports_data[model_name][condition]:
                qa_appearance[qa_id]['models'].add(model_name)
                qa_appearance[qa_id]['conditions'][condition].add(model_name)
    
    return qa_appearance


def print_qa_appearance_stats(qa_appearance, n_models):
    """打印QA出现次数统计"""
    print("=" * 100)
    print("� QA出现频次统计")
    print("=" * 100)
    print()
    
    # 按出现模型数量分组
    appearance_groups = defaultdict(list)
    for qa_id, info in qa_appearance.items():
        count = len(info['models'])
        appearance_groups[count].append(qa_id)
    
    for count in sorted(appearance_groups.keys(), reverse=True):
        qa_list = sorted(appearance_groups[count])
        print(f"在 {count}/{n_models} 个模型中出现: {len(qa_list)} 个 QA")
    
    print()
    return appearance_groups


def print_common_qa_details(common_data, reports_data, max_display=50):
    """打印共同QA的详细列表"""
    print("=" * 100)
    print("📝 所有模型共同的双条件QA列表")
    print("=" * 100)
    print()
    
    common_both = sorted(common_data['common_both'])
    
    if len(common_both) == 0:
        print("  (无)")
    else:
        displayed = common_both[:max_display]
        for i, qa_id in enumerate(displayed, 1):
            print(f"  {i:4d}. {qa_id}")
        
        if len(common_both) > max_display:
            print(f"\n  ... 还有 {len(common_both) - max_display} 个未显示")
    
    print()


def print_category_distribution(common_data, qa_category_maps):
    """打印共同QA的类别分布"""
    print("=" * 100)
    print("📊 所有模型共同双条件QA的类别分布")
    print("=" * 100)
    print()
    
    # 使用第一个可用的类别映射（所有模型应该有相同的QA-类别映射）
    category_map = {}
    for model_map in qa_category_maps.values():
        category_map.update(model_map)
    
    # 统计双条件QA的类别分布
    common_both = common_data['common_both']
    category_counts = defaultdict(list)
    unknown_count = 0
    
    for qa_id in common_both:
        if qa_id in category_map:
            category_counts[category_map[qa_id]].append(qa_id)
        else:
            unknown_count += 1
    
    # 打印表头
    print(f"{'类别':<20} {'数量':>10} {'占比':>10}")
    print("-" * 45)
    
    total = len(common_both)
    for category in sorted(category_counts.keys()):
        count = len(category_counts[category])
        percentage = count / total * 100 if total > 0 else 0
        print(f"{category:<20} {count:>10} {percentage:>9.1f}%")
    
    if unknown_count > 0:
        print(f"{'(未知)':<20} {unknown_count:>10} {unknown_count/total*100:>9.1f}%")
    
    print("-" * 45)
    print(f"{'合计':<20} {total:>10} {'100.0%':>10}")
    print()
    
    # 同样分析仅条件1和仅条件2
    for condition_name, condition_key in [('仅条件1', 'common_cond1'), ('仅条件2', 'common_cond2')]:
        common_set = common_data[condition_key]
        if len(common_set) > 0:
            print(f"\n📊 所有模型共同{condition_name}的QA类别分布:")
            print("-" * 45)
            cond_category_counts = defaultdict(int)
            for qa_id in common_set:
                if qa_id in category_map:
                    cond_category_counts[category_map[qa_id]] += 1
            
            cond_total = len(common_set)
            for category in sorted(cond_category_counts.keys()):
                count = cond_category_counts[category]
                percentage = count / cond_total * 100 if cond_total > 0 else 0
                print(f"{category:<20} {count:>10} {percentage:>9.1f}%")
            print(f"{'合计':<20} {cond_total:>10} {'100.0%':>10}")
    
    print()
    return category_counts


def print_difficulty_reasoning_distribution(common_data, qa_metadata):
    """打印共同QA的难度和推理类型分布"""
    print("=" * 100)
    print("📊 所有模型共同双条件QA的难度分布")
    print("=" * 100)
    print()
    
    common_both = common_data['common_both']
    
    # 统计难度分布
    difficulty_counts = defaultdict(int)
    reasoning_counts = defaultdict(int)
    difficulty_reasoning_matrix = defaultdict(lambda: defaultdict(int))
    
    for qa_id in common_both:
        if qa_id in qa_metadata:
            difficulty = qa_metadata[qa_id].get('difficulty', 'unknown')
            reasoning_type = qa_metadata[qa_id].get('reasoning_type', 'unknown')
            difficulty_counts[difficulty] += 1
            reasoning_counts[reasoning_type] += 1
            difficulty_reasoning_matrix[difficulty][reasoning_type] += 1
        else:
            difficulty_counts['unknown'] += 1
            reasoning_counts['unknown'] += 1
    
    total = len(common_both)
    
    # 打印难度分布
    print(f"{'难度 (Difficulty)':<20} {'数量':>10} {'占比':>10}")
    print("-" * 45)
    for difficulty in sorted(difficulty_counts.keys()):
        count = difficulty_counts[difficulty]
        percentage = count / total * 100 if total > 0 else 0
        print(f"{difficulty:<20} {count:>10} {percentage:>9.1f}%")
    print("-" * 45)
    print(f"{'合计':<20} {total:>10} {'100.0%':>10}")
    
    print()
    print("=" * 100)
    print("📊 所有模型共同双条件QA的推理类型分布")
    print("=" * 100)
    print()
    
    # 打印推理类型分布
    print(f"{'推理类型 (Reasoning Type)':<25} {'数量':>10} {'占比':>10}")
    print("-" * 50)
    for reasoning_type in sorted(reasoning_counts.keys()):
        count = reasoning_counts[reasoning_type]
        percentage = count / total * 100 if total > 0 else 0
        print(f"{reasoning_type:<25} {count:>10} {percentage:>9.1f}%")
    print("-" * 50)
    print(f"{'合计':<25} {total:>10} {'100.0%':>10}")
    
    # 打印难度×推理类型交叉分析
    print()
    print("=" * 100)
    print("📊 难度 × 推理类型 交叉分析")
    print("=" * 100)
    print()
    
    # 获取所有的推理类型
    all_reasoning_types = sorted(reasoning_counts.keys())
    
    # 打印表头
    header = f"{'难度':<12}"
    for rt in all_reasoning_types:
        header += f" {rt[:15]:>15}"
    header += f" {'合计':>10}"
    print(header)
    print("-" * (12 + 16 * len(all_reasoning_types) + 10))
    
    # 打印每行
    for difficulty in sorted(difficulty_counts.keys()):
        row = f"{difficulty:<12}"
        row_total = 0
        for rt in all_reasoning_types:
            count = difficulty_reasoning_matrix[difficulty][rt]
            row_total += count
            row += f" {count:>15}"
        row += f" {row_total:>10}"
        print(row)
    
    # 打印列合计
    footer = f"{'合计':<12}"
    for rt in all_reasoning_types:
        footer += f" {reasoning_counts[rt]:>15}"
    footer += f" {total:>10}"
    print("-" * (12 + 16 * len(all_reasoning_types) + 10))
    print(footer)
    
    print()
    
    # 分别分析仅条件1和仅条件2
    for condition_name, condition_key in [('仅条件1', 'common_cond1'), ('仅条件2', 'common_cond2')]:
        common_set = common_data[condition_key]
        if len(common_set) > 0:
            print(f"\n📊 所有模型共同{condition_name}的QA难度和推理类型分布:")
            print("-" * 60)
            
            cond_difficulty_counts = defaultdict(int)
            cond_reasoning_counts = defaultdict(int)
            
            for qa_id in common_set:
                if qa_id in qa_metadata:
                    cond_difficulty_counts[qa_metadata[qa_id].get('difficulty', 'unknown')] += 1
                    cond_reasoning_counts[qa_metadata[qa_id].get('reasoning_type', 'unknown')] += 1
            
            cond_total = len(common_set)
            
            print(f"\n难度分布:")
            for difficulty in sorted(cond_difficulty_counts.keys()):
                count = cond_difficulty_counts[difficulty]
                percentage = count / cond_total * 100 if cond_total > 0 else 0
                print(f"  {difficulty:<15} {count:>8} ({percentage:>5.1f}%)")
            
            print(f"\n推理类型分布:")
            for rt in sorted(cond_reasoning_counts.keys()):
                count = cond_reasoning_counts[rt]
                percentage = count / cond_total * 100 if cond_total > 0 else 0
                print(f"  {rt:<20} {count:>8} ({percentage:>5.1f}%)")
    
    print()
    return difficulty_counts, reasoning_counts


def save_common_qa_to_file(common_data, output_path):
    """将共同的QA ID保存到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 所有模型共同的QA ID\n\n")
        
        f.write("## 所有模型都满足双条件的QA\n")
        for qa_id in sorted(common_data['common_both']):
            f.write(f"{qa_id}\n")
        
        f.write(f"\n## 所有模型都仅满足条件1的QA\n")
        for qa_id in sorted(common_data['common_cond1']):
            f.write(f"{qa_id}\n")
        
        f.write(f"\n## 所有模型都仅满足条件2的QA\n")
        for qa_id in sorted(common_data['common_cond2']):
            f.write(f"{qa_id}\n")
        
        f.write(f"\n## 所有模型都出现的QA (任意条件)\n")
        for qa_id in sorted(common_data['all_common']):
            f.write(f"{qa_id}\n")
    
    print(f"📄 共同QA列表已保存到: {output_path}")


def main():
    # 定义文件路径
    base_dir = Path("/home/wzfeng/RAGPhoto/Data_Agentic_RAG/evaluation/results")
    
    # 自动发现所有报告文件
    print("=" * 100)
    print("🔍 发现的模型报告文件")
    print("=" * 100)
    print()
    
    reports = discover_reports(base_dir)
    
    if len(reports) == 0:
        print("❌ 错误: 未找到任何 dual_condition_report.txt 文件")
        return
    
    for model_name, file_path in sorted(reports.items()):
        print(f"  ✓ {model_name}: {file_path}")
    
    print(f"\n共发现 {len(reports)} 个模型的报告\n")
    
    # 解析所有报告
    print("正在解析报告文件...\n")
    reports_data = {}
    qa_category_maps = {}
    for model_name, file_path in reports.items():
        qa_by_condition, qa_category_map = extract_qa_ids_from_report(file_path)
        reports_data[model_name] = qa_by_condition
        qa_category_maps[model_name] = qa_category_map
    
    # 加载QA元数据 (difficulty 和 reasoning_type)
    print("正在加载QA元数据 (difficulty, reasoning_type)...\n")
    qa_metadata = load_qa_metadata(base_dir)
    print(f"已加载 {len(qa_metadata)} 个QA的元数据\n")
    
    # 打印各模型统计
    print_model_summary(reports_data)
    
    # 两两对比
    if len(reports_data) >= 2:
        pairwise_results = compare_all_models(reports_data)
        print_pairwise_comparison(pairwise_results, reports_data)
    
    # 所有模型共同分析
    common_data = print_common_across_all(reports_data)
    
    # QA出现频次分析
    qa_appearance = analyze_qa_appearance(reports_data)
    appearance_groups = print_qa_appearance_stats(qa_appearance, len(reports_data))
    
    # 打印类别分布统计
    category_counts = print_category_distribution(common_data, qa_category_maps)
    
    # 打印难度和推理类型分布统计
    difficulty_counts, reasoning_counts = print_difficulty_reasoning_distribution(common_data, qa_metadata)
    
    # 打印共同QA详情
    print_common_qa_details(common_data, reports_data)
    
    # 保存共同QA到文件
    output_file = base_dir / "common_qa_across_models.txt"
    save_common_qa_to_file(common_data, output_file)
    
    print()
    print("=" * 100)
    print("✅ 对比分析完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
