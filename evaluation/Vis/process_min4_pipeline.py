#!/usr/bin/env python3
"""
Pipeline for QA Results with Minimum Model Support.
This script consolidates filtering, metrics recalculation, and semantic visualization into one file.
"""

import json
import shutil
import pickle
import time
import sys
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

# Enable importing from evaluation path
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from compare_models import load_dual_condition_qa_by_model_count
except ImportError as e:
    print(f"Cannot import compare_models.py: {e}")
    sys.exit(1)


def do_filter_min4(results_dir: Path, new_results_dir: Path, min_models: int = 4):
    """Filter QA ids that satisfy dual conditions across min_models models."""
    print(f"--- Step 1: Extracting QA ids satisfying min_models={min_models} from {results_dir} ---")
    filtered_qa_ids, _ , _ = load_dual_condition_qa_by_model_count(results_dir, min_models=min_models)
    filtered_qa_ids = set(filtered_qa_ids) # Convert to set for expansion
    
    # Force include all cross-doc questions regardless of model count
    print(f"Scanning for all 'cross-doc' questions to force include...")
    cross_doc_count = 0
    for res_file in results_dir.rglob('results.json'):
        if 'results_min4' in res_file.parts: continue
        try:
            with open(res_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for qa in data.get('qa_results', []):
                if qa.get('reasoning_type') == 'cross-doc':
                    qid = qa.get('question_id')
                    if qid and qid not in filtered_qa_ids:
                        filtered_qa_ids.add(qid)
                        cross_doc_count += 1
        except: continue
    
    print(f"Initial filter: {len(filtered_qa_ids) - cross_doc_count} IDs. Force included {cross_doc_count} additional 'cross-doc' IDs.")
    print(f"Total processing IDs: {len(filtered_qa_ids)}")
    
    if not filtered_qa_ids:
        print("No QA IDs to process.")
        return False
        
    if new_results_dir.exists():
        print(f"Removing existing directory: {new_results_dir}")
        shutil.rmtree(new_results_dir)
        
    print(f"Creating new directory: {new_results_dir}")
    new_results_dir.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in ['visualize_results', 'model_comparison', 'results_min4']:
            continue
            
        model_name = model_dir.name
        new_model_dir = new_results_dir / model_name
        
        for category_dir in model_dir.iterdir():
            if not category_dir.is_dir(): continue
            
            category_name = category_dir.name
            new_category_dir = new_model_dir / category_name
            
            for mode_dir in category_dir.iterdir():
                if not mode_dir.is_dir(): continue
                
                mode_name = mode_dir.name
                new_mode_dir = new_category_dir / mode_name
                
                results_file = mode_dir / 'results.json'
                if results_file.exists():
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except:
                        continue
                        
                    if 'qa_results' in data:
                        new_qa_results = [qa for qa in data['qa_results'] if qa.get('question_id') in filtered_qa_ids]
                        if new_qa_results:
                            new_data = data.copy()
                            new_data['qa_results'] = new_qa_results
                            if 'summary' in new_data:
                                new_data['summary']['total_questions'] = len(new_qa_results)
                                new_data['summary']['successful_questions'] = len([q for q in new_qa_results if not q.get('error')])
                            
                            new_mode_dir.mkdir(parents=True, exist_ok=True)
                            with open(new_mode_dir / 'results.json', 'w', encoding='utf-8') as f:
                                json.dump(new_data, f, ensure_ascii=False, indent=2)
                            
                            total_copied += len(new_qa_results)
                            
    print(f"Successfully filtered and saved QA data into {new_results_dir}")
    return True


def do_recalc_metrics(results_dir: Path):
    """Recalculate summary metrics for all results in the new directory."""
    print(f"\n--- Step 2: Recalculating Metrics in {results_dir} ---")
    for results_file in results_dir.rglob('results.json'):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            qa_results = data.get('qa_results', [])
            if not qa_results: continue
            
            num_questions = len(qa_results)
            if 'metadata' in data:
                data['metadata']['num_questions_completed'] = num_questions
                data['metadata']['num_questions_total'] = num_questions
            
            metric_sums = {}
            metric_counts = {}
            total_time = 0.0
            total_iterations = 0
            total_quality_score = 0.0
            error_count = 0
            
            for qa in qa_results:
                resp_time = qa.get('response_time')
                if resp_time is not None: total_time += resp_time
                iters = qa.get('iterations')
                if iters is not None: total_iterations += iters
                q_score = qa.get('quality_score')
                if q_score is not None: total_quality_score += q_score
                if qa.get('error'): error_count += 1
                
                ragas = qa.get('ragas_scores', {})
                for k, v in ragas.items():
                    if k.startswith('_') or v is None: continue
                    if k not in metric_sums:
                        metric_sums[k] = 0.0
                        metric_counts[k] = 0
                    metric_sums[k] += v
                    metric_counts[k] += 1
                    
            avg_ragas_metrics = {}
            for k in metric_sums:
                if metric_counts[k] > 0:
                    avg_ragas_metrics[k] = metric_sums[k] / metric_counts[k]
                    
            if avg_ragas_metrics:
                avg_ragas_metrics['average_score'] = sum(avg_ragas_metrics.values()) / len(avg_ragas_metrics)
            
            valid_qa_count = num_questions - error_count
            new_performance = {
                "avg_response_time": total_time / num_questions if num_questions > 0 else 0,
                "total_time": total_time,
                "error_count": error_count,
                "avg_iterations": total_iterations / num_questions if num_questions > 0 else 0,
                "avg_quality_score": total_quality_score / valid_qa_count if valid_qa_count > 0 else 0
            }
            
            if 'summary' not in data: data['summary'] = {}
            data['summary']['num_questions_completed'] = num_questions
            data['summary']['num_questions_total'] = num_questions
            data['summary']['ragas_metrics'] = avg_ragas_metrics
            data['summary']['weighted_score'] = None
            data['summary']['performance'] = new_performance
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error processing {results_file}: {e}")
    print("Recalculation complete.")


def do_visualization(results_dir: Path, out_dir: Path):
    """Generate embeddings for questions and plot clusters/distributions."""
    print(f"\n--- Step 3: Semantic QA Visualization ---")
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        import seaborn as sns
        from sklearn.manifold import TSNE
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"Missing required package for visualization: {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_cache_path = out_dir / "embeddings_cache.pkl"
    
    unique_questions = {}
    
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
    
    # Read unique questions and categories from results_dir
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in ['visualize_results', 'model_comparison', 'visualize']: continue
        for cat_dir in model_dir.iterdir():
            if not cat_dir.is_dir(): continue
            orig_cat = cat_dir.name
            category = CATEGORY_MAP.get(orig_cat.lower(), orig_cat)
            
            for mode_dir in cat_dir.iterdir():
                res_file = mode_dir / 'results.json'
                if res_file.exists():
                    try:
                        with open(res_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for qa in data.get('qa_results', []):
                                qid = qa.get('question_id')
                                qtext = qa.get('question')
                                difficulty = qa.get('difficulty', 'unknown')
                                reasoning_type = qa.get('reasoning_type', 'unknown')
                                if qid and qtext and qid not in unique_questions:
                                    unique_questions[qid] = {
                                        'question': qtext,
                                        'category': category,
                                        'difficulty': difficulty,
                                        'reasoning_type': reasoning_type
                                    }
                    except:
                        pass

    questions = []
    categories = []
    for qid, info in unique_questions.items():
        questions.append(info['question'])
        categories.append(info['category'])
        
    if not questions:
        print("No questions found for visualization.")
        return
        
    print(f"Found {len(questions)} unique questions across {len(set(categories))} categories.")
    
    embeddings = None
    if embeddings_cache_path.exists():
        with open(embeddings_cache_path, 'rb') as f:
            cache = pickle.load(f)
            if cache.get('texts') == questions:
                embeddings = cache.get('embeddings')
                print("Loaded embeddings from cache.")
                
    if embeddings is None:
        print("Generating embeddings using sentence-transformers (all-MiniLM-L6-v2)...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            print("Loaded model from local HuggingFace cache (offline mode).")
        except Exception:
            print("Local cache not available, trying online download...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(questions, show_progress_bar=True)
        with open(embeddings_cache_path, 'wb') as f:
            pickle.dump({'embeddings': embeddings, 'texts': questions}, f)
            
    print("Applying t-SNE reduction...")
    perplexity = min(30, max(1, (len(embeddings) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, metric='cosine')
    results = tsne.fit_transform(embeddings)
    
    unique_cats = sorted(set(categories))
    
    # Colors inspired by compare_models.py
    custom_colors = ['#ADC0D1', '#9BC3C6', '#F4E5BE', '#D4D2D2', '#B8D4C8', '#E8C8B8']
    cat_to_color = {cat: custom_colors[i % len(custom_colors)] for i, cat in enumerate(unique_cats)}
    
    # -- 1. Scatter Plot --
    fig, ax = plt.subplots(figsize=(10, 8))
    for cat in unique_cats:
        mask = np.array([c == cat for c in categories])
        ax.scatter(results[mask, 0], results[mask, 1], 
                   c=[cat_to_color[cat]], label=cat, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_dir / 'min4_qa_tsne_visualization.png', dpi=300)
    plt.close()
    
    # Function to save pie charts
    def save_pie_chart(counts_dict, sorted_keys, colors, title, filename):
        _pct_fontsize = 25  # 5x enlarged percentage labels

        def _draw_pie(ax_):
            plot_colors = [colors[i % len(colors)] for i in range(len(sorted_keys))]
            counts = [counts_dict[k] for k in sorted_keys]
            wedges_, texts_, autotexts_ = ax_.pie(
                counts, labels=None, colors=plot_colors,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
                startangle=90, textprops={'fontsize': _pct_fontsize}, pctdistance=0.72,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
            )
            for wedge_, autotext_ in zip(wedges_, autotexts_):
                autotext_.set_color('black')
                autotext_.set_fontweight('bold')
            ax_.axis('equal')
            return wedges_, counts

        # --- with legend ---
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, counts = _draw_pie(ax)
        total = sum(counts)
        legend_labels = [f'{k}: {counts_dict[k]} ({(counts_dict[k]/total)*100:.1f}%)' for k in sorted_keys]
        ax.legend(wedges, legend_labels, title=title, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=300)
        plt.close()

        # --- without legend, transparent background ---
        stem, ext = filename.rsplit('.', 1)
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        fig2.patch.set_alpha(0)
        ax2.patch.set_alpha(0)
        _draw_pie(ax2)
        plt.tight_layout()
        plt.savefig(out_dir / f'{stem}_nolegend.{ext}', dpi=300, transparent=True)
        plt.close()

    # -- 2. Category Pie Chart --
    cat_counts = {cat: 0 for cat in unique_cats}
    for cat in categories: cat_counts[cat] += 1
    save_pie_chart(cat_counts, unique_cats, custom_colors, "Categories", 'min4_qa_category_pie.png')

    # -- 3. Difficulty Pie Chart --
    difficulties = [info['difficulty'] for info in unique_questions.values()]
    # Fixed order: easy, medium, hard
    diff_order = ['easy', 'medium', 'hard']
    unique_diffs = [d for d in diff_order if d in set(difficulties)]
    # Add any other unexpected values
    unique_diffs += sorted(set(difficulties) - set(diff_order))
    
    diff_counts = {d: 0 for d in unique_diffs}
    for d in difficulties: diff_counts[d] += 1
    
    # User colors: 绿(Green), 黄(Yellow), 黑(Black)
    # Green=#2A9D8F, Yellow=#E9C46A, Black=#264653
    diff_colors = ['#ADC0D1', '#9BC3C6', '#F4E5BE', '#D4D2D2']
    save_pie_chart(diff_counts, unique_diffs, diff_colors, "Difficulty", 'min4_qa_difficulty_pie.png')

    # -- 4. Reasoning Type Pie Chart --
    reasonings = [info['reasoning_type'] for info in unique_questions.values()]
    # Prefer order: single-hop, multi-hop, cross-doc
    reasoning_order = ['single-hop', 'multi-hop', 'cross-doc']
    unique_reasonings = [r for r in reasoning_order if r in set(reasonings)]
    unique_reasonings += sorted(set(reasonings) - set(reasoning_order))
    
    reasoning_counts = {r: 0 for r in unique_reasonings}
    for r in reasonings: reasoning_counts[r] += 1
    
    # Reasoning pie uses a red palette with percentage labels shown on-chart
    reasoning_colors = ['#ADC0D1', '#9BC3C6', '#F4E5BE', '#D4D2D2']
    save_pie_chart(reasoning_counts, unique_reasonings, reasoning_colors, "Reasoning Type", 'min4_qa_reasoning_pie.png')

    print(f"Visualizations saved to {out_dir}")


def main():
    base_dir = Path('/home/wzfeng/RAGPhoto/Data_Agentic_RAG/evaluation')
    results_dir = base_dir / 'results'
    filtered_dir = base_dir / 'results_min4'
    vis_out_dir = filtered_dir / 'visualize'
    
    if do_filter_min4(results_dir, filtered_dir, min_models=4):
        do_recalc_metrics(filtered_dir)
        do_visualization(filtered_dir, vis_out_dir)

if __name__ == '__main__':
    main()
