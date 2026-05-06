#!/usr/bin/env python3
"""
Repair Null Metrics Script
Re-computes null evaluation metrics in existing results.json files.
Skips metrics that are N/A for direct_llm mode.

# 先预览要修复的内容
python3 repair_null_metrics.py --dry-run

# 实际修复
python3 repair_null_metrics.py

# 修复特定目录
python3 repair_null_metrics.py --dir results/computational

# 仅重新计算平均值（不修复 null）
python3 repair_null_metrics.py --recalc-only --dir results/computational

# 使用示例
python3 repair_null_metrics.py \
    --eval-api-key "your-eval-api-key" \
    --answer-api-key "your-answer-api-key" \
    --dir results/claude-4.5-mimo/characterization

# 如果只需要评估模型的 API Key（修复指标主要用这个）
python3 repair_null_metrics.py \
    --eval-api-key "" \
    --dir results/characterization

python3 repair_null_metrics.py \
    --eval-api-key "" \
    --eval-api-base "" \
    --eval-model "mimo-v2-flash-free" \
    --dir results/claude-4.5-mimo/computational

# 先预览
python3 repair_null_metrics.py \
    --eval-api-key "" \
    --dry-run \
    --dir results/claude-4.5-mimo/computational

# ========== 新增功能 ==========

# 跳过 context_precision，只计算 context_recall 和 answer_similarity
python3 repair_null_metrics.py \
    --skip-metrics context_precision \
    --dir results/computational

# 仅计算指定的指标（逗号分隔）
python3 repair_null_metrics.py \
    --include-metrics context_recall,answer_similarity \
    --dir results/computational

# 并行计算多个指标（加快速度，但注意 API 速率限制）
python3 repair_null_metrics.py \
    --parallel \
    --skip-metrics context_precision \
    --dir results/computational

# 组合使用：并行 + 跳过 context_precision
python3 repair_null_metrics.py \
    --eval-api-key "your-key" \
    --parallel \
    --skip-metrics context_precision \
    --dir results/computational

# ========== 多QA并行计算 ==========

# 对一个文件中的多个QA并行计算指定指标（默认4个QA同时计算）
python3 repair_null_metrics.py \
    --parallel-qa \
    --include-metrics context_precision \
    --dir results/computational

# 指定并行QA数量（例如同时计算8个QA）
python3 repair_null_metrics.py \
    --parallel-qa 8 \
    --include-metrics context_recall,answer_similarity \
    --dir results/computational

# 组合使用：多QA并行 + 跳过难算的指标
python3 repair_null_metrics.py \
    --eval-api-key "your-key" \
    --parallel-qa 4 \
    --skip-metrics context_precision \
    --dir results/computational

# ========== Context 数量限制 ==========

# 限制 context_precision 只评估前 5 个 context chunks（默认）
python3 repair_null_metrics.py \
    --eval-api-key "your-key" \
    --include-metrics context_precision \
    --dir results/computational

# 自定义 context 数量限制（例如只评估前 3 个）
python3 repair_null_metrics.py \
    --eval-api-key "your-key" \
    --max-contexts 3 \
    --include-metrics context_precision \
    --dir results/computational

# 不限制 context 数量（评估所有）
python3 repair_null_metrics.py \
    --eval-api-key "your-key" \
    --max-contexts 0 \
    --include-metrics context_precision \
    --dir results/computational

"""

import json
import os
import sys
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random

# Suppress RAGAS/httpx event loop warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Event loop is closed.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*coroutine.*was never awaited.*')
warnings.filterwarnings('ignore', message='.*Task exception was never retrieved.*')

try:
    from ragas import RunConfig
    # Configure RAGAS RunConfig for better performance and timeout handling
    RAGAS_RUN_CONFIG = RunConfig(
        timeout=300,       # 5 minutes
        max_retries=3,
        max_wait=30,
        max_workers=1      # Sequential execution to avoid rate limits
    )
except ImportError:
    RAGAS_RUN_CONFIG = None

# Add evaluation directory to path
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from custom_metrics import CustomFaithfulness, CustomAnswerCorrectness, CustomAnswerRelevancy

def load_env():
    """Load environment variables from .env file"""
    env_file = eval_dir / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def get_eval_model_config():
    """Get evaluation model configuration from environment (fallback)"""
    return {
        'model': os.getenv('EVAL_LLM_MODEL', 'glm-4.7'),
        'api_key': os.getenv('EVAL_LLM_API_KEY', ''),
        'api_base': os.getenv('EVAL_LLM_BASE_URL', ''),
        'temperature': 0.0
    }

def get_embedding_config():
    """Get embedding configuration from environment"""
    return {
        'model': os.getenv('EVAL_EMBEDDING_MODEL', 'text-embedding-3-small'),
        'api_key': os.getenv('EVAL_EMBEDDING_API_KEY', ''),
        'api_base': os.getenv('EVAL_EMBEDDING_BASE_URL', '')
    }

def get_eval_config_from_metadata(metadata: dict):
    """
    Extract evaluation model configuration from results file metadata.
    Uses the original configuration that was used during evaluation.
    Command line args (via env vars) can override metadata settings.
    """
    eval_info = metadata.get('evaluation', {})
    
    # Get model name (ragas_model is for faithfulness/correctness evaluation)
    # Command line --eval-model overrides metadata
    model = os.getenv('EVAL_LLM_MODEL') or eval_info.get('ragas_model') or eval_info.get('fact_check_model')
    
    # Command line --eval-api-base overrides metadata
    api_base = os.getenv('EVAL_LLM_BASE_URL') or eval_info.get('api_base', '')
    temperature = eval_info.get('temperature', 0.0)
    
    if not model:
        print("  ⚠️ No evaluation model found in metadata or command line, using environment config")
        return None
    
    # Get API key from environment - EVAL_LLM_API_KEY is set by --eval-api-key
    # Always check EVAL_LLM_API_KEY first (command line takes priority)
    api_key = os.getenv('EVAL_LLM_API_KEY', '')
    
    # If not set, try api_base specific keys
    if not api_key:
        if 'bigmodel.cn' in api_base:
            api_key = os.getenv('ZHIPU_API_KEY', '')
        elif 'dmxapi' in api_base:
            api_key = os.getenv('ANSWER_LLM_API_KEY', os.getenv('DMX_API_KEY', ''))
        elif 'openai' in api_base:
            api_key = os.getenv('OPENAI_API_KEY', '')
    
    config = {
        'model': model,
        'api_key': api_key,
        'api_base': api_base,
        'temperature': temperature
    }
    
    # Show if we're using overrides
    override_msg = ""
    if os.getenv('EVAL_LLM_MODEL'):
        override_msg += " (model overridden)"
    if os.getenv('EVAL_LLM_BASE_URL'):
        override_msg += " (api_base overridden)"
    
    print(f"  📋 Using config{override_msg}:")
    print(f"     Model: {model}")
    print(f"     API Base: {api_base}")
    
    return config

def repair_null_metrics(results_path: str, dry_run: bool = False, recalc_only: bool = False,
                         custom_only: bool = False, ragas_only: bool = False,
                         include_metrics: list = None, skip_metrics: list = None,
                         parallel: bool = False, parallel_qa: int = 0,
                         max_contexts: int = 5, order: str = 'forward'):
    """
    Repair null metrics in a results.json file.
    每计算完一个指标就立即保存，避免中途出错导致已计算的结果丢失。
    
    Args:
        results_path: Path to results.json
        recalc_only: If True, only recalculate averages without repairing
        dry_run: If True, only analyze without making changes
        custom_only: If True, only repair custom metrics (faithfulness, answer_correctness)
        ragas_only: If True, only repair RAGAS metrics (context_precision, context_recall)
        include_metrics: List of specific metrics to repair (overrides custom_only/ragas_only)
        skip_metrics: List of metrics to skip
        parallel: If True, compute metrics in parallel for each QA item
        parallel_qa: Number of QA items to process in parallel (0 = sequential)
        max_contexts: Max number of context chunks for context_precision (0 = no limit, default: 5)
        order: Processing order - 'forward' (default), 'reverse', or 'random'
    """
    print(f"\n{'='*60}")
    print(f"Processing: {results_path}")
    print(f"{'='*60}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    mode = data.get('metadata', {}).get('mode', 'unknown')
    qa_results = data.get('qa_results', [])
    
    # For direct_llm mode, only certain metrics can be computed (no contexts available)
    # - CAN compute: answer_correctness, answer_similarity, answer_relevancy
    # - CANNOT compute: context_recall, context_precision, faithfulness (requires contexts)
    is_direct_llm = (mode == 'direct_llm')
    if is_direct_llm:
        print(f"  📋 Mode: {mode} - will only compute metrics that don't require contexts")
        print(f"     ✅ Can compute: answer_correctness, answer_similarity, answer_relevancy")
        print(f"     ❌ Cannot compute: context_recall, context_precision, faithfulness")
    
    # All available metrics
    all_metrics = ['faithfulness', 'answer_correctness', 'answer_relevancy',
                   'context_precision', 'context_recall', 'answer_similarity']
    
    # Determine which metrics to process
    if include_metrics:
        # Use explicit include list
        metrics_to_process = set(include_metrics)
        print(f"  📋 Including only specified metrics: {list(metrics_to_process)}")
    elif custom_only:
        metrics_to_process = {'faithfulness', 'answer_correctness', 'answer_relevancy'}
    elif ragas_only:
        metrics_to_process = {'context_precision', 'context_recall', 'answer_similarity'}
    else:
        metrics_to_process = set(all_metrics)
    
    # Apply skip list
    if skip_metrics:
        metrics_to_process -= set(skip_metrics)
        print(f"  📋 Skipping metrics: {skip_metrics}")
    
    print(f"  📋 Metrics to process: {list(metrics_to_process)}")
    
    if parallel:
        print(f"  ⚡ Parallel mode enabled (multiple metrics per QA)")
    if parallel_qa > 0:
        print(f"  ⚡ Parallel QA mode enabled: {parallel_qa} QA items at a time")
    if max_contexts > 0:
        print(f"  📦 context_precision: 仅评估前 {max_contexts} 个 context chunks")
    else:
        print(f"  📦 context_precision: 评估所有 context chunks")

    # Create backup at the beginning (only if not dry run)
    backup_path = results_path + '.backup'
    if not dry_run and not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Backup created: {backup_path}")
    
    def save_results():
        """Helper function to save results immediately after each metric repair"""
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  💾 Results saved to: {results_path}")
    
    # Get evaluation config from metadata first, fallback to environment
    metadata = data.get('metadata', {})
    eval_config = get_eval_config_from_metadata(metadata)
    if eval_config is None:
        eval_config = get_eval_model_config()
        print(f"  📋 Using fallback config from environment:")
        print(f"     Model: {eval_config['model']}")
        print(f"     API Base: {eval_config['api_base']}")
    
    # Initialize metrics
    faithfulness_metric = CustomFaithfulness(model_config=eval_config, verbose=True)
    correctness_metric = CustomAnswerCorrectness(model_config=eval_config, verbose=True)
    
    # Get embedding configuration from environment for CustomAnswerRelevancy
    embedding_config = get_embedding_config()
    if embedding_config['api_key'] and embedding_config['api_base']:
        print(f"  📋 Embedding config:")
        print(f"     Model: {embedding_config['model']}")
        print(f"     API Base: {embedding_config['api_base']}")
    
    # Pass embedding_config to CustomAnswerRelevancy for semantic similarity
    relevancy_metric = CustomAnswerRelevancy(
        model_config=eval_config, 
        embedding_config=embedding_config if embedding_config['api_key'] else None,
        verbose=True
    )
    
    repaired_count = 0
    
    # Thread-safe lock for saving results
    save_lock = threading.Lock()
    
    def save_results_safe():
        """Thread-safe save results"""
        with save_lock:
            with open(results_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    # ========== Parallel QA Processing ==========
    if parallel_qa > 0 and not dry_run:
        print(f"\n⚡ Starting parallel QA processing with {parallel_qa} workers...")
        
        # First pass: collect all QA items that need repair
        qa_repair_tasks = []  # List of (qa_index, qa, needs_repair)
        
        for i, qa in enumerate(qa_results):
            scores = qa.get('ragas_scores', {})
            question = qa.get('question', '')
            answer = qa.get('generated_answer', '')
            ground_truth = qa.get('ground_truth', '')
            contexts = qa.get('contexts', [])
            
            needs_repair = []
            
            if 'faithfulness' in metrics_to_process:
                if scores.get('faithfulness') is None and contexts:
                    needs_repair.append('faithfulness')
            if 'answer_correctness' in metrics_to_process:
                if scores.get('answer_correctness') is None and answer and ground_truth:
                    needs_repair.append('answer_correctness')
            if 'answer_relevancy' in metrics_to_process:
                if scores.get('answer_relevancy') is None and answer and question:
                    needs_repair.append('answer_relevancy')
            if 'context_recall' in metrics_to_process:
                if scores.get('context_recall') is None and ground_truth and contexts:
                    needs_repair.append('context_recall')
            if 'context_precision' in metrics_to_process:
                if scores.get('context_precision') is None and contexts:
                    needs_repair.append('context_precision')
            if 'answer_similarity' in metrics_to_process:
                if scores.get('answer_similarity') is None and answer and ground_truth:
                    needs_repair.append('answer_similarity')
            
            if needs_repair:
                qa_repair_tasks.append((i, qa, needs_repair))
        
        print(f"  📋 Found {len(qa_repair_tasks)} QA items needing repair")
        
        # Apply processing order
        if order == 'reverse':
            qa_repair_tasks = qa_repair_tasks[::-1]
            print(f"  🔄 Processing order: REVERSE (from end to start)")
        elif order == 'random':
            random.shuffle(qa_repair_tasks)
            print(f"  🔀 Processing order: RANDOM (shuffled)")
        else:
            print(f"  ➡️ Processing order: FORWARD (default)")
        
        if qa_repair_tasks:
            # Define a function to process a single QA item
            def process_single_qa(task):
                """Process a single QA item and return repair results"""
                qa_idx, qa, needs_repair = task
                scores = qa.get('ragas_scores', {})
                question = qa.get('question', '')
                answer = qa.get('generated_answer', '')
                ground_truth = qa.get('ground_truth', '')
                contexts = qa.get('contexts', [])
                question_id = qa.get('question_id', 'unknown')
                
                results = []  # List of (metric_name, new_score, error)
                
                for metric in needs_repair:
                    try:
                        if metric == 'faithfulness':
                            result = faithfulness_metric.compute(answer=answer, contexts=contexts)
                            new_score = result.get('custom_faithfulness')
                            results.append(('faithfulness', new_score, result.get('_error') if new_score is None else None))
                        
                        elif metric == 'answer_correctness':
                            result = correctness_metric.compute(question=question, answer=answer, ground_truth=ground_truth)
                            new_score = result.get('custom_answer_correctness')
                            results.append(('answer_correctness', new_score, result.get('_error') if new_score is None else None))
                        
                        elif metric == 'answer_relevancy':
                            result = relevancy_metric.compute(question, answer)
                            new_score = result.get('custom_answer_relevancy')
                            if new_score is not None:
                                new_score = float(new_score)
                            results.append(('answer_relevancy', new_score, result.get('_error') if new_score is None else None))
                        
                        elif metric == 'context_recall':
                            from ragas.metrics import context_recall
                            from datasets import Dataset
                            from ragas import evaluate
                            from langchain_openai import ChatOpenAI
                            
                            eval_data = {
                                'question': [question],
                                'answer': [answer],
                                'contexts': [[c for c in contexts if c and c.strip()]],
                                'ground_truth': [ground_truth]
                            }
                            dataset = Dataset.from_dict(eval_data)
                            ragas_llm = ChatOpenAI(
                                model=eval_config['model'],
                                openai_api_key=eval_config['api_key'],
                                openai_api_base=eval_config['api_base'],
                                temperature=0.0
                            )
                            result = evaluate(dataset, metrics=[context_recall], llm=ragas_llm, run_config=RAGAS_RUN_CONFIG)
                            raw_score = result['context_recall']
                            new_score = raw_score[0] if isinstance(raw_score, list) else raw_score
                            if new_score is not None and not (isinstance(new_score, float) and new_score != new_score):
                                results.append(('context_recall', float(new_score), None))
                            else:
                                results.append(('context_recall', None, 'RAGAS returned NaN'))
                        
                        elif metric == 'context_precision':
                            from ragas.metrics import context_precision
                            from datasets import Dataset
                            from ragas import evaluate
                            from langchain_openai import ChatOpenAI
                            
                            # Limit context chunks for context_precision
                            filtered_contexts = [c for c in contexts if c and c.strip()]
                            if max_contexts > 0:
                                filtered_contexts = filtered_contexts[:max_contexts]
                            
                            eval_data = {
                                'question': [question],
                                'answer': [answer],
                                'contexts': [filtered_contexts],
                                'ground_truth': [ground_truth]
                            }
                            dataset = Dataset.from_dict(eval_data)
                            ragas_llm = ChatOpenAI(
                                model=eval_config['model'],
                                openai_api_key=eval_config['api_key'],
                                openai_api_base=eval_config['api_base'],
                                temperature=0.0
                            )
                            result = evaluate(dataset, metrics=[context_precision], llm=ragas_llm, run_config=RAGAS_RUN_CONFIG)
                            raw_score = result['context_precision']
                            new_score = raw_score[0] if isinstance(raw_score, list) else raw_score
                            if new_score is not None and not (isinstance(new_score, float) and new_score != new_score):
                                results.append(('context_precision', float(new_score), None))
                            else:
                                results.append(('context_precision', None, 'RAGAS returned NaN'))
                        
                        elif metric == 'answer_similarity':
                            from ragas.metrics import answer_similarity
                            from datasets import Dataset
                            from ragas import evaluate
                            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                            
                            eval_data = {
                                'question': [question],
                                'answer': [answer],
                                'contexts': [[c for c in contexts if c and c.strip()] if contexts else []],
                                'ground_truth': [ground_truth]
                            }
                            dataset = Dataset.from_dict(eval_data)
                            ragas_llm = ChatOpenAI(
                                model=eval_config['model'],
                                openai_api_key=eval_config['api_key'],
                                openai_api_base=eval_config['api_base'],
                                temperature=0.0
                            )
                            emb_config = get_embedding_config()
                            ragas_embeddings = OpenAIEmbeddings(
                                model=emb_config['model'],
                                openai_api_key=emb_config['api_key'] or eval_config['api_key'],
                                openai_api_base=emb_config['api_base'] or eval_config['api_base']
                            )
                            result = evaluate(dataset, metrics=[answer_similarity], llm=ragas_llm, 
                                            embeddings=ragas_embeddings, run_config=RAGAS_RUN_CONFIG)
                            raw_score = result['answer_similarity']
                            new_score = raw_score[0] if isinstance(raw_score, list) else raw_score
                            if new_score is not None and not (isinstance(new_score, float) and new_score != new_score):
                                results.append(('answer_similarity', float(new_score), None))
                            else:
                                results.append(('answer_similarity', None, 'RAGAS returned NaN'))
                    
                    except Exception as e:
                        results.append((metric, None, str(e)))
                
                return (qa_idx, question_id, results)
            
            # Process QAs in parallel
            completed = 0
            with ThreadPoolExecutor(max_workers=parallel_qa) as executor:
                futures = {executor.submit(process_single_qa, task): task for task in qa_repair_tasks}
                
                for future in as_completed(futures):
                    try:
                        qa_idx, question_id, results = future.result()
                        completed += 1
                        
                        # Update scores
                        qa = qa_results[qa_idx]
                        scores = qa.get('ragas_scores', {})
                        
                        print(f"\n[{completed}/{len(qa_repair_tasks)}] {question_id}")
                        
                        for metric_name, new_score, error in results:
                            if new_score is not None:
                                scores[metric_name] = new_score
                                nan_metrics = scores.get('_nan_metrics', [])
                                if metric_name in nan_metrics:
                                    nan_metrics.remove(metric_name)
                                    scores['_nan_metrics'] = nan_metrics
                                print(f"  ✅ {metric_name}: {new_score:.4f}")
                                repaired_count += 1
                            else:
                                print(f"  ⚠️ {metric_name} failed: {error}")
                        
                        # Save after each QA completion (thread-safe)
                        save_results_safe()
                        
                    except Exception as e:
                        task = futures[future]
                        print(f"  ❌ QA processing error: {e}")
            
            print(f"\n✅ Parallel QA processing complete. Repaired {repaired_count} metrics.")
    
    # ========== Sequential Processing (original behavior) ==========
    else:
        # Create processing order
        qa_indices = list(range(len(qa_results)))
        if order == 'reverse':
            qa_indices = qa_indices[::-1]
            print(f"  🔄 Processing order: REVERSE (from end to start)")
        elif order == 'random':
            random.shuffle(qa_indices)
            print(f"  🔀 Processing order: RANDOM (shuffled)")
        else:
            print(f"  ➡️ Processing order: FORWARD (default)")
        
        for idx in qa_indices:
            qa = qa_results[idx]
            i = idx  # Keep original index for display
            scores = qa.get('ragas_scores', {})
            question = qa.get('question', '')
            answer = qa.get('generated_answer', '')
            ground_truth = qa.get('ground_truth', '')
            contexts = qa.get('contexts', [])
            
            needs_repair = []
            
            # Define metric categories by implementation:
            # - Custom metrics: Use Custom* classes (faithfulness, answer_correctness, answer_relevancy)
            # - RAGAS metrics: Use RAGAS library (context_precision, context_recall, answer_similarity)
            
            # Check for null faithfulness (Custom metric - CustomFaithfulness)
            if 'faithfulness' in metrics_to_process:
                if scores.get('faithfulness') is None and contexts:
                    needs_repair.append('faithfulness')
            
            # Check for null answer_correctness (Custom metric - CustomAnswerCorrectness)
            if 'answer_correctness' in metrics_to_process:
                if scores.get('answer_correctness') is None and answer and ground_truth:
                    needs_repair.append('answer_correctness')
            
            # Check for null answer_relevancy (Custom metric - CustomAnswerRelevancy)
            if 'answer_relevancy' in metrics_to_process:
                if scores.get('answer_relevancy') is None and answer and question:
                    needs_repair.append('answer_relevancy')
            
            # Check for null context_recall (RAGAS library metric)
            if 'context_recall' in metrics_to_process:
                if scores.get('context_recall') is None and ground_truth and contexts:
                    needs_repair.append('context_recall')
            
            # Check for null context_precision (RAGAS library metric)
            if 'context_precision' in metrics_to_process:
                if scores.get('context_precision') is None and contexts:
                    needs_repair.append('context_precision')
            
            # Check for null answer_similarity (RAGAS library metric)
            if 'answer_similarity' in metrics_to_process:
                if scores.get('answer_similarity') is None and answer and ground_truth:
                    needs_repair.append('answer_similarity')
            
            if not needs_repair:
                continue
            
            print(f"\n[{i+1}/{len(qa_results)}] {qa.get('question_id', 'unknown')}")
            print(f"  Null metrics: {needs_repair}")
            
            if dry_run:
                print(f"  [DRY RUN] Would repair: {needs_repair}")
                repaired_count += len(needs_repair)
                continue
            
            # Define metric computation functions for parallel execution
            def compute_faithfulness():
                """Compute faithfulness metric"""
                try:
                    result = faithfulness_metric.compute(
                        answer=answer,
                        contexts=contexts
                    )
                    new_score = result.get('custom_faithfulness')
                    if new_score is not None:
                        return ('faithfulness', new_score, None)
                    else:
                        return ('faithfulness', None, result.get('_error'))
                except Exception as e:
                    return ('faithfulness', None, str(e))
            
            def compute_answer_correctness():
                """Compute answer_correctness metric"""
                try:
                    result = correctness_metric.compute(
                        question=question,
                        answer=answer,
                        ground_truth=ground_truth
                    )
                    new_score = result.get('custom_answer_correctness')
                    if new_score is not None:
                        return ('answer_correctness', new_score, None)
                    else:
                        return ('answer_correctness', None, result.get('_error'))
                except Exception as e:
                    return ('answer_correctness', None, str(e))
            
            def compute_answer_relevancy():
                """Compute answer_relevancy metric"""
                try:
                    result = relevancy_metric.compute(question, answer)
                    new_score = result.get('custom_answer_relevancy')
                    if new_score is not None:
                        return ('answer_relevancy', float(new_score), None)
                    else:
                        return ('answer_relevancy', None, result.get('_error'))
                except Exception as e:
                    return ('answer_relevancy', None, str(e))
            
            def compute_context_recall():
                """Compute context_recall metric using RAGAS"""
                try:
                    from ragas.metrics import context_recall
                    from datasets import Dataset
                    from ragas import evaluate
                    from langchain_openai import ChatOpenAI
                    
                    eval_data = {
                        'question': [question],
                        'answer': [answer],
                        'contexts': [[c for c in contexts if c and c.strip()]],
                        'ground_truth': [ground_truth]
                    }
                    dataset = Dataset.from_dict(eval_data)
                    
                    ragas_llm = ChatOpenAI(
                        model=eval_config['model'],
                        openai_api_key=eval_config['api_key'],
                        openai_api_base=eval_config['api_base'],
                        temperature=0.0
                    )
                    
                    result = evaluate(dataset, metrics=[context_recall], llm=ragas_llm, run_config=RAGAS_RUN_CONFIG)
                    raw_score = result['context_recall']
                    if isinstance(raw_score, list):
                        new_score = raw_score[0] if raw_score else None
                    else:
                        new_score = raw_score
                    
                    if new_score is not None and not (isinstance(new_score, float) and new_score != new_score):
                        return ('context_recall', float(new_score), None)
                    else:
                        return ('context_recall', None, 'RAGAS returned NaN')
                except Exception as e:
                    return ('context_recall', None, str(e))
            
            def compute_context_precision():
                """Compute context_precision metric using RAGAS"""
                try:
                    from ragas.metrics import context_precision
                    from datasets import Dataset
                    from ragas import evaluate
                    from langchain_openai import ChatOpenAI
                    
                    # Limit context chunks for context_precision (first 5 only)
                    filtered_contexts = contexts[:5]
                    
                    eval_data = {
                        'question': [question],
                        'answer': [answer],
                        'contexts': [filtered_contexts],
                        'ground_truth': [ground_truth]
                    }
                    dataset = Dataset.from_dict(eval_data)
                    
                    ragas_llm = ChatOpenAI(
                        model=eval_config['model'],
                        openai_api_key=eval_config['api_key'],
                        openai_api_base=eval_config['api_base'],
                        temperature=0.0
                    )
                    
                    result = evaluate(dataset, metrics=[context_precision], llm=ragas_llm, run_config=RAGAS_RUN_CONFIG)
                    raw_score = result['context_precision']
                    if isinstance(raw_score, list):
                        new_score = raw_score[0] if raw_score else None
                    else:
                        new_score = raw_score
                    
                    if new_score is not None and not (isinstance(new_score, float) and new_score != new_score):
                        return ('context_precision', float(new_score), None)
                    else:
                        return ('context_precision', None, 'RAGAS returned NaN')
                except Exception as e:
                    return ('context_precision', None, str(e))
            
            def compute_answer_similarity():
                """Compute answer_similarity metric using RAGAS"""
                try:
                    from ragas.metrics import answer_similarity
                    from datasets import Dataset
                    from ragas import evaluate
                    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                    
                    eval_data = {
                        'question': [question],
                        'answer': [answer],
                        'contexts': [[c for c in contexts if c and c.strip()] if contexts else []],
                        'ground_truth': [ground_truth]
                    }
                    dataset = Dataset.from_dict(eval_data)
                    
                    ragas_llm = ChatOpenAI(
                        model=eval_config['model'],
                        openai_api_key=eval_config['api_key'],
                        openai_api_base=eval_config['api_base'],
                        temperature=0.0
                    )
                    
                    emb_config = get_embedding_config()
                    embedding_api_key = emb_config['api_key'] or eval_config['api_key']
                    embedding_api_base = emb_config['api_base'] or eval_config['api_base']
                    embedding_model = emb_config['model']
                    
                    ragas_embeddings = OpenAIEmbeddings(
                        model=embedding_model,
                        openai_api_key=embedding_api_key,
                        openai_api_base=embedding_api_base
                    )
                    
                    result = evaluate(
                        dataset, 
                        metrics=[answer_similarity], 
                        llm=ragas_llm, 
                        embeddings=ragas_embeddings,
                        run_config=RAGAS_RUN_CONFIG
                    )
                    
                    raw_score = result['answer_similarity']
                    if isinstance(raw_score, list):
                        new_score = raw_score[0] if raw_score else None
                    else:
                        new_score = raw_score
                    
                    if new_score is not None and not (isinstance(new_score, float) and new_score != new_score):
                        return ('answer_similarity', float(new_score), None)
                    else:
                        return ('answer_similarity', None, 'RAGAS returned NaN')
                except Exception as e:
                    return ('answer_similarity', None, str(e))
            
            # Map metric names to compute functions
            metric_funcs = {
                'faithfulness': compute_faithfulness,
                'answer_correctness': compute_answer_correctness,
                'answer_relevancy': compute_answer_relevancy,
                'context_recall': compute_context_recall,
                'context_precision': compute_context_precision,
                'answer_similarity': compute_answer_similarity
            }
            
            # Compute metrics (parallel or sequential)
            if parallel and len(needs_repair) > 1:
                # Parallel execution
                print(f"  ⚡ Computing {len(needs_repair)} metrics in parallel...")
                results_list = []
                with ThreadPoolExecutor(max_workers=min(len(needs_repair), 4)) as executor:
                    futures = {executor.submit(metric_funcs[m]): m for m in needs_repair}
                    for future in as_completed(futures):
                        metric_name = futures[future]
                        try:
                            result = future.result()
                            results_list.append(result)
                        except Exception as e:
                            results_list.append((metric_name, None, str(e)))
            
                # Process results and save once
                for metric_name, new_score, error in results_list:
                    if new_score is not None:
                        scores[metric_name] = new_score
                        nan_metrics = scores.get('_nan_metrics', [])
                        if metric_name in nan_metrics:
                            nan_metrics.remove(metric_name)
                            scores['_nan_metrics'] = nan_metrics
                        print(f"  ✅ {metric_name} repaired: {new_score:.4f}")
                        repaired_count += 1
                    else:
                        print(f"  ⚠️ {metric_name} failed: {error}")
                
                # Save once after all parallel computations
                save_results()
            else:
                # Sequential execution (original behavior)
                # Repair faithfulness
                if 'faithfulness' in needs_repair:
                    try:
                        result = faithfulness_metric.compute(
                            answer=answer,
                            contexts=contexts
                        )
                        new_score = result.get('custom_faithfulness')
                        if new_score is not None:
                            scores['faithfulness'] = new_score
                            # Remove from _nan_metrics if present
                            nan_metrics = scores.get('_nan_metrics', [])
                            if 'faithfulness' in nan_metrics:
                                nan_metrics.remove('faithfulness')
                                scores['_nan_metrics'] = nan_metrics
                            print(f"  ✅ faithfulness repaired: {new_score}")
                            repaired_count += 1
                            # 立即保存结果
                            save_results()
                        else:
                            print(f"  ⚠️ faithfulness still failed: {result.get('_error')}")
                    except Exception as e:
                        print(f"  ❌ faithfulness error: {e}")
                
                # Repair answer_correctness
                if 'answer_correctness' in needs_repair:
                    try:
                        result = correctness_metric.compute(
                            question=question,
                            answer=answer,
                            ground_truth=ground_truth
                        )
                        new_score = result.get('custom_answer_correctness')
                        if new_score is not None:
                            scores['answer_correctness'] = new_score
                            # Remove from _nan_metrics if present
                            nan_metrics = scores.get('_nan_metrics', [])
                            if 'answer_correctness' in nan_metrics:
                                nan_metrics.remove('answer_correctness')
                                scores['_nan_metrics'] = nan_metrics
                            print(f"  ✅ answer_correctness repaired: {new_score}")
                            repaired_count += 1
                            # 立即保存结果
                            save_results()
                        else:
                            print(f"  ⚠️ answer_correctness still failed: {result.get('_error')}")
                    except Exception as e:
                        print(f"  ❌ answer_correctness error: {e}")
                
                # Repair context_recall (using RAGAS library)
                if 'context_recall' in needs_repair:
                    try:
                        from ragas.metrics import context_recall
                        from datasets import Dataset
                        from ragas import evaluate
                        from langchain_openai import ChatOpenAI
                        
                        # Create dataset for RAGAS
                        eval_data = {
                            'question': [question],
                            'answer': [answer],
                            'contexts': [[c for c in contexts if c and c.strip()]],
                            'ground_truth': [ground_truth]
                        }
                        dataset = Dataset.from_dict(eval_data)
                        
                        # Initialize LLM for RAGAS
                        ragas_llm = ChatOpenAI(
                            model=eval_config['model'],
                            openai_api_key=eval_config['api_key'],
                            openai_api_base=eval_config['api_base'],
                            temperature=0.0
                        )
                        
                        # Compute context_recall using RAGAS
                        result = evaluate(dataset, metrics=[context_recall], llm=ragas_llm, run_config=RAGAS_RUN_CONFIG)
                        
                        # RAGAS returns a dict with metric name as key
                        raw_score = result['context_recall']
                        # Handle if it returns a list
                        if isinstance(raw_score, list):
                            new_score = raw_score[0] if raw_score else None
                        else:
                            new_score = raw_score
                        
                        if new_score is not None and not (isinstance(new_score, float) and new_score != new_score):  # not NaN
                            scores['context_recall'] = float(new_score)
                            # Remove from _nan_metrics if present
                            nan_metrics = scores.get('_nan_metrics', [])
                            if 'context_recall' in nan_metrics:
                                nan_metrics.remove('context_recall')
                                scores['_nan_metrics'] = nan_metrics
                            print(f"  ✅ context_recall repaired (RAGAS): {new_score:.4f}")
                            repaired_count += 1
                            # 立即保存结果
                            save_results()
                        else:
                            print(f"  ⚠️ context_recall still failed: RAGAS returned NaN")
                    except Exception as e:
                        print(f"  ❌ context_recall error (RAGAS): {e}")
                
                # Repair context_precision (using RAGAS library)
                if 'context_precision' in needs_repair:
                    try:
                        from ragas.metrics import context_precision
                        from datasets import Dataset
                        from ragas import evaluate
                        from langchain_openai import ChatOpenAI
                        
                        # Limit context chunks for context_precision (first 5 only)
                        filtered_contexts = contexts[:5]
                        
                        # Create dataset for RAGAS
                        eval_data = {
                            'question': [question],
                            'answer': [answer],
                            'contexts': [filtered_contexts],
                            'ground_truth': [ground_truth]
                        }
                        dataset = Dataset.from_dict(eval_data)
                        
                        # Initialize LLM for RAGAS
                        ragas_llm = ChatOpenAI(
                            model=eval_config['model'],
                            openai_api_key=eval_config['api_key'],
                            openai_api_base=eval_config['api_base'],
                            temperature=0.0
                        )
                        
                        # Compute context_precision using RAGAS
                        result = evaluate(dataset, metrics=[context_precision], llm=ragas_llm, run_config=RAGAS_RUN_CONFIG)
                        
                        # RAGAS returns a dict with metric name as key
                        raw_score = result['context_precision']
                        # Handle if it returns a list
                        if isinstance(raw_score, list):
                            new_score = raw_score[0] if raw_score else None
                        else:
                            new_score = raw_score
                        
                        if new_score is not None and not (isinstance(new_score, float) and new_score != new_score):  # not NaN
                            scores['context_precision'] = float(new_score)
                            # Remove from _nan_metrics if present
                            nan_metrics = scores.get('_nan_metrics', [])
                            if 'context_precision' in nan_metrics:
                                nan_metrics.remove('context_precision')
                                scores['_nan_metrics'] = nan_metrics
                            print(f"  ✅ context_precision repaired (RAGAS): {new_score:.4f}")
                            repaired_count += 1
                            # 立即保存结果
                            save_results()
                        else:
                            print(f"  ⚠️ context_precision still failed: RAGAS returned NaN")
                    except Exception as e:
                        print(f"  ❌ context_precision error (RAGAS): {e}")
                
                # Repair answer_relevancy (using Custom metric - combines semantic + LLM)
                if 'answer_relevancy' in needs_repair:
                    try:
                        # CustomAnswerRelevancy.compute() only takes question and answer
                        result = relevancy_metric.compute(question, answer)
                        new_score = result.get('custom_answer_relevancy')
                        if new_score is not None:
                            scores['answer_relevancy'] = float(new_score)
                            # Remove from _nan_metrics if present
                            nan_metrics = scores.get('_nan_metrics', [])
                            if 'answer_relevancy' in nan_metrics:
                                nan_metrics.remove('answer_relevancy')
                                scores['_nan_metrics'] = nan_metrics
                            print(f"  ✅ answer_relevancy repaired: {new_score:.4f}")
                            repaired_count += 1
                            # 立即保存结果
                            save_results()
                        else:
                            print(f"  ⚠️ answer_relevancy still failed: {result.get('_error')}")
                    except Exception as e:
                        print(f"  ❌ answer_relevancy error: {e}")
                
                # Repair answer_similarity (using RAGAS library)
                if 'answer_similarity' in needs_repair:
                    try:
                        from ragas.metrics import answer_similarity
                        from datasets import Dataset
                        from ragas import evaluate
                        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                        
                        # Create dataset for RAGAS
                        eval_data = {
                            'question': [question],
                            'answer': [answer],
                            'contexts': [[c for c in contexts if c and c.strip()] if contexts else []],
                            'ground_truth': [ground_truth]
                        }
                        dataset = Dataset.from_dict(eval_data)
                        
                        # Initialize LLM for RAGAS
                        ragas_llm = ChatOpenAI(
                            model=eval_config['model'],
                            openai_api_key=eval_config['api_key'],
                            openai_api_base=eval_config['api_base'],
                            temperature=0.0
                        )
                        
                        # Initialize Embeddings for RAGAS
                        # Answer similarity needs embeddings to calculate cosine similarity
                        emb_config = get_embedding_config()
                        
                        # Fallback to eval config if embedding key is missing
                        embedding_api_key = emb_config['api_key'] or eval_config['api_key']
                        embedding_api_base = emb_config['api_base'] or eval_config['api_base']
                        embedding_model = emb_config['model']
                        
                        ragas_embeddings = OpenAIEmbeddings(
                            model=embedding_model,
                            openai_api_key=embedding_api_key,
                            openai_api_base=embedding_api_base
                        )
                        
                        # Compute answer_similarity using RAGAS
                        result = evaluate(
                            dataset, 
                            metrics=[answer_similarity], 
                            llm=ragas_llm, 
                            embeddings=ragas_embeddings,
                            run_config=RAGAS_RUN_CONFIG
                        )
                        
                        # RAGAS returns a dict with metric name as key
                        raw_score = result['answer_similarity']
                        # Handle if it returns a list
                        if isinstance(raw_score, list):
                            new_score = raw_score[0] if raw_score else None
                        else:
                            new_score = raw_score
                        
                        if new_score is not None and not (isinstance(new_score, float) and new_score != new_score):  # not NaN
                            scores['answer_similarity'] = float(new_score)
                            # Remove from _nan_metrics if present
                            nan_metrics = scores.get('_nan_metrics', [])
                            if 'answer_similarity' in nan_metrics:
                                nan_metrics.remove('answer_similarity')
                                scores['_nan_metrics'] = nan_metrics
                            print(f"  ✅ answer_similarity repaired (RAGAS): {new_score:.4f}")
                            repaired_count += 1
                            # 立即保存结果
                            save_results()
                        else:
                            print(f"  ⚠️ answer_similarity still failed: RAGAS returned NaN")
                    except Exception as e:
                        print(f"  ❌ answer_similarity error (RAGAS): {e}")
    
    # Recalculate _successful_count and _nan_metrics for each QA
    should_recalc = (not dry_run and repaired_count > 0) or recalc_only
    if should_recalc:
        print("\n📊 Recalculating _successful_count and _nan_metrics for all QAs...")
        
        # Metrics to check (excluding internal ones)
        core_metrics = ['context_precision', 'context_recall', 'answer_similarity', 
                       'answer_relevancy', 'faithfulness', 'answer_correctness']
        
        for qa in qa_results:
            scores = qa.get('ragas_scores', {})
            
            # Recalculate _nan_metrics
            nan_metrics = []
            successful_count = 0
            
            for metric in core_metrics:
                value = scores.get(metric)
                if value is None:
                    nan_metrics.append(metric)
                else:
                    successful_count += 1
            
            scores['_nan_metrics'] = nan_metrics
            scores['_successful_count'] = successful_count
        
        # Recalculate ragas_metrics (averages)
        print("📊 Recalculating ragas_metrics averages...")
        metrics_to_average = ['context_recall', 'answer_similarity', 'answer_relevancy', 
                              'faithfulness', 'answer_correctness']
        
        new_averages = {}
        for metric in metrics_to_average:
            values = []
            for qa in qa_results:
                v = qa.get('ragas_scores', {}).get(metric)
                if v is not None and not (isinstance(v, float) and (v != v)):  # not NaN
                    values.append(v)
            if values:
                new_averages[metric] = sum(values) / len(values)
                print(f"  {metric}: {new_averages[metric]:.4f} (n={len(values)})")
            else:
                new_averages[metric] = None
        
        # Calculate context_precision average (now 0-1 from RAGAS, no reciprocal needed)
        cp_values = []
        for qa in qa_results:
            v = qa.get('ragas_scores', {}).get('context_precision')
            if v is not None and isinstance(v, (int, float)) and not (isinstance(v, float) and v != v):
                cp_values.append(v)
        
        if cp_values:
            new_averages['context_precision'] = sum(cp_values) / len(cp_values)
            print(f"  context_precision: {new_averages['context_precision']:.4f} (n={len(cp_values)})")
        else:
            new_averages['context_precision'] = None
        
        # Calculate overall average score (all metrics including context_precision)
        avg_values = [v for k, v in new_averages.items() 
                      if v is not None and k != 'average_score']
        if avg_values:
            new_averages['average_score'] = sum(avg_values) / len(avg_values)
            print(f"  average_score: {new_averages['average_score']:.4f}")
        
        # Update data
        data['ragas_metrics'] = new_averages
        
        # Save after recalculating averages
        save_results()
        print(f"💾 Averages recalculated and saved: {results_path}")
    
    return repaired_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Repair null metrics in evaluation results')
    parser.add_argument('--dir', default='results/computational', 
                        help='Directory to scan for results.json files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only analyze without making changes')
    parser.add_argument('--recalc-only', action='store_true',
                        help='Only recalculate averages without repairing null metrics')
    parser.add_argument('--custom-only', action='store_true',
                        help='Only repair custom metrics (Custom* classes): faithfulness, answer_correctness, answer_relevancy')
    parser.add_argument('--ragas-only', action='store_true',
                        help='Only repair RAGAS library metrics: context_precision, context_recall, answer_similarity')
    parser.add_argument('--include-metrics', type=str, default=None,
                        help='Comma-separated list of specific metrics to repair (e.g., "context_recall,answer_similarity"). '
                             'Available: faithfulness, answer_correctness, answer_relevancy, context_precision, context_recall, answer_similarity')
    parser.add_argument('--skip-metrics', type=str, default=None,
                        help='Comma-separated list of metrics to skip (e.g., "context_precision"). '
                             'Useful to skip problematic metrics that block others.')
    parser.add_argument('--parallel', action='store_true',
                        help='Compute multiple metrics in parallel for each QA item. '
                             'May speed up processing but watch for API rate limits.')
    parser.add_argument('--parallel-qa', type=int, nargs='?', const=4, default=0,
                        help='Process multiple QA items in parallel. Specify number of workers (default: 4 if flag used). '
                             'E.g., --parallel-qa for 4 workers, --parallel-qa 8 for 8 workers.')
    parser.add_argument('--max-contexts', type=int, default=5,
                        help='Max number of context chunks for context_precision evaluation (default: 5). '
                             'Set to 0 to evaluate all contexts (may be slow).')
    parser.add_argument('--answer-api-key', type=str, default=None,
                        help='API key for answer generation model')
    parser.add_argument('--eval-api-key', type=str, default=None,
                        help='API key for evaluation model')
    parser.add_argument('--eval-api-base', type=str, default=None,
                        help='API base URL for evaluation model (overrides metadata)')
    parser.add_argument('--eval-model', type=str, default=None,
                        help='Model name for evaluation (overrides metadata)')
    parser.add_argument('--order', type=str, default='forward', choices=['forward', 'reverse', 'random'],
                        help='Processing order: forward (default), reverse (from end to start), random (shuffled)')
    args = parser.parse_args()
    
    # Load environment
    load_env()
    
    # If API keys are provided via command line, set them as environment variables
    if args.answer_api_key:
        os.environ['ANSWER_LLM_API_KEY'] = args.answer_api_key
        print(f"Using answer API key from command line: {args.answer_api_key[:10]}...")
    if args.eval_api_key:
        os.environ['EVAL_LLM_API_KEY'] = args.eval_api_key
        print(f"Using eval API key from command line: {args.eval_api_key[:10]}...")
    if args.eval_api_base:
        os.environ['EVAL_LLM_BASE_URL'] = args.eval_api_base
        print(f"Using eval API base from command line: {args.eval_api_base}")
    if args.eval_model:
        os.environ['EVAL_LLM_MODEL'] = args.eval_model
        print(f"Using eval model from command line: {args.eval_model}")
    
    results_dir = Path(args.dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"Scanning: {results_dir}")
    if args.recalc_only:
        print("Mode: RECALCULATE AVERAGES ONLY")
    else:
        print(f"Mode: {'DRY RUN' if args.dry_run else 'REPAIR'}")
    
    # Parse include/skip metrics
    include_metrics = None
    skip_metrics = None
    
    if args.include_metrics:
        include_metrics = [m.strip() for m in args.include_metrics.split(',')]
        print(f"Include metrics: {include_metrics}")
    if args.skip_metrics:
        skip_metrics = [m.strip() for m in args.skip_metrics.split(',')]
        print(f"Skip metrics: {skip_metrics}")
    
    if args.custom_only:
        print("Filter: CUSTOM METRICS ONLY (faithfulness, answer_correctness, answer_relevancy)")
    elif args.ragas_only:
        print("Filter: RAGAS LIBRARY METRICS ONLY (context_precision, context_recall, answer_similarity)")
    
    if args.parallel:
        print("⚡ Parallel mode (per QA metrics): ENABLED")
    if args.parallel_qa > 0:
        print(f"⚡ Parallel QA mode: {args.parallel_qa} QA items at a time")
    
    if args.max_contexts > 0:
        print(f"📦 context_precision: 仅评估前 {args.max_contexts} 个 context chunks")
    else:
        print(f"📦 context_precision: 评估所有 context chunks")
    
    total_repaired = 0
    
    # Find all results.json files
    for results_file in results_dir.rglob('results.json'):
        repaired = repair_null_metrics(
            str(results_file), 
            dry_run=args.dry_run, 
            recalc_only=args.recalc_only,
            custom_only=args.custom_only,
            ragas_only=args.ragas_only,
            include_metrics=include_metrics,
            skip_metrics=skip_metrics,
            parallel=args.parallel,
            parallel_qa=args.parallel_qa,
            max_contexts=args.max_contexts,
            order=args.order
        )
        total_repaired += repaired
    
    print(f"\n{'='*60}")
    print(f"Total metrics {'would be ' if args.dry_run else ''}repaired: {total_repaired}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
