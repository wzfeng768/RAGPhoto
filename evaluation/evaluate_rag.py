#!/usr/bin/env python3
"""
RAG System Evaluation Script
Evaluates three modes: Agentic RAG + KG, Agentic RAG (no KG), and Direct LLM
Uses RAGAS library for comprehensive evaluation metrics
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Load evaluation .env file
def load_eval_env():
    """Load environment variables from evaluation/.env"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ Loaded evaluation config from {env_file}")

load_eval_env()

# Evaluation config from .env (uses EVAL_ prefix to avoid conflicts with agentic_rag)
class EvalEnvConfig:
    """Configuration loaded from evaluation/.env"""
    # LLM Config (BigModel) - for RAGAS metrics computation
    llm_api_key = os.getenv("EVAL_LLM_API_KEY", "")
    llm_base_url = os.getenv("EVAL_LLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4")
    llm_model = os.getenv("EVAL_LLM_MODEL", "glm-4.7")
    llm_temperature = float(os.getenv("EVAL_LLM_TEMPERATURE", "0.3"))

    # Embedding Config (DMXAPI)
    embedding_api_key = os.getenv("EVAL_EMBEDDING_API_KEY", "")
    embedding_base_url = os.getenv("EVAL_EMBEDDING_BASE_URL", "https://www.dmxapi.cn/v1")
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-small")

    # Reranker Config (DMXAPI)
    reranker_api_key = os.getenv("EVAL_RERANKER_API_KEY", "")
    reranker_base_url = os.getenv("EVAL_RERANKER_BASE_URL", "https://www.dmxapi.cn/v1")
    reranker_model = os.getenv("EVAL_RERANKER_MODEL", "qwen3-reranker-8b")

eval_env = EvalEnvConfig()

# Add parent agentic_rag to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "agentic_rag"))

# Import from agentic_rag
from core.agent_orchestrator import AgentOrchestrator
from config.config import config as agentic_config

# Import evaluation config
from eval_config import eval_config

# Import Custom Metrics (to replace buggy RAGAS metrics)
try:
    from custom_metrics import (
        CustomAnswerRelevancy, 
        CustomFaithfulness, 
        CustomAnswerCorrectness
        # CustomContextPrecision removed - using RAGAS native
    )
    CUSTOM_METRICS_AVAILABLE = True
except ImportError:
    CUSTOM_METRICS_AVAILABLE = False

# Import RAGAS evaluation library
try:
    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import (
        # faithfulness,      # REMOVED - using CustomFaithfulness
        # answer_relevancy,  # REMOVED - using CustomAnswerRelevancy
        context_precision,  # RESTORED - using RAGAS native implementation
        context_recall,
        # answer_correctness, # REMOVED - using CustomAnswerCorrectness
        answer_similarity
    )
    from langchain_openai import ChatOpenAI
    
    # Configure RAGAS RunConfig for better performance
    RAGAS_RUN_CONFIG = RunConfig(
        timeout=180,       # 3 minutes per LLM call (faithfulness needs more time)
        max_retries=2,     # Fewer retries for faster failure
        max_wait=20,       # 20s between retries
        max_workers=2      # Reduce concurrent calls
    )
    
    RAGAS_AVAILABLE = True
    print("✅ RAGAS evaluation library loaded successfully")
except ImportError as e:
    print(f"⚠️  RAGAS library not installed: {e}")
    print(f"   Install with: pip install ragas datasets langchain-openai")
    print(f"   Evaluation will skip RAGAS metrics (still saves Q&A results)")
    RAGAS_AVAILABLE = False
    RAGAS_RUN_CONFIG = None

# Suppress RAGAS/httpx event loop warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Event loop is closed.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*coroutine.*was never awaited.*')
warnings.filterwarnings('ignore', message='.*Task exception was never retrieved.*')

# Import OpenAI for direct LLM evaluation
from openai import OpenAI


class RAGEvaluator:
    """
    RAG System Evaluator
    
    Evaluates three different question-answering approaches:
    1. Agentic RAG with Knowledge Graph
    2. Agentic RAG without Knowledge Graph (vector only)
    3. Direct LLM (no retrieval)
    """
    
    def __init__(self, dataset: str = 'computational', verbose: bool = True, 
                 incremental_save: bool = True, resume: bool = False,
                 answer_model_config: dict = None, eval_model_config: dict = None,
                 skip_ragas: bool = False, results_subdir: str = None,
                 generate_only: bool = False, eval_only: bool = False,
                 dry_run: bool = False):
        """
        Initialize evaluator
        
        Args:
            dataset: Dataset name to evaluate (from AVAILABLE_DATASETS)
            verbose: Enable verbose output
            incremental_save: Enable incremental saving after each question
            resume: Resume from existing progress
            answer_model_config: Configuration for answer generation model
                {
                    'model': 'gpt-4o-mini',
                    'api_key': 'sk-...',
                    'api_base': 'https://...',
                    'temperature': 0.3
                }
            eval_model_config: Configuration for evaluation model (RAGAS and fact-check)
                {
                    'model': 'gpt-4',
                    'api_key': 'sk-...',
                    'api_base': 'https://...',
                    'temperature': 0.0
                }
            skip_ragas: Skip RAGAS evaluation (only compute fact-checking)
            results_subdir: Subdirectory for results (e.g., model name for multi-model evaluation)
            generate_only: Phase 1 - Only generate answers, skip all evaluation metrics
            eval_only: Phase 2 - Only compute evaluation metrics on existing answers
        """
        self.verbose = verbose
        self.dataset = dataset
        self.eval_config = eval_config
        self.incremental_save = incremental_save
        self.resume = resume
        self.skip_ragas = skip_ragas
        self.results_subdir = results_subdir
        self.generate_only = generate_only
        self.eval_only = eval_only
        self.dry_run = dry_run
        
        # Model configurations (use provided or default)
        # answer_model_config: for answer generation, uses agentic_rag/.env
        self.answer_model_config = answer_model_config or {
            'model': agentic_config.llm_model,
            'api_key': agentic_config.openai_api_key,
            'api_base': agentic_config.openai_base_url,
            'temperature': agentic_config.llm_temperature
        }
        
        # eval_model_config: for RAGAS/metrics evaluation, uses evaluation/.env
        self.eval_model_config = eval_model_config or {
            'model': eval_env.llm_model,
            'api_key': eval_env.llm_api_key,
            'api_base': eval_env.llm_base_url,
            'temperature': 0.0
        }
        
        # eval_embedding_config: for embedding in custom metrics, uses evaluation/.env
        self.eval_embedding_config = {
            'model': eval_env.embedding_model,
            'api_key': eval_env.embedding_api_key,
            'api_base': eval_env.embedding_base_url
        }
        
        # Retry settings for NaN handling
        self.max_ragas_retries = 3
        self.retry_delay = 2  # seconds
        
        # Set the dataset with optional results_subdir
        self.eval_config.set_dataset(dataset, results_subdir=results_subdir)
        
        # Validate configuration
        self.eval_config.validate()
        
        # Initialize OpenAI client for direct LLM mode using answer generation config
        self.llm_client = OpenAI(
            api_key=self.answer_model_config['api_key'],
            base_url=self.answer_model_config['api_base']
        )
        
        # Initialize RAGAS LLM (reuse for incremental evaluation) using eval model config
        if RAGAS_AVAILABLE:
            self.ragas_llm = ChatOpenAI(
                model=self.eval_model_config['model'],
                temperature=self.eval_model_config['temperature'],
                openai_api_key=self.eval_model_config['api_key'],
                openai_api_base=self.eval_model_config['api_base']
            )
            # RAGAS metrics suite (3 metrics)
            # context_precision -> RESTORED to RAGAS native
            # faithfulness -> CustomFaithfulness
            # answer_relevancy -> CustomAnswerRelevancy
            # answer_correctness -> CustomAnswerCorrectness
            self.ragas_metrics_list = [
                context_precision,  # RAGAS native
                context_recall,
                answer_similarity
            ]
        
        if CUSTOM_METRICS_AVAILABLE:
            self.custom_answer_relevancy = CustomAnswerRelevancy(
                model_config=self.eval_model_config,
                embedding_config=self.eval_embedding_config,  # Uses evaluation/.env
                verbose=False
            )
            self.custom_faithfulness = CustomFaithfulness(
                model_config=self.eval_model_config,
                verbose=False
            )
            self.custom_answer_correctness = CustomAnswerCorrectness(
                model_config=self.eval_model_config,
                verbose=False
            )
            # Note: context_precision is now using RAGAS native implementation
        else:
            self.custom_answer_relevancy = None
            self.custom_faithfulness = None
            self.custom_answer_correctness = None

        # Simplified metric system:
        # - RAGAS (3): context_precision, context_recall, answer_similarity
        # - Custom (3): answer_relevancy, faithfulness, answer_correctness

        if self.verbose:
            print(f"\n{'='*80}")
            print("RAG System Evaluator Initialized")
            print(f"{'='*80}")
            print(f"Dataset: {self.dataset}")
            print(f"Test Data: {self.eval_config.test_data_path.name}")
            print(f"Answer Generation Model: {self.answer_model_config['model']}")
            print(f"Evaluation Model: {self.eval_model_config['model']}")
            print(f"Results Directory: {self.eval_config.result_files['agentic_with_kg'].parent.parent}")
            print(f"Fact Checker: Enabled (Factual Correctness Priority)")
            print(f"Incremental Save: {'Enabled' if self.incremental_save else 'Disabled'}")
            print(f"Resume Mode: {'Enabled' if self.resume else 'Disabled'}")
            if self.generate_only:
                print(f"🚀 Mode: GENERATE ONLY (Phase 1 - answers with placeholder metrics)")
            elif self.eval_only:
                print(f"📊 Mode: EVAL ONLY (Phase 2 - compute metrics on existing answers)")
            print(f"{'='*80}\n")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test questions from JSON file
        
        Returns:
            List of QA pairs
        """
        if self.verbose:
            print(f"📚 Loading test data from {self.eval_config.test_data_path}...")
        
        with open(self.eval_config.test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = data.get('qa_pairs', [])
        
        # Filter based on test mode
        if self.eval_config.test_mode == "first_n":
            qa_pairs = qa_pairs[:self.eval_config.num_test_questions]
        elif self.eval_config.test_mode == "random":
            import random
            random.shuffle(qa_pairs)
            qa_pairs = qa_pairs[:self.eval_config.num_test_questions]
        # else: use all questions
        
        if self.verbose:
            print(f"✅ Loaded {len(qa_pairs)} questions for evaluation\n")
        
        return qa_pairs
    
    def load_existing_progress(self, mode: str) -> tuple[List[Dict], Dict]:
        """
        Load existing evaluation progress from results file
        
        Args:
            mode: Evaluation mode
            
        Returns:
            Tuple of (qa_results list, full results dict)
        """
        result_file = self.eval_config.result_files[mode]
        
        if not result_file.exists():
            if self.verbose:
                print(f"📂 No existing progress found, starting fresh...\n")
            return [], {}
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            qa_results = existing_results.get('qa_results', [])
            
            if self.verbose:
                print(f"📂 Found existing progress: {len(qa_results)} questions already completed")
                if qa_results:
                    print(f"   Resuming from question {len(qa_results) + 1}...\n")
            
            return qa_results, existing_results
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error loading existing progress: {e}")
                print(f"   Starting fresh...\n")
            return [], {}
    
    def _has_generation_error(self, qa_result: dict) -> bool:
        """
        Check if a QA result has an error that requires re-generation.
        
        Detects:
        - HTML error pages (API gateway errors)
        - Rate limit errors (429)
        - Empty answers with error messages
        - Network/connection errors
        
        Args:
            qa_result: The QA result dictionary
            
        Returns:
            True if the result should be re-generated
        """
        error = qa_result.get('error')
        answer = qa_result.get('generated_answer', '')
        
        # Check if generated_answer itself contains an error message
        # (This can happen when error field is null but answer generation failed)
        if answer:
            answer_lower = answer.lower()
            error_prefixes = [
                'error generating answer',
                'error:',
                'failed to generate',
                'exception:',
                "'nonetype' object has no attribute"
            ]
            for prefix in error_prefixes:
                if prefix in answer_lower:
                    return True
        
        # No error field and no error in answer means successful
        if not error:
            return False
        
        error_str = str(error).lower()
        
        # Check for HTML error responses (API gateway returned error page)
        if '<!doctype' in error_str or '<html' in error_str:
            return True
        
        # Check for rate limit errors
        if '429' in error_str or 'rate' in error_str or 'limit' in error_str:
            return True
        
        # Check for network/connection errors
        if 'connection' in error_str or 'timeout' in error_str or 'network' in error_str:
            return True
        
        # Check for empty answer with any error
        if not answer or not answer.strip():
            return True
        
        return False
    
    def _filter_for_resume(self, qa_results: List[Dict], qa_pairs: List[Dict]) -> tuple[List[Dict], set, List[Dict]]:
        """
        Filter QA results for resume mode, separating successful and errored results.
        
        Args:
            qa_results: Existing QA results from previous run
            qa_pairs: All QA pairs to process
            
        Returns:
            Tuple of:
            - successful_results: List of QA results without errors (to keep)
            - completed_ids: Set of question IDs that are successfully completed
            - errored_results: List of QA results that have errors (to re-generate)
        """
        successful_results = []
        errored_results = []
        completed_ids = set()
        
        for r in qa_results:
            qid = r.get('question_id')
            if self._has_generation_error(r):
                errored_results.append(r)
            else:
                successful_results.append(r)
                completed_ids.add(qid)
        
        if self.verbose and errored_results:
            print(f"🔄 Found {len(errored_results)} questions with errors that will be re-generated:")
            for r in errored_results[:5]:  # Show first 5
                error = r.get('error') or r.get('generated_answer', '')[:60]
                error_preview = str(error)[:60]
                print(f"   - {r.get('question_id')}: {error_preview}...")
            if len(errored_results) > 5:
                print(f"   ... and {len(errored_results) - 5} more")
            print()
        
        return successful_results, completed_ids, errored_results
    
    def _clean_answer_for_evaluation(self, answer: str) -> str:
        """
        Clean answer text for RAGAS evaluation
        
        Removes formatting markers like [Sources: ...] that may interfere
        with RAGAS metrics calculation. The original answer is preserved
        for saving to results.
        
        Args:
            answer: Original generated answer
            
        Returns:
            Cleaned answer text for evaluation only
        """
        import re
        # Remove [Sources: ...] markers
        cleaned = re.sub(r'\n*\[Sources:.*?\]', '', answer, flags=re.IGNORECASE)
        # Remove [No sources] markers
        cleaned = re.sub(r'\n*\[No sources\]', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()
    
    def _compute_single_ragas_metric(self, dataset, metric, metric_name: str, timeout_seconds: int = 60) -> tuple:
        """
        Compute a single RAGAS metric with proper asyncio handling
        
        RAGAS uses asyncio internally. To avoid event loop conflicts:
        1. Run evaluate() directly (not in ThreadPoolExecutor)
        2. Use RAGAS RunConfig for internal timeout control
        3. Handle asyncio exceptions gracefully
        
        Args:
            dataset: RAGAS Dataset object
            metric: RAGAS metric object to compute  
            metric_name: Name of the metric for logging
            timeout_seconds: Timeout in seconds (used as reference, actual timeout via RunConfig)
            
        Returns:
            Tuple of (metric_name, score) or (metric_name, None) on failure
        """
        import math
        import asyncio
        
        try:
            # Create a custom RunConfig with the specified timeout
            metric_run_config = RunConfig(
                timeout=timeout_seconds,  # Per-metric timeout
                max_retries=2,            # Quick retries
                max_wait=20,              # 20s between retries
                max_workers=2             # Reduce concurrent calls
            )
            
            # Run RAGAS evaluate directly
            eval_result = evaluate(
                dataset,
                metrics=[metric],
                llm=self.ragas_llm,
                run_config=metric_run_config
            )
            
            if eval_result.scores and len(eval_result.scores) > 0:
                scores = eval_result.scores[0]
                value = scores.get(metric_name)
                
                # Check for NaN
                if value is not None and isinstance(value, float) and math.isnan(value):
                    if self.verbose:
                        print(f"        ⚠️  {metric_name}: NaN")
                    return (metric_name, None)
                
                if value is not None:
                    if self.verbose:
                        print(f"        ✓ {metric_name}: {value:.4f}")
                    return (metric_name, value)
                    
            return (metric_name, None)
        
        except asyncio.TimeoutError:
            if self.verbose:
                print(f"        ⏱️  {metric_name}: Timeout ({timeout_seconds}s)")
            return (metric_name, None)
        except TimeoutError:
            if self.verbose:
                print(f"        ⏱️  {metric_name}: Timeout ({timeout_seconds}s)")
            return (metric_name, None)
        except IndexError as ie:
            # Known RAGAS bug with answer_relevancy
            if self.verbose:
                print(f"        ⚠️  {metric_name}: IndexError (RAGAS bug)")
            return (metric_name, None)
        except RuntimeError as re:
            # Handle "Event loop is closed" errors
            if "Event loop" in str(re):
                if self.verbose:
                    print(f"        ⚠️  {metric_name}: Event loop error (retrying...)")
                # Try to create a new event loop
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return self._compute_single_ragas_metric(dataset, metric, metric_name, timeout_seconds)
                except:
                    return (metric_name, None)
            if self.verbose:
                print(f"        ⚠️  {metric_name}: RuntimeError ({str(re)[:30]})")
            return (metric_name, None)
        except Exception as e:
            if self.verbose:
                print(f"        ⚠️  {metric_name}: Error ({type(e).__name__})")
            return (metric_name, None)
    
    def _compute_ragas_single(self, question: str, answer: str, 
                              ground_truth: str, contexts: List[str]) -> Dict:
        """
        Compute RAGAS metrics for a single Q&A pair - FULLY ISOLATED approach
        
        IMPORTANT: This method ensures complete isolation:
        1. Creates a fresh Dataset for this question only
        2. Each metric is computed in a separate thread
        3. No shared state between questions or metrics
        
        Args:
            question: Question text
            answer: Generated answer (final answer from RAG system)
            ground_truth: Ground truth answer
            contexts: List of context strings
            
        Returns:
            Dictionary of RAGAS scores for this single question
        """
        if not RAGAS_AVAILABLE:
            return {}
        
        try:
            import math
            import gc  # Garbage collection for memory cleanup
            
            # Clean answer for evaluation (remove [Sources: ...] markers)
            clean_answer = self._clean_answer_for_evaluation(answer)
            
            if self.verbose:
                if clean_answer != answer:
                    print(f"      📝 Answer cleaned for RAGAS (removed source markers)")
            
            # Prepare contexts for this single question
            valid_contexts = []  # Initialize before the if block
            if contexts and len(contexts) > 0:
                valid_contexts = [c for c in contexts if c and c.strip()]
                contexts_for_ragas = [valid_contexts] if valid_contexts else [['No context available']]
            else:
                contexts_for_ragas = [['No context available']]
            
            # Create a FRESH Dataset object for this single question
            # This ensures no state is shared between questions
            data = {
                'question': [question],
                'answer': [clean_answer],
                'contexts': contexts_for_ragas,
                'ground_truth': [ground_truth]
            }
            
            # RAGAS metrics config (3 metrics)
            metrics_config = [
                (context_precision, 'context_precision', 90),  # RAGAS native
                (context_recall, 'context_recall', 90),
                (answer_similarity, 'answer_similarity', 60),
            ]
            
            if self.verbose:
                print(f"      🔄 Computing RAGAS metrics ({len(metrics_config)}) + Custom metrics...")
            
            # Initialize scores dict
            clean_scores = {}
            successful_count = 0
            
            # Note: context_precision is now computed via RAGAS in Phase 1
            # Phase 1: Compute RAGAS metrics
            failed_metrics = []  # Store failed metrics for retry
            
            for metric, metric_name, timeout in metrics_config:
                # Create a fresh Dataset for EACH metric to ensure complete isolation
                dataset = Dataset.from_dict(data)
                
                metric_result = self._compute_single_ragas_metric(
                    dataset, metric, metric_name, timeout
                )
                
                name, value = metric_result
                clean_scores[name] = value
                
                if value is not None:
                    successful_count += 1
                else:
                    # Store failed metric for retry
                    failed_metrics.append((metric, metric_name, timeout))
                
                # Clean up dataset to free memory
                del dataset
            
            # Phase 2: Retry failed metrics separately (if any)
            if failed_metrics and self.verbose:
                print(f"      🔄 Retrying {len(failed_metrics)} failed metrics...")
            
            for metric, metric_name, timeout in failed_metrics:
                # Create fresh dataset for retry
                dataset = Dataset.from_dict(data)
                
                # Retry with slightly longer timeout
                retry_timeout = min(timeout + 60, 240)  # Max 4 minutes
                
                if self.verbose:
                    print(f"        🔁 Retrying {metric_name} (timeout: {retry_timeout}s)...")
                
                metric_result = self._compute_single_ragas_metric(
                    dataset, metric, metric_name, retry_timeout
                )
                
                name, value = metric_result
                
                if value is not None:
                    clean_scores[name] = value
                    successful_count += 1
                    # Remove from failed list (update nan_metrics later)
                
                del dataset
            
            # Force garbage collection after processing this question
            gc.collect()
            
            # Phase 3: Compute Custom Answer Relevancy (replaces buggy RAGAS version)
            if self.custom_answer_relevancy:
                if self.verbose:
                    print(f"      🔄 Computing Custom Answer Relevancy...")
                
                try:
                    relevancy_result = self.custom_answer_relevancy.compute(question, clean_answer)
                    relevancy_score = relevancy_result.get('custom_answer_relevancy')
                    
                    if relevancy_score is not None:
                        clean_scores['answer_relevancy'] = relevancy_score
                        successful_count += 1
                        if self.verbose:
                            print(f"        ✓ answer_relevancy (Custom): {relevancy_score:.4f}")
                    else:
                        clean_scores['answer_relevancy'] = None
                        if self.verbose:
                            print(f"        ⚠️  answer_relevancy (Custom): Failed")
                            
                except Exception as e:
                    if self.verbose:
                        print(f"        ⚠️  Custom Answer Relevancy error: {str(e)}")
                    clean_scores['answer_relevancy'] = None

            # Phase 4: Compute Custom Faithfulness (replaces slow/timed-out RAGAS version)
            if self.custom_faithfulness:
                if self.verbose:
                    print(f"      🔄 Computing Custom Faithfulness...")
                
                try:
                    # Extracts statements and verifies against context
                    faith_result = self.custom_faithfulness.compute(
                        answer=clean_answer,
                        contexts=valid_contexts if valid_contexts else []
                    )
                    
                    faith_score = faith_result.get('custom_faithfulness')
                    
                    if faith_score is not None:
                        clean_scores['faithfulness'] = faith_score
                        successful_count += 1
                        if self.verbose:
                            print(f"        ✓ faithfulness (Custom): {faith_score:.4f}")
                    else:
                        clean_scores['faithfulness'] = None
                        if self.verbose:
                            error_msg = faith_result.get('_error', 'unknown error')
                            print(f"        ⚠️  faithfulness (Custom): Failed ({error_msg})")
                            
                except Exception as e:
                    if self.verbose:
                        print(f"        ⚠️  Custom Faithfulness error: {str(e)}")
                    clean_scores['faithfulness'] = None
            
            # Phase 5: Compute Custom Answer Correctness (replaces FactChecker)
            if self.custom_answer_correctness:
                if self.verbose:
                    print(f"      🔄 Computing Custom Answer Correctness...")
                
                try:
                    correctness_result = self.custom_answer_correctness.compute(
                        question=question,
                        answer=clean_answer,
                        ground_truth=ground_truth
                    )
                    
                    correctness_score = correctness_result.get('custom_answer_correctness')
                    
                    if correctness_score is not None:
                        clean_scores['answer_correctness'] = correctness_score
                        successful_count += 1
                        if self.verbose:
                            print(f"        ✓ answer_correctness (Custom): {correctness_score:.4f}")
                    else:
                        clean_scores['answer_correctness'] = None
                        if self.verbose:
                            error_msg = correctness_result.get('_error', 'unknown error')
                            print(f"        ⚠️  answer_correctness (Custom): Failed ({error_msg})")
                            
                except Exception as e:
                    if self.verbose:
                        print(f"        ⚠️  Custom Answer Correctness error: {str(e)}")
                    clean_scores['answer_correctness'] = None
            
            # Calculate final nan_metrics (those that still failed after retry)
            nan_metrics = [name for name, value in clean_scores.items() 
                          if value is None and not name.startswith('_')]
            
            if self.verbose:
                # Total: 3 RAGAS metrics + 3 custom metrics (relevancy, faithfulness, correctness)
                custom_metrics_count = sum([
                    1 if self.custom_answer_relevancy else 0,
                    1 if self.custom_faithfulness else 0,
                    1 if self.custom_answer_correctness else 0
                ])
                total_metrics_count = len(metrics_config) + custom_metrics_count
                print(f"      📊 Metrics completed: {successful_count}/{total_metrics_count} metrics successful")
                if nan_metrics:
                    print(f"      ⚠️  Failed metrics: {', '.join(nan_metrics)}")
            
            # Add metadata
            if nan_metrics:
                clean_scores['_nan_metrics'] = nan_metrics
            clean_scores['_successful_count'] = successful_count
            
            return clean_scores
        
        except Exception as e:
            if self.verbose:
                print(f"      ⚠️  RAGAS error: {str(e)}")
                print(f"      Error type: {type(e).__name__}")
            return {'error': str(e)}
    
    def _compute_ragas_single_with_retry(self, question: str, answer: str,
                                         ground_truth: str, contexts: List[str],
                                         retry_count: int = 0) -> Dict:
        """
        Compute RAGAS metrics with smart retry mechanism
        
        Since metrics are now computed sequentially with individual timeouts,
        we only retry if a majority of metrics failed.
        
        Args:
            question: Question text
            answer: Generated answer
            ground_truth: Ground truth answer
            contexts: List of context strings
            retry_count: Current retry attempt (0-indexed)
            
        Returns:
            Dictionary of RAGAS scores
        """
        if retry_count >= self.max_ragas_retries:
            if self.verbose:
                print(f"      ❌ Max retries ({self.max_ragas_retries}) reached, giving up")
            return {'error': 'max_retries_exceeded', 'retry_count': retry_count}
        
        # Attempt to compute scores
        scores = self._compute_ragas_single(question, answer, ground_truth, contexts)
        
        # Check for complete error
        if 'error' in scores and len(scores) == 1:
            if self.verbose:
                print(f"      ❌ Complete failure, skipping retries")
            return scores
        
        # Count successful metrics (those with numeric values)
        import math
        total_metrics = 6  # faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness, answer_similarity
        
        successful_count = scores.get('_successful_count', 0)
        if successful_count == 0:
            # Count manually if not provided
            for key, value in scores.items():
                if not key.startswith('_') and key != 'error':
                    if value is not None and isinstance(value, (int, float)) and not math.isnan(value):
                        successful_count += 1
        
        # Only retry if less than half the metrics succeeded
        failure_threshold = total_metrics // 2  # 3 or more failures trigger retry
        failed_count = total_metrics - successful_count
        
        if failed_count >= failure_threshold and retry_count < self.max_ragas_retries - 1:
            if self.verbose:
                print(f"      ⚠️  {failed_count}/{total_metrics} metrics failed")
                print(f"      🔄 Retrying in {self.retry_delay}s (attempt {retry_count + 2}/{self.max_ragas_retries})...")
            
            time.sleep(self.retry_delay)
            
            # Retry
            retry_scores = self._compute_ragas_single_with_retry(
                question, answer, ground_truth, contexts, retry_count + 1
            )
            
            # Merge: keep best results from both attempts
            merged = {}
            for key in set(list(scores.keys()) + list(retry_scores.keys())):
                if key.startswith('_'):
                    continue
                old_val = scores.get(key)
                new_val = retry_scores.get(key)
                # Prefer non-None values
                if new_val is not None:
                    merged[key] = new_val
                elif old_val is not None:
                    merged[key] = old_val
                else:
                    merged[key] = None
            
            return merged
        
        # Success or acceptable failure rate
        if self.verbose and retry_count > 0:
            print(f"      ✅ Retry improved results (attempt {retry_count + 1})")
        
        return scores
    
    def _compute_fact_single(self, question: str, answer: str,
                            ground_truth: str, contexts: List[str]) -> Dict:
        """
        Compute fact-checking metrics for a single Q&A pair
        
        Args:
            question: Question text
            answer: Generated answer
            ground_truth: Ground truth answer
            contexts: List of context strings
            
        Returns:
            Dictionary of fact-checking scores
        """
        try:
            fact_metrics = self.fact_checker.compute_factual_correctness(
                question=question,
                generated_answer=answer,
                ground_truth=ground_truth,
                contexts=contexts
            )
            
            # Simplify for storage with safe .get() to avoid KeyError
            return {
                'factual_correctness': round(fact_metrics.get('factual_correctness', 0.0), 4),
                'key_facts_coverage': round(fact_metrics.get('key_facts_coverage', 0.0), 4),
                'fact_accuracy': round(fact_metrics.get('fact_accuracy', 0.0), 4),
                'fact_precision': round(fact_metrics.get('fact_precision', 0.0), 4),
                'fact_recall': round(fact_metrics.get('fact_recall', 0.0), 4),
                'hallucination_rate': round(fact_metrics.get('hallucination_rate', 0.0), 4),
                'context_supported_rate': round(fact_metrics.get('context_supported_rate', 0.0), 4),
                'critical_errors_count': len(fact_metrics.get('critical_errors', [])),
                'moderate_errors': fact_metrics.get('moderate_errors', 0),
                'minor_errors': fact_metrics.get('minor_errors', 0)
            }
        
        except Exception as e:
            if self.verbose:
                print(f"      ⚠️  Fact-check error: {str(e)}")
            return {'error': str(e)}
    

    def save_incremental_results(self, qa_results: List[Dict], mode: str, 
                                 status: str = 'in_progress', total_questions: int = 0):
        """
        Save incremental evaluation results
        
        Args:
            qa_results: List of completed QA results
            mode: Evaluation mode
            status: Status ('in_progress' or 'completed')
            total_questions: Total number of questions to process
        """
        # Compute aggregated metrics (streamlined: RAGAS + Fact Checker only)
        ragas_metrics = self._compute_aggregate_ragas(qa_results)
        fact_metrics = self._compute_aggregate_fact(qa_results)
        
        # Compute weighted score (simplified: just use RAGAS average which now includes all 6 metrics)
        # Note: answer_correctness (from FactChecker) is already included in ragas_metrics
        weighted_score = ragas_metrics.get('average_score')
        
        # Build results
        results = {
            'metadata': {
                'answer_generation': {
                    'model': self.answer_model_config['model'],
                    'api_base': self.answer_model_config['api_base'],
                    'temperature': self.answer_model_config['temperature']
                },
                'evaluation': {
                    'ragas_model': self.eval_model_config['model'],
                    'fact_check_model': self.eval_model_config['model'],
                    'api_base': self.eval_model_config['api_base'],
                    'temperature': self.eval_model_config['temperature'],
                    'metrics_version': 'full_v2_integrated'
                },
                'reranker': {
                    'enabled': self.eval_config.enable_reranking,
                    'model': self.eval_config.reranker_model if self.eval_config.enable_reranking else None
                },
                'mode': mode,
                'dataset': self.dataset,
                'test_set': self.eval_config.test_data_path.stem,
                'num_questions_completed': len(qa_results),
                'num_questions_total': total_questions or len(qa_results),
                'timestamp': datetime.now().isoformat(),
                'status': status
            },
            'qa_results': qa_results,
            'ragas_metrics': ragas_metrics,
            # 'fact_metrics': fact_metrics, # Hidden as requested (included in RAGAS)
            'weighted_score': round(weighted_score, 4) if weighted_score is not None else None,
            'performance': self._compute_performance_metrics(qa_results, mode)
        }
        
        # Save using existing save_results method
        self.save_results(results, mode)
    
    def _compute_aggregate_ragas(self, qa_results: List[Dict]) -> Dict:
        """Compute aggregated RAGAS metrics from individual results"""
        if not RAGAS_AVAILABLE:
            return {}
        
        # Extract individual RAGAS scores
        ragas_scores = [r.get('ragas_scores', {}) for r in qa_results 
                       if not r.get('error') and r.get('ragas_scores')]
        
        if not ragas_scores:
            return {}
        
        # Get all metric names
        metric_names = set()
        for scores in ragas_scores:
            metric_names.update(scores.keys())
        metric_names.discard('error')
        
        # Compute averages
        import math
        aggregated = {}
        for metric in metric_names:
            # Skip private/internal metrics that start with '_'
            if metric.startswith('_'):
                continue
            
            values = []
            for scores in ragas_scores:
                value = scores.get(metric)
                # Only include numeric values (skip lists, dicts, None, NaN)
                if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                    values.append(value)
            
            if values:
                aggregated[metric] = sum(values) / len(values)
            else:
                aggregated[metric] = None
        
        # Compute average score from all metrics (context_precision is now 0-1 from RAGAS)
        metrics_to_average = ['context_precision', 'context_recall', 'answer_similarity', 
                              'answer_relevancy', 'faithfulness', 'answer_correctness']
        valid_values = []
        for metric in metrics_to_average:
            v = aggregated.get(metric)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                valid_values.append(v)
        
        if valid_values:
            aggregated['average_score'] = sum(valid_values) / len(valid_values)
        else:
            aggregated['average_score'] = None
        
        return aggregated
    
    def _compute_aggregate_fact(self, qa_results: List[Dict]) -> Dict:
        """Compute aggregated fact-checking metrics from individual results"""
        # Extract individual fact scores
        fact_scores = [r.get('fact_scores', {}) for r in qa_results 
                      if not r.get('error') and r.get('fact_scores')]
        
        if not fact_scores:
            return {}
        
        # Compute averages
        aggregated = {
            'factual_correctness': sum(s.get('factual_correctness', 0) for s in fact_scores) / len(fact_scores),
            'key_facts_coverage': sum(s.get('key_facts_coverage', 0) for s in fact_scores) / len(fact_scores),
            'fact_accuracy': sum(s.get('fact_accuracy', 0) for s in fact_scores) / len(fact_scores),
            'fact_precision': sum(s.get('fact_precision', 0) for s in fact_scores) / len(fact_scores),
            'fact_recall': sum(s.get('fact_recall', 0) for s in fact_scores) / len(fact_scores),
            'hallucination_rate': sum(s.get('hallucination_rate', 0) for s in fact_scores) / len(fact_scores),
            'context_supported_rate': sum(s.get('context_supported_rate', 0) for s in fact_scores) / len(fact_scores),
            'total_critical_errors': sum(s.get('critical_errors_count', 0) for s in fact_scores),
            'total_moderate_errors': sum(s.get('moderate_errors', 0) for s in fact_scores),
            'total_minor_errors': sum(s.get('minor_errors', 0) for s in fact_scores)
        }
        
        # Round values
        for key in aggregated:
            if isinstance(aggregated[key], float):
                aggregated[key] = round(aggregated[key], 4)
        
        return aggregated
    

    def _compute_performance_metrics(self, qa_results: List[Dict], mode: str) -> Dict:
        """Compute performance metrics from QA results"""
        if not qa_results:
            return {}
        
        perf = {
            'avg_response_time': sum(r.get('response_time', 0) for r in qa_results) / len(qa_results),
            'total_time': sum(r.get('response_time', 0) for r in qa_results),
            'error_count': sum(1 for r in qa_results if r.get('error'))
        }
        
        # Add agentic-specific metrics
        if 'agentic' in mode:
            perf['avg_iterations'] = sum(r.get('iterations', 0) for r in qa_results) / len(qa_results)
            perf['avg_quality_score'] = sum(r.get('quality_score', 0) for r in qa_results) / len(qa_results)
        
        return perf
    
    def evaluate_agentic_with_kg(self, qa_pairs: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate Agentic RAG with Knowledge Graph
        
        Args:
            qa_pairs: List of question-answer pairs
            
        Returns:
            Evaluation results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("Mode 1: Agentic RAG + Knowledge Graph")
            print(f"{'='*80}\n")
        
        # Temporarily set answer generation model configuration
        original_model = agentic_config.llm_model
        original_api_key = agentic_config.openai_api_key
        original_api_base = agentic_config.openai_base_url
        original_temperature = agentic_config.llm_temperature
        
        agentic_config.llm_model = self.answer_model_config['model']
        agentic_config.openai_api_key = self.answer_model_config['api_key']
        agentic_config.openai_base_url = self.answer_model_config['api_base']
        agentic_config.llm_temperature = self.answer_model_config['temperature']
        
        try:
            # Initialize agent with KG enabled
            agent = AgentOrchestrator(
                enable_kg=True,
                max_iterations=self.eval_config.max_iterations,
                quality_threshold=self.eval_config.quality_threshold,
                verbose=False  # Suppress agent's verbose output during batch
            )
            
            results = self._run_evaluation(agent, qa_pairs, mode="agentic_with_kg")
            return results
        finally:
            # Restore original configuration
            agentic_config.llm_model = original_model
            agentic_config.openai_api_key = original_api_key
            agentic_config.openai_base_url = original_api_base
            agentic_config.llm_temperature = original_temperature
    
    def evaluate_agentic_no_kg(self, qa_pairs: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate Agentic RAG without Knowledge Graph
        
        Args:
            qa_pairs: List of question-answer pairs
            
        Returns:
            Evaluation results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("Mode 2: Agentic RAG (Vector Only)")
            print(f"{'='*80}\n")
        
        # Temporarily set answer generation model configuration
        original_model = agentic_config.llm_model
        original_api_key = agentic_config.openai_api_key
        original_api_base = agentic_config.openai_base_url
        original_temperature = agentic_config.llm_temperature
        
        agentic_config.llm_model = self.answer_model_config['model']
        agentic_config.openai_api_key = self.answer_model_config['api_key']
        agentic_config.openai_base_url = self.answer_model_config['api_base']
        agentic_config.llm_temperature = self.answer_model_config['temperature']
        
        try:
            # Initialize agent with KG disabled
            agent = AgentOrchestrator(
                enable_kg=False,
                max_iterations=self.eval_config.max_iterations,
                quality_threshold=self.eval_config.quality_threshold,
                verbose=False
            )
            
            results = self._run_evaluation(agent, qa_pairs, mode="agentic_no_kg")
            return results
        finally:
            # Restore original configuration
            agentic_config.llm_model = original_model
            agentic_config.openai_api_key = original_api_key
            agentic_config.openai_base_url = original_api_base
            agentic_config.llm_temperature = original_temperature
    
    def evaluate_direct_llm(self, qa_pairs: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate Direct LLM (no retrieval) with incremental save support
        
        Args:
            qa_pairs: List of question-answer pairs
            
        Returns:
            Evaluation results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("Mode 3: Direct LLM (No Retrieval)")
            print(f"{'='*80}\n")
        
        mode = 'direct_llm'
        
        # Load existing progress if resume is enabled
        if self.resume:
            all_qa_results, _ = self.load_existing_progress(mode)
            
            # Dry-run mode: analyze existing results only, don't create anything
            if self.dry_run:
                # If no existing results, just report and return
                if not all_qa_results:
                    result_file = self.eval_config.result_files[mode]
                    print(f"\n{'='*60}")
                    print(f"DRY RUN: No existing results found")
                    print(f"{'='*60}")
                    print(f"  📂 Expected path: {result_file}")
                    print(f"  ℹ️  Nothing to analyze - run without --dry-run to start fresh")
                    print(f"{'='*60}\n")
                    return {
                        'metadata': {'mode': mode, 'dry_run': True, 'no_existing_results': True},
                        'qa_results': []
                    }
                
                successful_results, completed_ids, errored_results = self._filter_for_resume(all_qa_results, qa_pairs)
                
                # Calculate pending questions (not yet completed successfully)
                pending_questions = []
                for idx, qa in enumerate(qa_pairs, 1):
                    qid = qa.get('id', f'qa_{idx}')
                    if qid not in completed_ids:
                        pending_questions.append(qa)
                
                print(f"\n{'='*60}")
                print(f"DRY RUN: Would process {len(pending_questions)} questions")
                print(f"{'='*60}")
                print(f"  ✅ Successfully completed: {len(successful_results)}")
                print(f"  🔄 With errors (to re-generate): {len(errored_results)}")
                print(f"  📋 Not yet processed: {len(pending_questions) - len(errored_results)}")
                
                if errored_results:
                    print(f"\nQuestions to re-generate:")
                    for r in errored_results:
                        error_preview = str(r.get('error', ''))[:80]
                        print(f"  - {r.get('question_id')}: {error_preview}...")
                
                print(f"\n{'='*60}")
                print(f"To actually run, remove --dry-run flag")
                print(f"{'='*60}\n")
                
                # Return existing results without changes
                return {
                    'metadata': {'mode': mode, 'dry_run': True, 'would_process': len(pending_questions)},
                    'qa_results': all_qa_results
                }
            
            # Normal resume mode (not dry-run)
            successful_results, completed_ids, errored_results = self._filter_for_resume(all_qa_results, qa_pairs)
            qa_results = list(successful_results)
        else:
            qa_results = []
            completed_ids = set()
        
        total_questions = len(qa_pairs)
        
        # Use tqdm progress bar for QA processing
        pbar = tqdm(enumerate(qa_pairs, 1), total=total_questions, 
                    desc="Direct LLM", unit="Q", 
                    disable=not TQDM_AVAILABLE,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i, qa in pbar:
            question_id = qa.get('id', f'qa_{i}')
            
            # Skip if already completed successfully (resume mode)
            if question_id in completed_ids:
                if self.verbose:
                    print(f"[{i}/{total_questions}] ⏭️  Skipping (already completed): {question_id}")
                continue
            
            question = qa['question']
            ground_truth = qa['answer']
            
            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"[{len(qa_results)+1}/{total_questions}] Processing Question {i}")
                print(f"{'─'*80}")
                print(f"📝 Question: {question[:100]}{'...' if len(question) > 100 else ''}")
            
            # Step 1: Generate answer
            start_time = time.time()
            
            try:
                if self.verbose:
                    print(f"🤖 Calling LLM...")
                
                # Direct LLM call without any context (using answer generation model config)
                response = self.llm_client.chat.completions.create(
                    model=self.answer_model_config['model'],
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Answer the question based on your knowledge."
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    temperature=self.answer_model_config['temperature'],
                    max_tokens=agentic_config.llm_max_tokens
                )
                
                answer = response.choices[0].message.content
                response_time = time.time() - start_time
                
                qa_result = {
                    'question_id': question_id,
                    'question': question,
                    'ground_truth': ground_truth,
                    'generated_answer': answer,
                    'contexts': [],  # No contexts for direct LLM
                    'sources': [],
                    'response_time': round(response_time, 2),
                    'iterations': 0,
                    'error': None,
                    'category': qa.get('category', ''),
                    'source_title': qa.get('source_title', ''),
                    'difficulty': qa.get('difficulty', ''),
                    'reasoning_type': qa.get('reasoning_type', '')
                }
                
                if self.verbose:
                    print(f"   ✅ Answer generated in {response_time:.2f}s")
                
                # Step 2: Compute metrics for Direct LLM (or use placeholder in generate_only mode)
                # Note: context_precision, context_recall, faithfulness are N/A for Direct LLM
                import math
                
                # Generate-only mode: skip all metrics computation
                if self.generate_only:
                    if self.verbose:
                        print(f"⏭️  Skipping metrics (generate-only mode)")
                    ragas_scores = {
                        'context_precision': None,
                        'context_recall': None,
                        'faithfulness': None,
                        'answer_similarity': None,
                        'answer_relevancy': None,
                        'answer_correctness': None,
                        '_placeholder': True,
                        '_note': 'Pending evaluation (generate-only mode)'
                    }
                else:
                    if self.verbose:
                        print(f"📊 Computing metrics for Direct LLM (no context)...")
                    
                    ragas_start = time.time()
                    
                    # Initialize with NaN for context-dependent metrics
                    ragas_scores = {
                        'context_precision': float('nan'),   # N/A - no contexts
                        'context_recall': float('nan'),      # N/A - no contexts
                        'faithfulness': float('nan'),        # N/A - no contexts
                    }
                    successful_count = 0
                
                    # Compute answer_similarity (RAGAS)
                    if RAGAS_AVAILABLE and not self.skip_ragas:
                        try:
                            data = {
                                'question': [question],
                                'answer': [answer],
                                'contexts': [['No context available']],
                                'ground_truth': [ground_truth]
                            }
                            dataset = Dataset.from_dict(data)
                            
                            metric_result = self._compute_single_ragas_metric(
                                dataset, answer_similarity, 'answer_similarity', 60
                            )
                            name, value = metric_result
                            if value is not None:
                                ragas_scores['answer_similarity'] = value
                                successful_count += 1
                                if self.verbose:
                                    print(f"        ✓ answer_similarity: {value:.4f}")
                        except Exception as e:
                            if self.verbose:
                                print(f"        ⚠️  answer_similarity error: {str(e)}")
                            ragas_scores['answer_similarity'] = None
                    
                    # Compute answer_relevancy (Custom)
                    if self.custom_answer_relevancy:
                        try:
                            relevancy_result = self.custom_answer_relevancy.compute(question, answer)
                            relevancy_score = relevancy_result.get('custom_answer_relevancy')
                            if relevancy_score is not None:
                                ragas_scores['answer_relevancy'] = relevancy_score
                                successful_count += 1
                                if self.verbose:
                                    print(f"        ✓ answer_relevancy (Custom): {relevancy_score:.4f}")
                        except Exception as e:
                            if self.verbose:
                                print(f"        ⚠️  answer_relevancy error: {str(e)}")
                            ragas_scores['answer_relevancy'] = None
                    
                    # Compute answer_correctness (Custom)
                    if self.custom_answer_correctness:
                        try:
                            correctness_result = self.custom_answer_correctness.compute(
                                question=question,
                                answer=answer,
                                ground_truth=ground_truth
                            )
                            correctness_score = correctness_result.get('custom_answer_correctness')
                            if correctness_score is not None:
                                ragas_scores['answer_correctness'] = correctness_score
                                successful_count += 1
                                if self.verbose:
                                    print(f"        ✓ answer_correctness (Custom): {correctness_score:.4f}")
                        except Exception as e:
                            if self.verbose:
                                print(f"        ⚠️  answer_correctness error: {str(e)}")
                            ragas_scores['answer_correctness'] = None
                    
                    ragas_time = time.time() - ragas_start
                    ragas_scores['_successful_count'] = successful_count
                    ragas_scores['_note'] = 'Direct LLM mode: context metrics are N/A'
                    
                    if self.verbose:
                        print(f"   ✅ Metrics computed in {ragas_time:.2f}s")
                        print(f"      (context_precision, context_recall, faithfulness = NaN for Direct LLM)")
                
                qa_result['ragas_scores'] = ragas_scores

                

            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Error: {str(e)}")
                
                qa_result = {
                    'question_id': question_id,
                    'question': question,
                    'ground_truth': ground_truth,
                    'generated_answer': "",
                    'contexts': [],
                    'sources': [],
                    'response_time': 0,
                    'iterations': 0,
                    'error': str(e),
                    'category': qa.get('category', ''),
                    'source_title': qa.get('source_title', ''),
                    'difficulty': qa.get('difficulty', ''),
                    'reasoning_type': qa.get('reasoning_type', '')
                }
            
            # Step 4: Add to results
            qa_results.append(qa_result)
            
            # Step 5: Incremental save
            if self.incremental_save:
                status = 'completed' if len(qa_results) >= total_questions else 'in_progress'
                
                if self.verbose:
                    print(f"💾 Saving progress ({len(qa_results)}/{total_questions})...")
                
                self.save_incremental_results(qa_results, mode, status, total_questions)
                
                if self.verbose:
                    print(f"   ✅ Progress saved")
            
            if self.verbose:
                print(f"✅ Question {len(qa_results)}/{total_questions} completed and saved")
        
        # Final aggregation
        if not self.incremental_save:
            # Use original batch computation
            ragas_metrics = self._compute_ragas_metrics(qa_results) if RAGAS_AVAILABLE else {}
            fact_metrics = self._compute_fact_correctness(qa_results)
            self._attach_individual_metrics(qa_results, ragas_metrics, fact_metrics)
            
            ragas_avg = ragas_metrics.get('average_score', 0.0) if ragas_metrics and ragas_metrics.get('average_score') is not None else 0.0
            fact_score = fact_metrics.get('factual_correctness', 0.0) if fact_metrics else 0.0
            weighted_score = ragas_avg * 0.3 + fact_score * 0.7 if (ragas_avg is not None and fact_score is not None) else None
            
            results = {
                'metadata': {
                    'answer_generation': {
                        'model': self.answer_model_config['model'],
                        'api_base': self.answer_model_config['api_base'],
                        'temperature': self.answer_model_config['temperature']
                    },
                    'evaluation': {
                        'ragas_model': self.eval_model_config['model'],
                        'fact_check_model': self.eval_model_config['model'],
                        'api_base': self.eval_model_config['api_base'],
                        'temperature': self.eval_model_config['temperature']
                    },
                    'mode': mode,
                    'dataset': self.dataset,
                    'test_set': self.eval_config.test_data_path.stem,
                    'num_questions': len(qa_pairs),
                    'timestamp': datetime.now().isoformat()
                },
                'qa_results': qa_results,
                'ragas_metrics': ragas_metrics,
                'fact_metrics': fact_metrics,
                'weighted_score': round(weighted_score, 4) if weighted_score is not None else None,
                'performance': self._compute_performance_metrics(qa_results, mode)
            }
        else:
            # Load the final saved results
            _, results = self.load_existing_progress(mode)
            if not results:
                # Fallback: reconstruct from qa_results
                results = {
                    'metadata': {
                        'answer_generation': {
                            'model': self.answer_model_config['model'],
                            'api_base': self.answer_model_config['api_base'],
                            'temperature': self.answer_model_config['temperature']
                        },
                        'evaluation': {
                            'ragas_model': self.eval_model_config['model'],
                            'fact_check_model': self.eval_model_config['model'],
                            'api_base': self.eval_model_config['api_base'],
                            'temperature': self.eval_model_config['temperature']
                        },
                        'mode': mode,
                        'dataset': self.dataset,
                        'test_set': self.eval_config.test_data_path.stem,
                        'num_questions': len(qa_pairs),
                        'timestamp': datetime.now().isoformat(),
                        'status': 'completed'
                    },
                    'qa_results': qa_results,
                    'ragas_metrics': self._compute_aggregate_ragas(qa_results),
                    'fact_metrics': self._compute_aggregate_fact(qa_results),
                    'performance': self._compute_performance_metrics(qa_results, mode)
                }
                # Compute weighted score
                ragas_avg = results['ragas_metrics'].get('average_score', 0.0)
                fact_score = results['fact_metrics'].get('factual_correctness', 0.0)
                results['weighted_score'] = round(ragas_avg * 0.3 + fact_score * 0.7, 4) if ragas_avg and fact_score else None
        
        return results
    
    def _run_evaluation(
        self, 
        agent: AgentOrchestrator, 
        qa_pairs: List[Dict],
        mode: str
    ) -> Dict[str, Any]:
        """
        Run evaluation for agentic modes with incremental save support
        
        Args:
            agent: Initialized agent orchestrator
            qa_pairs: List of question-answer pairs
            mode: Evaluation mode name
            
        Returns:
            Evaluation results
        """
        # Load existing progress if resume is enabled
        if self.resume:
            all_qa_results, _ = self.load_existing_progress(mode)
            
            # Dry-run mode: analyze existing results only, don't create anything
            if self.dry_run:
                mode_name = "Agentic+KG" if "with_kg" in mode else "Agentic (no KG)"
                
                # If no existing results, just report and return
                if not all_qa_results:
                    result_file = self.eval_config.result_files[mode]
                    print(f"\n{'='*60}")
                    print(f"DRY RUN ({mode_name}): No existing results found")
                    print(f"{'='*60}")
                    print(f"  📂 Expected path: {result_file}")
                    print(f"  ℹ️  Nothing to analyze - run without --dry-run to start fresh")
                    print(f"{'='*60}\n")
                    return {
                        'metadata': {'mode': mode, 'dry_run': True, 'no_existing_results': True},
                        'qa_results': []
                    }
                
                successful_results, completed_ids, errored_results = self._filter_for_resume(all_qa_results, qa_pairs)
                
                # Calculate pending questions (not yet completed successfully)
                pending_questions = []
                for idx, qa in enumerate(qa_pairs, 1):
                    qid = qa.get('id', f'qa_{idx}')
                    if qid not in completed_ids:
                        pending_questions.append(qa)
                
                print(f"\n{'='*60}")
                print(f"DRY RUN ({mode_name}): Would process {len(pending_questions)} questions")
                print(f"{'='*60}")
                print(f"  ✅ Successfully completed: {len(successful_results)}")
                print(f"  🔄 With errors (to re-generate): {len(errored_results)}")
                print(f"  📋 Not yet processed: {len(pending_questions) - len(errored_results)}")
                
                if errored_results:
                    print(f"\nQuestions to re-generate:")
                    for r in errored_results:
                        error_preview = str(r.get('error', ''))[:80]
                        print(f"  - {r.get('question_id')}: {error_preview}...")
                
                print(f"\n{'='*60}")
                print(f"To actually run, remove --dry-run flag")
                print(f"{'='*60}\n")
                
                # Return existing results without changes
                return {
                    'metadata': {'mode': mode, 'dry_run': True, 'would_process': len(pending_questions)},
                    'qa_results': all_qa_results
                }
            
            # Normal resume mode (not dry-run)
            successful_results, completed_ids, errored_results = self._filter_for_resume(all_qa_results, qa_pairs)
            qa_results = list(successful_results)
        else:
            qa_results = []
            completed_ids = set()
        
        total_questions = len(qa_pairs)
        start_idx = len(qa_results)
        
        # Use tqdm progress bar for QA processing
        mode_name = "Agentic+KG" if "with_kg" in mode else "Agentic"
        pbar = tqdm(enumerate(qa_pairs, 1), total=total_questions,
                    desc=mode_name, unit="Q",
                    disable=not TQDM_AVAILABLE,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i, qa in pbar:
            question_id = qa.get('id', f'qa_{i}')
            
            # Skip if already completed successfully (resume mode)
            if question_id in completed_ids:
                if self.verbose:
                    print(f"[{i}/{total_questions}] ⏭️  Skipping (already completed): {question_id}")
                continue
            
            question = qa['question']
            ground_truth = qa['answer']
            
            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"[{len(qa_results)+1}/{total_questions}] Processing Question {i}")
                print(f"{'─'*80}")
                print(f"📝 Question: {question[:100]}{'...' if len(question) > 100 else ''}")
            
            # Step 1: Generate answer
            start_time = time.time()
            
            try:
                if self.verbose:
                    print(f"🤖 Querying agent...")
                
                # Query the agent
                result = agent.query(question)
                response_time = time.time() - start_time
                
                # Handle both dict and string results
                if isinstance(result, str):
                    # If result is a string, treat it as the answer
                    generated_answer = result
                    contexts = []
                    sources = []
                    iterations = 0
                    quality_score = 0.0
                elif isinstance(result, dict):
                    # If result is a dict, extract all fields
                    generated_answer = result.get('final_answer', result.get('answer', ''))
                    # Extract contexts as list of strings
                    raw_contexts = result.get('contexts', [])
                    
                    # Validate raw_contexts is a list
                    if not isinstance(raw_contexts, list):
                        if self.verbose:
                            print(f"   ⚠️  contexts is not a list: {type(raw_contexts)}")
                        raw_contexts = []
                    
                    contexts = []
                    contexts_with_scores = []  # Rich format with scores
                    has_rerank_scores = False
                    for ctx in raw_contexts:
                        if isinstance(ctx, dict):
                            content = ctx.get('content', ctx.get('text', ''))
                            # Handle None values
                            if content is None:
                                content = ''
                            contexts.append(content)
                            
                            # Preserve scoring info
                            ctx_info = {'content': content}
                            if 'rerank_score' in ctx:
                                ctx_info['rerank_score'] = ctx['rerank_score']
                                has_rerank_scores = True
                            if 'similarity' in ctx:
                                ctx_info['similarity'] = ctx['similarity']
                            contexts_with_scores.append(ctx_info)
                        elif isinstance(ctx, str):
                            contexts.append(ctx)
                            contexts_with_scores.append({'content': ctx})
                        else:
                            if self.verbose:
                                print(f"   ⚠️  Skipping non-dict/str context: {type(ctx)}")
                    contexts = [c for c in contexts if c and c.strip()]
                    
                    # Validate sources is a list
                    sources = result.get('sources', [])
                    if not isinstance(sources, list):
                        sources = []
                    
                    iterations = result.get('iterations', 0)
                    quality_score = result.get('quality_score', 0.0)
                else:
                    # Unknown result type
                    generated_answer = str(result)
                    contexts = []
                    contexts_with_scores = []
                    has_rerank_scores = False
                    sources = []
                    iterations = 0
                    quality_score = 0.0
                
                qa_result = {
                    'question_id': question_id,
                    'question': question,
                    'ground_truth': ground_truth,
                    'generated_answer': generated_answer,
                    'contexts': contexts,  # Keep string list for RAGAS compatibility
                    'contexts_with_scores': contexts_with_scores,  # Rich format with rerank_score
                    'sources': sources,
                    'response_time': round(response_time, 2),
                    'iterations': iterations,
                    'quality_score': quality_score,
                    'error': None,
                    'category': qa.get('category', ''),
                    'source_title': qa.get('source_title', ''),
                    'difficulty': qa.get('difficulty', ''),
                    'reasoning_type': qa.get('reasoning_type', '')
                }
                
                if self.verbose:
                    print(f"   ✅ Answer generated in {response_time:.2f}s")
                    print(f"      Iterations: {iterations}")
                    print(f"      Quality: {quality_score:.2f}")
                    print(f"      Contexts: {len(contexts)}")
                
                # Step 2: Compute RAGAS metrics (with retry) or use placeholder in generate_only mode
                if self.generate_only:
                    if self.verbose:
                        print(f"⏭️  Skipping metrics (generate-only mode)")
                    ragas_scores = {
                        'context_precision': None,
                        'context_recall': None,
                        'faithfulness': None,
                        'answer_similarity': None,
                        'answer_relevancy': None,
                        'answer_correctness': None,
                        '_placeholder': True,
                        '_note': 'Pending evaluation (generate-only mode)'
                    }
                    qa_result['ragas_scores'] = ragas_scores
                elif RAGAS_AVAILABLE and not self.skip_ragas:
                    if self.verbose:
                        print(f"📊 Computing RAGAS metrics...")
                    
                    ragas_start = time.time()
                    ragas_scores = self._compute_ragas_single_with_retry(
                        question=question,
                        answer=qa_result['generated_answer'],
                        ground_truth=ground_truth,
                        contexts=contexts
                    )
                    ragas_time = time.time() - ragas_start
                    
                    qa_result['ragas_scores'] = ragas_scores
                    
                    if self.verbose:
                        # Performance warning for slow RAGAS
                        if ragas_time > 60:
                            print(f"   ⚠️  RAGAS took {ragas_time:.2f}s (>60s, consider optimization)")
                        print(f"   ✅ RAGAS computed in {ragas_time:.2f}s")
                        if 'error' not in ragas_scores:
                            for metric, value in ragas_scores.items():
                                if value is not None:
                                    # Only format as float if it's a number, not a list or dict
                                    if isinstance(value, (int, float)):
                                        print(f"      {metric}: {value:.4f}")
                                    else:
                                        print(f"      {metric}: {value}")
                                else:
                                    print(f"      {metric}: None (unavailable)")
                elif self.skip_ragas and self.verbose:
                    print(f"⏭️  Skipping RAGAS evaluation (--skip-ragas enabled)")
                
                # Note: fact_scores removed - answer_correctness is now computed via CustomAnswerCorrectness
                # in _compute_ragas_single and included in ragas_scores
                

            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Error: {str(e)}")
                    # Print full traceback for debugging
                    import traceback
                    print(f"   Detailed traceback:")
                    traceback.print_exc()
                
                qa_result = {
                    'question_id': question_id,
                    'question': question,
                    'ground_truth': ground_truth,
                    'generated_answer': "",
                    'contexts': [],
                    'sources': [],
                    'response_time': 0,
                    'iterations': 0,
                    'quality_score': 0.0,
                    'error': str(e),
                    'category': qa.get('category', ''),
                    'source_title': qa.get('source_title', ''),
                    'difficulty': qa.get('difficulty', ''),
                    'reasoning_type': qa.get('reasoning_type', '')
                }
            
            # Step 4: Add to results
            qa_results.append(qa_result)
            
            # Step 5: Incremental save
            if self.incremental_save:
                status = 'completed' if len(qa_results) >= total_questions else 'in_progress'
                
                if self.verbose:
                    print(f"💾 Saving progress ({len(qa_results)}/{total_questions})...")
                
                self.save_incremental_results(qa_results, mode, status, total_questions)
                
                if self.verbose:
                    print(f"   ✅ Progress saved")
            
            if self.verbose:
                print(f"✅ Question {len(qa_results)}/{total_questions} completed and saved")
        
        # Final aggregation (if not using incremental, or to finalize)
        if not self.incremental_save:
            # Use original batch computation
            ragas_metrics = self._compute_ragas_metrics(qa_results) if RAGAS_AVAILABLE else {}
            fact_metrics = self._compute_fact_correctness(qa_results)
            self._attach_individual_metrics(qa_results, ragas_metrics, fact_metrics)
            
            # Weighted score is just the average of the 6 integrated metrics
            weighted_score = ragas_metrics.get('average_score')
            
            results = {
                'metadata': {
                    'answer_generation': {
                        'model': self.answer_model_config['model'],
                        'api_base': self.answer_model_config['api_base'],
                        'temperature': self.answer_model_config['temperature']
                    },
                    'evaluation': {
                        'ragas_model': self.eval_model_config['model'],
                        'fact_check_model': self.eval_model_config['model'],
                        'api_base': self.eval_model_config['api_base'],
                        'temperature': self.eval_model_config['temperature']
                    },
                    'mode': mode,
                    'dataset': self.dataset,
                    'test_set': self.eval_config.test_data_path.stem,
                    'num_questions': len(qa_pairs),
                    'timestamp': datetime.now().isoformat(),
                    'kg_enabled': agent.enable_kg
                },
                'qa_results': qa_results,
                'ragas_metrics': ragas_metrics,
                # 'fact_metrics': fact_metrics, # Hidden as requested
                'weighted_score': round(weighted_score, 4) if weighted_score is not None else None,
                'performance': self._compute_performance_metrics(qa_results, mode)
            }
        else:
            # Load the final saved results
            _, results = self.load_existing_progress(mode)
            if not results:
                # Fallback: reconstruct from qa_results
                results = {
                    'metadata': {
                        'answer_generation': {
                            'model': self.answer_model_config['model'],
                            'api_base': self.answer_model_config['api_base'],
                            'temperature': self.answer_model_config['temperature']
                        },
                        'evaluation': {
                            'ragas_model': self.eval_model_config['model'],
                            'fact_check_model': self.eval_model_config['model'],
                            'api_base': self.eval_model_config['api_base'],
                            'temperature': self.eval_model_config['temperature']
                        },
                        'mode': mode,
                        'dataset': self.dataset,
                        'test_set': self.eval_config.test_data_path.stem,
                        'num_questions': len(qa_pairs),
                        'timestamp': datetime.now().isoformat(),
                        'kg_enabled': agent.enable_kg,
                        'status': 'completed'
                    },
                    'qa_results': qa_results,
                    'ragas_metrics': self._compute_aggregate_ragas(qa_results),
                    'fact_metrics': self._compute_aggregate_fact(qa_results),
                    'performance': self._compute_performance_metrics(qa_results, mode)
                }
                # Compute weighted score
                ragas_avg = results['ragas_metrics'].get('average_score', 0.0)
                fact_score = results['fact_metrics'].get('factual_correctness', 0.0)
                results['weighted_score'] = round(ragas_avg * 0.3 + fact_score * 0.7, 4) if ragas_avg and fact_score else None
        
        return results
    
    def _compute_ragas_metrics(self, qa_results: List[Dict]) -> Dict[str, float]:
        """
        Compute RAGAS evaluation metrics
        
        Args:
            qa_results: List of QA results
            
        Returns:
            Dictionary of RAGAS metrics
        """
        if not RAGAS_AVAILABLE:
            return {}
        
        if self.verbose:
            print(f"\n📊 Computing RAGAS metrics...")
        
        try:
            # Filter out results with errors
            valid_results = [r for r in qa_results if not r.get('error')]
            
            if not valid_results:
                if self.verbose:
                    print("  ⚠️  No valid results to evaluate")
                return {}
            
            # Prepare data for RAGAS
            # Check if we have any real contexts (to determine if this is a retrieval-based system)
            has_real_contexts = any(r['contexts'] for r in valid_results)
            
            if has_real_contexts:
                # For RAG systems: use actual contexts or empty placeholder
                data = {
                    'question': [r['question'] for r in valid_results],
                    'answer': [r['generated_answer'] for r in valid_results],
                    'contexts': [r['contexts'] if r['contexts'] else ['[No context retrieved for this question]'] for r in valid_results],
                    'ground_truth': [r['ground_truth'] for r in valid_results]
                }
            else:
                # For Direct LLM: provide minimal placeholder to avoid RAGAS errors
                # The context-based metrics will still be 0 or NaN, which is expected
                data = {
                    'question': [r['question'] for r in valid_results],
                    'answer': [r['generated_answer'] for r in valid_results],
                    'contexts': [[''] for r in valid_results],  # Empty context for each question
                    'ground_truth': [r['ground_truth'] for r in valid_results]
                }
            
            # Create dataset
            dataset = Dataset.from_dict(data)
            
            # Initialize LLM for RAGAS evaluation (uses eval_model_config from evaluation/.env)
            ragas_llm = ChatOpenAI(
                model=self.eval_model_config['model'],
                temperature=self.eval_model_config['temperature'],
                openai_api_key=self.eval_model_config['api_key'],
                openai_api_base=self.eval_model_config['api_base']
            )
            
            # Use RAGAS metrics for comprehensive evaluation
            # Note: Most metrics are now custom, only 2 RAGAS metrics remain
            metrics = [
                context_recall,
                answer_similarity
            ]
            
            # Run evaluation
            eval_result = evaluate(
                dataset,
                metrics=metrics,
                llm=ragas_llm
            )
            
            # Extract metrics - RAGAS returns scores as list of dicts
            # Compute average for each metric across all samples
            import math
            ragas_metrics = {}
            individual_scores = []  # Store individual scores for each question
            
            if eval_result.scores:
                # Get all metric names from first score
                metric_names = eval_result.scores[0].keys()
                
                # Store individual scores with detailed NaN logging
                nan_issues = []  # Track NaN issues for reporting
                for i, score in enumerate(eval_result.scores):
                    individual_score = {}
                    for metric_name in metric_names:
                        value = score.get(metric_name)
                        # Convert NaN to None for JSON serialization
                        if value is not None and not (isinstance(value, float) and math.isnan(value)):
                            individual_score[metric_name] = value
                        else:
                            individual_score[metric_name] = None
                            # Log NaN issues
                            if i < len(valid_results):
                                question_preview = valid_results[i]['question'][:80] + "..." if len(valid_results[i]['question']) > 80 else valid_results[i]['question']
                                nan_issues.append({
                                    'question_idx': i + 1,
                                    'metric': metric_name,
                                    'question': question_preview
                                })
                    individual_scores.append(individual_score)
                
                # Report NaN issues if verbose
                if self.verbose and nan_issues:
                    print(f"\n  ⚠️  Found {len(nan_issues)} NaN/None values in RAGAS metrics:")
                    for issue in nan_issues[:5]:  # Show first 5 issues
                        print(f"     Question #{issue['question_idx']}: {issue['metric']} = None")
                        print(f"       \"{issue['question']}\"")
                    if len(nan_issues) > 5:
                        print(f"     ... and {len(nan_issues) - 5} more issues")
                
                # Compute average for each metric (filtering out None/NaN values)
                for metric_name in metric_names:
                    values = []
                    for score in eval_result.scores:
                        value = score.get(metric_name)
                        # Only include valid numeric values (not None, not NaN)
                        if value is not None and not (isinstance(value, float) and math.isnan(value)):
                            values.append(value)
                    
                    if values:
                        ragas_metrics[metric_name] = sum(values) / len(values)
                    else:
                        ragas_metrics[metric_name] = None
            
            # Compute average score (only from valid metrics)
            if ragas_metrics:
                valid_values = [v for v in ragas_metrics.values() if v is not None and not (isinstance(v, float) and math.isnan(v))]
                if valid_values:
                    ragas_metrics['average_score'] = sum(valid_values) / len(valid_values)
                else:
                    ragas_metrics['average_score'] = None
            
            # Store individual scores for later attachment to qa_results
            ragas_metrics['_individual_scores'] = individual_scores
            
            if self.verbose:
                print("  ✅ RAGAS metrics computed successfully")
                for metric, value in ragas_metrics.items():
                    if metric != '_individual_scores' and value is not None:
                        if isinstance(value, (int, float)):
                            print(f"     {metric}: {value:.4f}")
                        else:
                            print(f"     {metric}: {value}")
            
            return ragas_metrics
        
        except Exception as e:
            if self.verbose:
                import traceback
                print(f"  ⚠️  Error computing RAGAS metrics: {str(e)}")
                print(f"     Traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def _attach_individual_metrics(self, qa_results: List[Dict], ragas_metrics: Dict, fact_metrics: Dict):
        """
        Attach individual evaluation metrics to each qa_result
        
        Args:
            qa_results: List of QA results to attach metrics to
            ragas_metrics: RAGAS metrics including _individual_scores
            fact_metrics: Fact correctness metrics including _individual_details
        """
        # Attach RAGAS individual scores
        if ragas_metrics and '_individual_scores' in ragas_metrics:
            individual_ragas = ragas_metrics['_individual_scores']
            for i, qa_result in enumerate(qa_results):
                if i < len(individual_ragas) and not qa_result.get('error'):
                    qa_result['ragas_scores'] = individual_ragas[i]
        
        # Attach Fact Correctness individual details
        if fact_metrics and '_individual_details' in fact_metrics:
            individual_facts = fact_metrics['_individual_details']
            for i, qa_result in enumerate(qa_results):
                if i < len(individual_facts) and not qa_result.get('error'):
                    qa_result['fact_scores'] = individual_facts[i]
        
        # Remove the _individual_ keys from the aggregate metrics (they shouldn't be in final output)
        if ragas_metrics and '_individual_scores' in ragas_metrics:
            del ragas_metrics['_individual_scores']
        if fact_metrics and '_individual_details' in fact_metrics:
            del fact_metrics['_individual_details']
    
    def _compute_fact_correctness(self, qa_results: List[Dict]) -> Dict[str, Any]:
        """
        Compute factual correctness metrics using custom fact checker.
        
        Args:
            qa_results: List of QA results
            
        Returns:
            Dictionary of fact-checking metrics
        """
        if self.verbose:
            print(f"\n📊 Computing Fact Correctness metrics...")
        
        try:
            # Filter out results with errors
            valid_results = [r for r in qa_results if not r.get('error')]
            
            if not valid_results:
                if self.verbose:
                    print("  ⚠️  No valid results to evaluate")
                return {}
            
            # Collect fact-checking results for all Q&A pairs
            all_fact_metrics = []
            
            for i, result in enumerate(valid_results, 1):
                if self.verbose:
                    print(f"  [{i}/{len(valid_results)}] Fact-checking answer...")
                
                fact_metrics = self.fact_checker.compute_factual_correctness(
                    question=result['question'],
                    generated_answer=result['generated_answer'],
                    ground_truth=result['ground_truth'],
                    contexts=result.get('contexts', [])
                )
                
                all_fact_metrics.append(fact_metrics)
            
            # Aggregate metrics across all results
            if not all_fact_metrics:
                return {}
            
            # Prepare individual details (simplified for storage)
            individual_details = []
            for m in all_fact_metrics:
                detail = {
                    'factual_correctness': round(m['factual_correctness'], 4),
                    'key_facts_coverage': round(m['key_facts_coverage'], 4),
                    'fact_accuracy': round(m['fact_accuracy'], 4),
                    'fact_precision': round(m['fact_precision'], 4),
                    'fact_recall': round(m['fact_recall'], 4),
                    'hallucination_rate': round(m['hallucination_rate'], 4),
                    'context_supported_rate': round(m['context_supported_rate'], 4),
                    'critical_errors_count': len(m['critical_errors']),
                    'moderate_errors': m['moderate_errors'],
                    'minor_errors': m['minor_errors']
                }
                individual_details.append(detail)
            
            # Compute averages
            aggregated = {
                'factual_correctness': sum(m['factual_correctness'] for m in all_fact_metrics) / len(all_fact_metrics),
                'key_facts_coverage': sum(m['key_facts_coverage'] for m in all_fact_metrics) / len(all_fact_metrics),
                'fact_accuracy': sum(m['fact_accuracy'] for m in all_fact_metrics) / len(all_fact_metrics),
                'fact_precision': sum(m['fact_precision'] for m in all_fact_metrics) / len(all_fact_metrics),
                'fact_recall': sum(m['fact_recall'] for m in all_fact_metrics) / len(all_fact_metrics),
                'hallucination_rate': sum(m['hallucination_rate'] for m in all_fact_metrics) / len(all_fact_metrics),
                'context_supported_rate': sum(m['context_supported_rate'] for m in all_fact_metrics) / len(all_fact_metrics),
                'total_critical_errors': sum(len(m['critical_errors']) for m in all_fact_metrics),
                'total_moderate_errors': sum(m['moderate_errors'] for m in all_fact_metrics),
                'total_minor_errors': sum(m['minor_errors'] for m in all_fact_metrics),
                '_individual_details': individual_details  # Store for later attachment
            }
            
            # Round all values
            for key in aggregated:
                if isinstance(aggregated[key], float):
                    aggregated[key] = round(aggregated[key], 4)
            
            if self.verbose:
                print("  ✅ Fact Correctness metrics computed successfully")
                print(f"     Factual Correctness: {aggregated['factual_correctness']:.4f}")
                print(f"     Key Facts Coverage: {aggregated['key_facts_coverage']:.4f}")
                print(f"     Critical Errors: {aggregated['total_critical_errors']}")
            
            return aggregated
        
        except Exception as e:
            if self.verbose:
                import traceback
                print(f"  ⚠️  Error computing Fact Correctness: {str(e)}")
                print(f"     Traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def save_results(self, results: Dict[str, Any], mode: str):
        """
        Save evaluation results to JSON file
        
        Args:
            results: Evaluation results
            mode: Evaluation mode
        """
        # Skip saving in dry-run mode (prevents directory creation)
        if self.dry_run:
            return
        
        import math
        
        # Custom JSON encoder to handle NaN values
        class NaNEncoder(json.JSONEncoder):
            def encode(self, obj):
                if isinstance(obj, float):
                    if math.isnan(obj):
                        return 'null'
                    elif math.isinf(obj):
                        return 'null'
                return super().encode(obj)
            
            def iterencode(self, obj, _one_shot=False):
                """Encode object, converting NaN/Inf to None"""
                return super().iterencode(self._convert_nan(obj), _one_shot)
            
            def _convert_nan(self, obj):
                """Recursively convert NaN/Inf to None"""
                if isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return None
                    return obj
                elif isinstance(obj, dict):
                    return {k: self._convert_nan(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [self._convert_nan(item) for item in obj]
                return obj
        
        output_file = self.eval_config.result_files[mode]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NaNEncoder)
        
        if self.verbose:
            print(f"\n💾 Results saved to: {output_file}")
    
    def generate_summary(self, all_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Generate summary comparing all evaluation modes
        
        Args:
            all_results: Dictionary mapping mode names to results
            
        Returns:
            Summary dictionary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset': self.dataset,
            'test_set': self.eval_config.test_data_path.stem,
            'num_questions': self.eval_config.num_test_questions,
            'answer_generation': {
                'model': self.answer_model_config['model'],
                'api_base': self.answer_model_config['api_base'],
                'temperature': self.answer_model_config['temperature']
            },
            'evaluation': {
                'ragas_model': self.eval_model_config['model'],
                'fact_check_model': self.eval_model_config['model'],
                'api_base': self.eval_model_config['api_base'],
                'temperature': self.eval_model_config['temperature']
            },
            'reranker': {
                'enabled': self.eval_config.enable_reranking,
                'model': self.eval_config.reranker_model if self.eval_config.enable_reranking else None
            },
            'comparison': {}
        }
        
        for mode, results in all_results.items():
            summary['comparison'][mode] = {
                'ragas_metrics': results.get('ragas_metrics', {}),
                'performance': results.get('performance', {}),
                'metadata': results.get('metadata', {})
            }
        
        # Save summary
        summary_file = self.eval_config.summary_file
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"\n📋 Evaluation summary saved to: {summary_file}")
        
        return summary
    
    def print_summary(self, all_results: Dict[str, Dict]):
        """
        Print formatted summary of all evaluations
        
        Args:
            all_results: Dictionary mapping mode names to results
        """
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        
        # Print comparison table
        modes = list(all_results.keys())
        
        if RAGAS_AVAILABLE:
            print("RAGAS Metrics:")
            print(f"{'─'*80}")
            
            # Get all metrics from first mode
            first_mode = modes[0]
            metrics = all_results[first_mode].get('ragas_metrics', {})
            
            if metrics and 'error' not in metrics:
                # Print header
                print(f"{'Metric':<30} ", end="")
                for mode in modes:
                    mode_name = mode.replace('_', ' ').title()[:15]
                    print(f"{mode_name:>15} ", end="")
                print()
                print(f"{'─'*80}")
                
                # Print each metric
                for metric in metrics.keys():
                    if metric != 'error':
                        print(f"{metric:<30} ", end="")
                        for mode in modes:
                            value = all_results[mode].get('ragas_metrics', {}).get(metric, 0)
                            # Handle None values (from NaN in RAGAS)
                            if value is None:
                                print(f"{'N/A':>15} ", end="")
                            else:
                                print(f"{value:>15.4f} ", end="")
                        print()
            else:
                print("  ⚠️  RAGAS metrics not available")
        
        print(f"\n{'─'*80}")
        print("Performance Metrics:")
        print(f"{'─'*80}")
        
        # Print header
        print(f"{'Metric':<30} ", end="")
        for mode in modes:
            mode_name = mode.replace('_', ' ').title()[:15]
            print(f"{mode_name:>15} ", end="")
        print()
        print(f"{'─'*80}")
        
        # Print performance metrics
        perf_metrics = ['avg_response_time', 'total_time', 'error_count']
        for metric in perf_metrics:
            print(f"{metric:<30} ", end="")
            for mode in modes:
                value = all_results[mode].get('performance', {}).get(metric, 0)
                print(f"{value:>15.2f} ", end="")
            print()
        
        # Add agentic-specific metrics
        if any('agentic' in mode for mode in modes):
            print(f"{'avg_iterations':<30} ", end="")
            for mode in modes:
                if 'agentic' in mode:
                    value = all_results[mode].get('performance', {}).get('avg_iterations', 0)
                    print(f"{value:>15.2f} ", end="")
                else:
                    print(f"{'N/A':>15} ", end="")
            print()
        
        print(f"{'='*80}\n")


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(description='Evaluate RAG systems with incremental save')
    parser.add_argument('--dataset', type=str, 
                       choices=['all', 'computational', 'characterization', 'stability', 
                               'materials', 'device', 'structure', 'processing', 'performance'],
                       default='computational',
                       help='Dataset to evaluate (default: computational)')
    parser.add_argument('--mode', type=str, choices=['all', 'agentic_with_kg', 'agentic_no_kg', 'direct_llm'],
                       default='all', help='Evaluation mode to run')
    parser.add_argument('--num_questions', type=str, default='10',
                       help='Number of questions to test (or "all")')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing progress (skip completed questions, re-generate errored ones)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview which questions would be re-generated (use with --resume)')
    parser.add_argument('--no-incremental', action='store_true',
                       help='Disable incremental save (use batch mode)')
    parser.add_argument('--skip-ragas', action='store_true',
                       help='Skip RAGAS evaluation (only compute fact-checking metrics)')
    
    # Two-phase evaluation workflow
    parser.add_argument('--generate-only', action='store_true',
                       help='Phase 1: Generate answers only, skip all evaluation metrics (use placeholders)')
    parser.add_argument('--eval-only', action='store_true',
                       help='Phase 2: Compute evaluation metrics on existing answers (uses repair_null_metrics internally)')
    
    # Answer generation model configuration
    parser.add_argument('--answer-model', type=str, default=None,
                       help='Model for answer generation (default: from config)')
    parser.add_argument('--answer-api-base', type=str, default=None,
                       help='API base URL for answer generation')
    parser.add_argument('--answer-api-key', type=str, default=None,
                       help='API key for answer generation')
    parser.add_argument('--answer-temperature', type=float, default=None,
                       help='Temperature for answer generation')
    
    # Evaluation model configuration
    parser.add_argument('--eval-model', type=str, default=None,
                       help='Model for evaluation (RAGAS and fact-check)')
    parser.add_argument('--eval-api-base', type=str, default=None,
                       help='API base URL for evaluation')
    parser.add_argument('--eval-api-key', type=str, default=None,
                       help='API key for evaluation')
    parser.add_argument('--eval-temperature', type=float, default=0.0,
                       help='Temperature for evaluation (default: 0.0 for consistency)')
    
    # Results output configuration
    parser.add_argument('--results-subdir', type=str, default=None,
                       help='Subdirectory for results (e.g., model name). Results go to results/<subdir>/<dataset>/')
    
    args = parser.parse_args()
    
    # --dry-run implies --resume (need to load existing results to analyze errors)
    if args.dry_run and not args.resume:
        args.resume = True
        print("ℹ️  --dry-run implies --resume, automatically enabled")
    
    # Update config
    if args.num_questions.lower() == 'all':
        eval_config.test_mode = 'all'
    else:
        eval_config.num_test_questions = int(args.num_questions)
    
    # Build answer generation model configuration (from agentic_rag/.env)
    # Note: agentic_config loads from agentic_rag/.env, which contains the actual model used for answer generation
    answer_model_config = {
        'model': args.answer_model or agentic_config.llm_model,
        'api_key': args.answer_api_key or agentic_config.openai_api_key,
        'api_base': args.answer_api_base or agentic_config.openai_base_url,
        'temperature': args.answer_temperature if args.answer_temperature is not None else agentic_config.llm_temperature
    }

    # Build evaluation model configuration (same as answer model by default)
    eval_model_config = {
        'model': args.eval_model or eval_env.llm_model,
        'api_key': args.eval_api_key or eval_env.llm_api_key,
        'api_base': args.eval_api_base or eval_env.llm_base_url,
        'temperature': args.eval_temperature
    }
    
    # Initialize evaluator with selected dataset and model configurations
    evaluator = RAGEvaluator(
        dataset=args.dataset, 
        verbose=args.verbose,
        incremental_save=not args.no_incremental,
        resume=args.resume,
        answer_model_config=answer_model_config,
        eval_model_config=eval_model_config,
        skip_ragas=args.skip_ragas,
        results_subdir=args.results_subdir,
        generate_only=args.generate_only,
        eval_only=args.eval_only,
        dry_run=args.dry_run
    )
    
    # Load test data
    qa_pairs = evaluator.load_test_data()
    
    # Run evaluations
    all_results = {}
    
    modes_to_run = []
    if args.mode == 'all':
        modes_to_run = ['agentic_with_kg', 'agentic_no_kg', 'direct_llm']
    else:
        modes_to_run = [args.mode]
    
    for mode in modes_to_run:
        if mode == 'agentic_with_kg':
            results = evaluator.evaluate_agentic_with_kg(qa_pairs)
        elif mode == 'agentic_no_kg':
            results = evaluator.evaluate_agentic_no_kg(qa_pairs)
        elif mode == 'direct_llm':
            results = evaluator.evaluate_direct_llm(qa_pairs)
        
        # Save results (if not already saved incrementally and not in dry-run mode)
        if not evaluator.incremental_save and not evaluator.dry_run:
            evaluator.save_results(results, mode)
        
        all_results[mode] = results
    
    # Generate and print summary (skip if dry-run mode)
    if len(all_results) > 1 and not evaluator.dry_run:
        evaluator.generate_summary(all_results)
        evaluator.print_summary(all_results)
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()

