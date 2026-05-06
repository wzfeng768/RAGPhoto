"""
Evaluation Configuration
Loads settings from parent agentic_rag config and adds evaluation-specific settings
"""

import sys
from pathlib import Path

# Add parent directory to path to import agentic_rag
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "agentic_rag"))

from config.config import config as agentic_config


# Available datasets mapping
AVAILABLE_DATASETS = {
    'all': 'all_qa_pairs.json',
    'computational': 'qa_by_category/computational_machine_learning.json',
    'characterization': 'qa_by_category/characterization_methods.json',
    'stability': 'qa_by_category/stability_degradation.json',
    'materials': 'qa_by_category/materials_design_synthesis.json',
    'device': 'qa_by_category/device_architecture_physics.json',
    'structure': 'qa_by_category/structure-property_relationships.json',
    'processing': 'qa_by_category/processing_fabrication.json',
    'performance': 'qa_by_category/performance_metrics.json'
}


class EvaluationConfig:
    """Configuration for RAG evaluation"""
    
    def __init__(self):
        """Initialize evaluation configuration"""
        # Inherit from agentic_rag config
        self.openai_api_key = agentic_config.openai_api_key
        self.openai_base_url = agentic_config.openai_base_url
        self.llm_model = agentic_config.llm_model
        
        # Evaluation-specific settings
        self.data_root = project_root / "photorag_QA" / "QA" / "final"
        self.results_dir = Path(__file__).parent / "results"
        
        # Current dataset (default to computational)
        self.current_dataset = 'computational'
        self.test_data_path = None  # Will be set by set_dataset()
        
        # Test settings
        self.num_test_questions = 10  # Number of questions to evaluate
        self.test_mode = "first_n"  # "first_n", "random", or "all"
        
        # RAGAS evaluation model (can be different from answering model)
        self.ragas_model = agentic_config.llm_model
        self.ragas_temperature = 0.0  # Use 0 for consistency in evaluation
        
        # Agent settings for different modes
        self.max_iterations = agentic_config.max_iterations
        self.quality_threshold = agentic_config.quality_threshold
        self.verbose = False  # Set to False to reduce output clutter during batch evaluation
        
        # Reranker settings
        self.reranker_model = getattr(agentic_config, 'reranker_model', 'qwen3-rerank')
        self.enable_reranking = getattr(agentic_config, 'enable_reranking', True)
        
        # Result file names (will be updated by set_dataset)
        self.result_files = {}
        self.summary_file = None
        
        # Set default dataset
        self.set_dataset('computational')
    
    def set_dataset(self, dataset_name: str, results_subdir: str = None):
        """
        Set the current dataset for evaluation
        
        Args:
            dataset_name: Name of the dataset (key from AVAILABLE_DATASETS)
            results_subdir: Optional subdirectory for results (e.g., model name)
                           If provided, results go to results/<results_subdir>/<dataset>/
                           If not provided, results go to results/<dataset>/
        
        Raises:
            ValueError: If dataset_name is not in AVAILABLE_DATASETS
        """
        if dataset_name not in AVAILABLE_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {list(AVAILABLE_DATASETS.keys())}"
            )
        
        self.current_dataset = dataset_name
        self.test_data_path = self.data_root / AVAILABLE_DATASETS[dataset_name]
        
        # Update result paths to include optional results_subdir and dataset subdirectory
        if results_subdir:
            dataset_results_dir = self.results_dir / results_subdir / dataset_name
        else:
            dataset_results_dir = self.results_dir / dataset_name
        
        self.result_files = {
            'agentic_with_kg': dataset_results_dir / "agentic_with_kg" / "results.json",
            'agentic_no_kg': dataset_results_dir / "agentic_no_kg" / "results.json",
            'direct_llm': dataset_results_dir / "direct_llm" / "results.json"
        }
        self.summary_file = dataset_results_dir / "evaluation_summary.json"
    
    def validate(self):
        """Validate configuration"""
        if self.test_data_path is None:
            raise ValueError("Dataset not set. Call set_dataset() first.")
        
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.test_data_path}")
        
        # Note: Results directories are created on-demand in save_results() 
        # to respect --results-subdir and --dry-run options
        
        return True


# Global evaluation config instance
eval_config = EvaluationConfig()

