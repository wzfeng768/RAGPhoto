#!/usr/bin/env python3
"""
Custom Evaluation Metrics Module
Implements custom evaluation metrics as stable alternatives to RAGAS

This module provides:
- CustomAnswerRelevancy: Semantic similarity + LLM-based relevancy scoring
- CustomFaithfulness: Evaluates answer faithfulness to retrieved contexts
- CustomAnswerCorrectness: Evaluates factual correctness against ground truth

NOTE: This module does NOT import any configuration from agentic_rag.
All configurations must be passed explicitly via model_config and embedding_config parameters.
"""

import numpy as np
import re
from typing import Dict, List, Optional
from openai import OpenAI


def _extract_score_from_response(response) -> Optional[float]:
    """
    Extract a numeric score from LLM response.
    Handles reasoning models (like glm-4.7) that may put content in reasoning_content.
    
    Returns:
        Float score between 0.0-1.0 or None if extraction failed
    """
    # First try main content
    message = response.choices[0].message
    content = message.content
    
    if content and content.strip():
        text = content.strip()
    else:
        # Fallback: try reasoning_content for reasoning models
        reasoning = getattr(message, 'reasoning_content', None)
        if reasoning:
            text = reasoning
        else:
            return None
    
    # Try to parse as float directly
    try:
        score = float(text)
        return max(0.0, min(1.0, score))
    except ValueError:
        pass
    
    # Try to extract first decimal number from text
    numbers = re.findall(r'(\d+\.?\d*)', text)
    if numbers:
        try:
            score = float(numbers[-1])  # Try last number (usually the final answer)
            return max(0.0, min(1.0, score))
        except ValueError:
            pass
    
    return None

class CustomAnswerRelevancy:
    """
    Custom Answer Relevancy Metric
    
    Evaluates how relevant an answer is to the given question using:
    1. Semantic similarity between question and answer embeddings
    2. LLM-based relevancy scoring
    
    The final score is a weighted combination of both approaches.
    """
    
    def __init__(self, model_config: dict, embedding_config: dict = None, 
                 verbose: bool = False, semantic_weight: float = 0.4, 
                 llm_weight: float = 0.6):
        """
        Initialize Custom Answer Relevancy metric
        
        Args:
            model_config: Configuration for LLM (REQUIRED)
                {
                    'model': 'glm-4.7',
                    'api_key': 'sk-...',
                    'api_base': 'https://...',
                    'temperature': 0.0
                }
            embedding_config: Configuration for Embedding API (optional, defaults to model_config)
                {
                    'model': 'text-embedding-3-small',
                    'api_key': 'sk-...',
                    'api_base': 'https://...'
                }
            verbose: Enable verbose output
            semantic_weight: Weight for semantic similarity (default: 0.4)
            llm_weight: Weight for LLM relevancy score (default: 0.6)
        """
        if model_config is None:
            raise ValueError("model_config is required for CustomAnswerRelevancy")
        
        self.verbose = verbose
        self.semantic_weight = semantic_weight
        self.llm_weight = llm_weight
        self.model_config = model_config
        
        # Embedding configuration (use provided or fallback to model_config)
        if embedding_config:
            self.embedding_model = embedding_config.get('model', 'text-embedding-3-small')
            self.embedding_api_key = embedding_config.get('api_key', model_config['api_key'])
            self.embedding_base_url = embedding_config.get('api_base', model_config['api_base'])
        else:
            # Fallback: use model_config for embedding (same API endpoint)
            self.embedding_model = 'text-embedding-3-small'
            self.embedding_api_key = model_config['api_key']
            self.embedding_base_url = model_config['api_base']
        
        # Initialize OpenAI client for LLM
        self.llm_client = OpenAI(
            api_key=self.model_config['api_key'],
            base_url=self.model_config['api_base']
        )
        
        # Initialize embedding client (may use different endpoint)
        self.embedding_client = OpenAI(
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url
        )
        
        if self.verbose:
            print(f"✅ CustomAnswerRelevancy initialized")
            print(f"   Semantic Weight: {self.semantic_weight}")
            print(f"   LLM Weight: {self.llm_weight}")
            print(f"   LLM Model: {self.model_config['model']}")
            print(f"   Embedding Model: {self.embedding_model}")
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Truncate long text to avoid token limits
            max_chars = 8000  # Approximate limit for embedding models
            truncated_text = text[:max_chars] if len(text) > max_chars else text
            
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=truncated_text
            )
            return response.data[0].embedding
        except Exception as e:
            if self.verbose:
                print(f"      ⚠️  Embedding error: {str(e)}")
            return None
    
    def compute_semantic_similarity(self, question: str, answer: str) -> Optional[float]:
        """
        Compute semantic similarity between question and answer
        
        Uses cosine similarity of embedding vectors
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Similarity score (0-1) or None if computation failed
        """
        # Get embeddings
        question_embedding = self._get_embedding(question)
        answer_embedding = self._get_embedding(answer)
        
        if question_embedding is None or answer_embedding is None:
            return None
        
        # Convert to numpy arrays
        q_vec = np.array(question_embedding)
        a_vec = np.array(answer_embedding)
        
        # Compute cosine similarity
        dot_product = np.dot(q_vec, a_vec)
        norm_q = np.linalg.norm(q_vec)
        norm_a = np.linalg.norm(a_vec)
        
        if norm_q == 0 or norm_a == 0:
            return 0.0
        
        similarity = dot_product / (norm_q * norm_a)
        
        # Clamp to [0, 1] range (cosine similarity can be negative)
        similarity = max(0.0, min(1.0, (similarity + 1) / 2))
        
        return float(similarity)
    
    def compute_llm_relevancy(self, question: str, answer: str, max_retries: int = 3) -> Optional[float]:
        """
        Use LLM to evaluate answer relevancy with retry mechanism
        
        Args:
            question: Question text
            answer: Answer text
            max_retries: Maximum number of retry attempts
            
        Returns:
            Relevancy score (0-1) or None if computation failed
        """
        import time
        import re
        
        prompt = f"""You are evaluating how relevant an answer is to a question.

Question: {question}

Answer: {answer}

Task: Rate the relevancy of the answer to the question.

Scoring Guide:
- 0.0-0.3: Irrelevant, does not address the question
- 0.4-0.6: Partially relevant, addresses some aspects
- 0.7-0.9: Mostly relevant, addresses the main question
- 1.0: Highly relevant, directly and fully answers the question

Please respond with ONLY a single number between 0.0 and 1.0."""

        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_config['model'],
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Score the answer relevancy. Respond with ONLY a single decimal number between 0.0 and 1.0, nothing else."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.model_config['temperature'],
                    max_tokens=2000,
                    timeout=30
                )
                
                # Parse the response - handle empty or None
                raw_content = response.choices[0].message.content
                if raw_content is None or not raw_content.strip():
                    if self.verbose:
                        print(f"      ⚠️  LLM returned empty response, retrying...")
                    continue
                result = raw_content.strip()
                
                # Extract number from response
                try:
                    score = float(result)
                    score = max(0.0, min(1.0, score))
                    return score
                except ValueError:
                    # Try to extract first number from response
                    numbers = re.findall(r'(\d+\.?\d*)', result)
                    if numbers:
                        score = float(numbers[0])
                        return max(0.0, min(1.0, score))
                    
                    if self.verbose:
                        print(f"      ⚠️  Failed to parse LLM response: {result}")
                    # Continue to next attempt
                    
            except Exception as e:
                if self.verbose and attempt == max_retries - 1:
                    print(f"      ⚠️  LLM relevancy error after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    time.sleep(wait_time)
        
        return None
    
    def compute(self, question: str, answer: str) -> Dict[str, Optional[float]]:
        """
        Compute custom answer relevancy metric
        
        Combines semantic similarity and LLM-based scoring
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Dictionary containing:
            - custom_answer_relevancy: Combined score (0-1)
            - semantic_similarity: Raw semantic similarity score
            - llm_relevancy: Raw LLM relevancy score
        """
        if self.verbose:
            print(f"      📊 Computing custom answer relevancy...")
        
        # Handle empty inputs
        if not question or not answer or not question.strip() or not answer.strip():
            return {
                'custom_answer_relevancy': 0.0,
                'semantic_similarity': 0.0,
                'llm_relevancy': 0.0,
                '_error': 'empty_input'
            }
        
        # Compute individual scores
        semantic_score = self.compute_semantic_similarity(question, answer)
        llm_score = self.compute_llm_relevancy(question, answer)
        
        if self.verbose:
            if semantic_score is not None:
                print(f"      ✓ Semantic similarity: {semantic_score:.4f}")
            else:
                print(f"      ⚠️  Semantic similarity: Failed")
            
            if llm_score is not None:
                print(f"      ✓ LLM relevancy: {llm_score:.4f}")
            else:
                print(f"      ⚠️  LLM relevancy: Failed")
        
        # Compute combined score
        combined_score = None
        
        if semantic_score is not None and llm_score is not None:
            # Both scores available - use weighted combination
            combined_score = (
                self.semantic_weight * semantic_score +
                self.llm_weight * llm_score
            )
        elif semantic_score is not None:
            # Only semantic score available
            combined_score = semantic_score
        elif llm_score is not None:
            # Only LLM score available
            combined_score = llm_score
        
        result = {
            'custom_answer_relevancy': round(combined_score, 4) if combined_score is not None else None,
            'semantic_similarity': round(semantic_score, 4) if semantic_score is not None else None,
            'llm_relevancy': round(llm_score, 4) if llm_score is not None else None
        }
        
        if self.verbose and combined_score is not None:
            print(f"      ✅ Custom answer relevancy: {combined_score:.4f}")
        
        return result


class CustomContextPrecision:
    """
    Custom Context Precision Metric
    
    Evaluates the rank position of the most relevant context.
    Returns the position (1, 2, 3...) of the first context that contains
    the answer to the question.
    
    Lower is better (1 = best, meaning most relevant is first).
    """
    
    def __init__(self, model_config: dict, verbose: bool = False):
        """
        Initialize Custom Context Precision metric
        
        Args:
            model_config: Configuration for LLM API (REQUIRED)
            verbose: Enable verbose output
        """
        if model_config is None:
            raise ValueError("model_config is required for CustomContextPrecision")
        
        self.verbose = verbose
        self.model_config = model_config
        
        # Initialize OpenAI client
        self.llm_client = OpenAI(
            api_key=self.model_config['api_key'],
            base_url=self.model_config['api_base']
        )
        
        if self.verbose:
            print(f"✅ CustomContextPrecision initialized")
            print(f"   LLM Model: {self.model_config['model']}")
    
    def compute(self, question: str, ground_truth: str, contexts: List[str]) -> Dict[str, Optional[int]]:
        """
        Compute the rank position of the most relevant context using GROUPED BATCH evaluation.
        Groups contexts (3 per group) to balance efficiency and accuracy.
        
        Args:
            question: Question text
            ground_truth: Ground truth answer
            contexts: List of context strings (in retrieval order)
            
        Returns:
            Dictionary containing:
            - context_precision: Rank position (1, 2, 3...) or None if not found
        """
        import re
        
        if self.verbose:
            print(f"      📊 Computing custom context precision (grouped batch)...")
        
        # Handle empty inputs
        if not contexts or len(contexts) == 0:
            return {
                'context_precision': None,
                '_error': 'no_contexts'
            }
        
        # Filter out empty contexts while keeping track of original positions
        valid_contexts_with_pos = [(i+1, c) for i, c in enumerate(contexts) if c and c.strip()]
        
        if not valid_contexts_with_pos:
            return {
                'context_precision': None,
                '_error': 'no_valid_contexts'
            }
        
        # Group contexts: 5 per batch, max 10 groups (50 contexts total)
        GROUP_SIZE = 5
        MAX_GROUPS = 10  # Can handle up to 50 contexts
        
        for group_idx in range(min(MAX_GROUPS, (len(valid_contexts_with_pos) + GROUP_SIZE - 1) // GROUP_SIZE)):
            start = group_idx * GROUP_SIZE
            end = min(start + GROUP_SIZE, len(valid_contexts_with_pos))
            group = valid_contexts_with_pos[start:end]
            
            # Build batch prompt for this group
            contexts_text = ""
            positions_in_group = []
            for pos, ctx in group:
                truncated = ctx[:600] if len(ctx) > 600 else ctx
                contexts_text += f"\n[{pos}] {truncated}\n"
                positions_in_group.append(pos)
            
            prompt = f"""Which context answers this question? Reply with context number or 0.

Question: {question}
Expected: {ground_truth[:200]}

Contexts:{contexts_text}

Reply ONLY with the number of the FIRST relevant context, or 0 if none."""

            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_config['model'],
                    messages=[
                        {"role": "system", "content": "Reply with ONLY a number."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                    timeout=30
                )
                
                raw_content = response.choices[0].message.content
                if raw_content:
                    result = raw_content.strip()
                    try:
                        position = int(result)
                        if position > 0 and position in positions_in_group:
                            if self.verbose:
                                print(f"        ✓ context_precision: {position} (found in group {group_idx+1})")
                            return {'context_precision': position}
                    except ValueError:
                        numbers = re.findall(r'\d+', result)
                        if numbers:
                            position = int(numbers[0])
                            if position > 0 and position in positions_in_group:
                                return {'context_precision': position}
            except Exception as e:
                if self.verbose:
                    print(f"        ⚠️  Group {group_idx+1} evaluation error: {str(e)}")
                continue
        
        # No relevant context found in any group
        if self.verbose:
            print(f"        ⚠️  context_precision: None (no relevant context found)")
        
        return {
            'context_precision': len(contexts) + 1,
            '_note': 'no_relevant_context_found'
        }


class CustomFaithfulness:
    """
    Custom Faithfulness Metric

    Evaluates whether the answer is grounded in and faithful to the provided contexts.
    Uses LLM-based evaluation to check:
    1. Are claims in the answer supported by the contexts?
    2. Does the answer introduce information not present in contexts?
    3. Does the answer contradict the contexts?
    """

    def __init__(self, model_config: dict, verbose: bool = False):
        """
        Initialize Custom Faithfulness metric

        Args:
            model_config: Configuration for LLM API (REQUIRED)
            verbose: Enable verbose output
        """
        if model_config is None:
            raise ValueError("model_config is required for CustomFaithfulness")
        
        self.verbose = verbose
        self.model_config = model_config

        # Initialize OpenAI client
        self.llm_client = OpenAI(
            api_key=self.model_config['api_key'],
            base_url=self.model_config['api_base']
        )

        if self.verbose:
            print(f"✅ CustomFaithfulness initialized")
            print(f"   LLM Model: {self.model_config['model']}")

    def compute(self, answer: str, contexts: List[str], max_retries: int = 3) -> Dict[str, Optional[float]]:
        """
        Compute faithfulness score for answer given contexts with retry mechanism

        Args:
            answer: Generated answer text
            contexts: List of context strings the answer should be grounded in
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary containing:
            - custom_faithfulness: Faithfulness score (0-1)
        """
        import time
        import re
        
        if self.verbose:
            print(f"      📊 Computing custom faithfulness...")

        # Handle empty or invalid inputs
        if not answer or not answer.strip():
            return {
                'custom_faithfulness': 0.0,
                '_error': 'empty_answer'
            }

        if not contexts or len(contexts) == 0:
            return {
                'custom_faithfulness': None,
                '_error': 'no_contexts'
            }

        # Filter out empty contexts
        valid_contexts = [c for c in contexts if c and c.strip()]
        if not valid_contexts:
            return {
                'custom_faithfulness': None,
                '_error': 'no_valid_contexts'
            }

        # Combine contexts with separators - limit to 5 contexts for efficiency
        combined_contexts = "\n---\n".join(valid_contexts[:5])

        # Truncate if too long (reduced from 6000 to 3000 for efficiency)
        max_context_chars = 3000
        if len(combined_contexts) > max_context_chars:
            combined_contexts = combined_contexts[:max_context_chars] + "...[truncated]"

        prompt = f"""You are evaluating whether an answer is faithful to the provided contexts.

Contexts:
{combined_contexts}

Answer: {answer}

Task: Determine if the answer is grounded in and supported by the contexts.

Scoring Guide:
- 0.0-0.3: Contradicts contexts or makes unsupported claims
- 0.4-0.6: Partially grounded, some unsupported information
- 0.7-0.9: Mostly grounded with minor unsupported details
- 1.0: Fully grounded in and faithful to the contexts

Please respond with ONLY a single number between 0.0 and 1.0."""

        last_error = "no_attempts_made"
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_config['model'],
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Score the faithfulness of the answer to the contexts. Respond with ONLY a single decimal number between 0.0 and 1.0, nothing else."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.model_config['temperature'],
                    max_tokens=2000,
                    timeout=60
                )

                # Use helper function to extract score (handles reasoning models)
                score = _extract_score_from_response(response)
                
                if score is not None:
                    if self.verbose:
                        print(f"      ✅ Custom faithfulness: {score:.4f}")
                    return {
                        'custom_faithfulness': round(score, 4)
                    }
                else:
                    last_error = "empty_or_unparseable_response"
                    if self.verbose:
                        raw = response.choices[0].message.content
                        print(f"      ⚠️  Failed to parse LLM response: {str(raw)[:50] if raw else 'empty'}...")
                    continue

            except Exception as e:
                last_error = str(e)
                if self.verbose and attempt == max_retries - 1:
                    print(f"      ⚠️  Faithfulness computation error after {max_retries} attempts: {last_error}")
                
                # Exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    time.sleep(wait_time)

        return {
            'custom_faithfulness': None,
            '_error': f'max_retries_exceeded: {last_error}'
        }


class CustomAnswerCorrectness:
    """
    Custom Answer Correctness Metric

    Evaluates the factual correctness of the answer compared to ground truth.
    Uses LLM-based evaluation to assess:
    1. Factual accuracy of the answer
    2. Completeness compared to ground truth
    3. Precision (no incorrect information)
    """

    def __init__(self, model_config: dict, verbose: bool = False):
        """
        Initialize Custom Answer Correctness metric

        Args:
            model_config: Configuration for LLM API (REQUIRED)
            verbose: Enable verbose output
        """
        if model_config is None:
            raise ValueError("model_config is required for CustomAnswerCorrectness")
        
        self.verbose = verbose
        self.model_config = model_config

        # Initialize OpenAI client
        self.llm_client = OpenAI(
            api_key=self.model_config['api_key'],
            base_url=self.model_config['api_base']
        )

        if self.verbose:
            print(f"✅ CustomAnswerCorrectness initialized")
            print(f"   LLM Model: {self.model_config['model']}")

    def compute(self, question: str, answer: str, ground_truth: str, max_retries: int = 3) -> Dict[str, Optional[float]]:
        """
        Compute correctness score for answer compared to ground truth with retry mechanism

        Args:
            question: The question being answered
            answer: Generated answer text
            ground_truth: Reference/correct answer
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary containing:
            - custom_answer_correctness: Correctness score (0-1)
        """
        import time
        import re
        
        if self.verbose:
            print(f"      📊 Computing custom answer correctness...")

        # Handle empty inputs
        if not answer or not answer.strip():
            return {
                'custom_answer_correctness': 0.0,
                '_error': 'empty_answer'
            }

        if not ground_truth or not ground_truth.strip():
            return {
                'custom_answer_correctness': None,
                '_error': 'no_ground_truth'
            }

        prompt = f"""You are evaluating the correctness of an AI-generated answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Task: Compare the generated answer to the ground truth and rate its correctness.

Scoring Guide:
- 0.0-0.3: Wrong or contradicts ground truth
- 0.4-0.6: Partially correct, missing important info
- 0.7-0.9: Mostly correct, same meaning
- 1.0: Fully correct and complete

Please respond with ONLY a single number between 0.0 and 1.0."""

        last_error = "no_attempts_made"
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_config['model'],
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Your task is to score answer correctness. Respond with ONLY a single decimal number between 0.0 and 1.0, nothing else."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.model_config['temperature'],
                    max_tokens=2000,
                    timeout=60
                )

                # Use helper function to extract score (handles reasoning models)
                score = _extract_score_from_response(response)
                
                if score is not None:
                    if self.verbose:
                        print(f"      ✅ Custom answer correctness: {score:.4f}")
                    return {
                        'custom_answer_correctness': round(score, 4)
                    }
                else:
                    last_error = "empty_or_unparseable_response"
                    if self.verbose:
                        raw = response.choices[0].message.content
                        print(f"      ⚠️  Failed to parse LLM response: {str(raw)[:50] if raw else 'empty'}...")
                    continue

            except Exception as e:
                if self.verbose and attempt == max_retries - 1:
                    print(f"      ⚠️  Correctness computation error after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    time.sleep(wait_time)

        return {
            'custom_answer_correctness': None,
            '_error': f'max_retries_exceeded: {last_error}'
        }


def test_custom_metrics():
    """Test all custom metrics with sample data"""
    print("\n" + "="*80)
    print("Testing All Custom Metrics")
    print("="*80 + "\n")

    # Initialize all metrics
    print("Initializing metrics...")
    relevancy_metric = CustomAnswerRelevancy(verbose=True)
    faithfulness_metric = CustomFaithfulness(verbose=True)
    correctness_metric = CustomAnswerCorrectness(verbose=True)
    print()

    # Test case with full context
    test_case = {
        "question": "What is the efficiency of perovskite solar cells?",
        "answer": "Perovskite solar cells have achieved power conversion efficiencies exceeding 25% in laboratory conditions, making them one of the fastest-advancing photovoltaic technologies.",
        "ground_truth": "Perovskite solar cells have demonstrated power conversion efficiencies of over 25% in research settings, representing rapid progress in solar technology.",
        "contexts": [
            "Recent studies show that perovskite solar cells have reached efficiencies above 25% under laboratory testing conditions.",
            "Perovskite-based photovoltaics are among the fastest-growing solar technologies in terms of efficiency improvements.",
            "The National Renewable Energy Laboratory reported certified efficiency records for perovskite cells exceeding 25.7%."
        ]
    }

    print("="*80)
    print("Test Case: Perovskite Solar Cell Efficiency")
    print("="*80)
    print(f"\nQuestion: {test_case['question']}")
    print(f"\nGround Truth: {test_case['ground_truth']}")
    print(f"\nGenerated Answer: {test_case['answer']}")
    print(f"\nNumber of Contexts: {len(test_case['contexts'])}")

    # Test Answer Relevancy
    print("\n" + "-"*80)
    print("1. Testing Custom Answer Relevancy")
    print("-"*80)
    relevancy_result = relevancy_metric.compute(
        test_case['question'],
        test_case['answer']
    )
    print("\nResults:")
    for key, value in relevancy_result.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")

    # Test Faithfulness
    print("\n" + "-"*80)
    print("2. Testing Custom Faithfulness")
    print("-"*80)
    faithfulness_result = faithfulness_metric.compute(
        test_case['answer'],
        test_case['contexts']
    )
    print("\nResults:")
    for key, value in faithfulness_result.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")

    # Test Answer Correctness
    print("\n" + "-"*80)
    print("3. Testing Custom Answer Correctness")
    print("-"*80)
    correctness_result = correctness_metric.compute(
        test_case['question'],
        test_case['answer'],
        test_case['ground_truth']
    )
    print("\nResults:")
    for key, value in correctness_result.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")

    # Summary
    print("\n" + "="*80)
    print("Summary of All Custom Metrics")
    print("="*80)
    print(f"custom_answer_relevancy:    {relevancy_result.get('custom_answer_relevancy')}")
    print(f"custom_faithfulness:        {faithfulness_result.get('custom_faithfulness')}")
    print(f"custom_answer_correctness:  {correctness_result.get('custom_answer_correctness')}")
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_custom_metrics()
