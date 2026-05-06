"""
Reflection Engine for Agentic RAG v2
Evaluates context quality and recommends next actions
"""

import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Import config using explicit path to avoid conflicts with data_pipeline/config
import sys
from pathlib import Path
_agentic_rag_root = Path(__file__).resolve().parent.parent
if str(_agentic_rag_root) not in sys.path:
    sys.path.insert(0, str(_agentic_rag_root))
try:
    from agentic_rag.config.config import config
except ImportError:
    # Script mode fallback (e.g., `python run_agentic.py` from project root)
    from config.config import config


class ReflectionEngine:
    """
    Reflection Engine that evaluates retrieval quality and recommends actions
    
    Core functions:
    1. Evaluate context quality (relevance, coverage, confidence)
    2. Identify information gaps
    3. Recommend next action (continue iteration or generate answer)
    """
    
    def __init__(self, rag_components, verbose: bool = False):
        """
        Initialize Reflection Engine
        
        Args:
            rag_components: RAGComponents instance for embedding and LLM
            verbose: Enable verbose logging
        """
        self.rag_components = rag_components
        self.verbose = verbose
        
        # Initialize OpenAI client for LLM-based evaluation
        self.llm_client = OpenAI(
            api_key=getattr(config, 'openai_api_key', ''),
            base_url=getattr(config, 'openai_base_url', 'https://api.openai.com/v1')
        )
        
        # Store quality threshold with default
        self.quality_threshold = getattr(config, 'quality_threshold', 0.7)
    
    def evaluate_context_quality(
        self,
        contexts: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of retrieved contexts
        
        Args:
            contexts: List of context dictionaries
            query: Original query
            
        Returns:
            Dictionary with quality scores and analysis
        """
        if not contexts:
            return {
                'relevance_score': 0.0,
                'coverage_score': 0.0,
                'confidence_score': 0.0,
                'overall_score': 0.0,
                'gaps': ['no_contexts_retrieved'],
                'recommendation': 'reformulate_query'
            }
        
        # 1. Compute relevance score
        relevance_score = self._compute_relevance(contexts, query)
        
        # 2. Compute coverage score
        coverage_score = self._compute_coverage(contexts, query)
        
        # 3. Compute confidence score
        confidence_score = self._compute_confidence(contexts)
        
        # 4. Compute overall score (weighted average)
        overall_score = (
            relevance_score * 0.4 +
            coverage_score * 0.4 +
            confidence_score * 0.2
        )
        
        # 5. Identify information gaps
        gaps = self._identify_gaps(contexts, query, overall_score)
        
        # 6. Recommend action
        recommendation = self._recommend_action(overall_score, gaps)
        
        result = {
            'relevance_score': round(relevance_score, 3),
            'coverage_score': round(coverage_score, 3),
            'confidence_score': round(confidence_score, 3),
            'overall_score': round(overall_score, 3),
            'gaps': gaps,
            'recommendation': recommendation,
            'reasoning': self._build_reasoning(
                relevance_score, coverage_score, confidence_score, gaps
            )
        }
        
        if self.verbose:
            print(f"\n🤔 Reflection:")
            print(f"   Relevance: {result['relevance_score']}")
            print(f"   Coverage: {result['coverage_score']}")
            print(f"   Confidence: {result['confidence_score']}")
            print(f"   Overall: {result['overall_score']}")
            print(f"   Gaps: {gaps}")
            print(f"   Recommendation: {recommendation}")
        
        return result
    
    def _compute_relevance(
        self,
        contexts: List[Dict[str, Any]],
        query: str
    ) -> float:
        """
        Compute relevance score using embedding similarity
        
        Method: Average cosine similarity between query and contexts
        """
        try:
            # Generate query embedding
            query_embeddings = self.rag_components.generate_embeddings([query])
            if not query_embeddings:
                return 0.5  # Default if embedding fails
            
            query_embedding = np.array(query_embeddings[0])
            
            # Collect similarities
            similarities = []
            
            for context in contexts[:10]:  # Limit to top 10 for efficiency
                # Use existing similarity if available
                if 'similarity' in context:
                    similarities.append(context['similarity'])
                elif 'rerank_score' in context:
                    similarities.append(context['rerank_score'])
                else:
                    # Compute similarity
                    content = context.get('content', '')
                    if content:
                        context_embeddings = self.rag_components.generate_embeddings([content])
                        if context_embeddings:
                            context_embedding = np.array(context_embeddings[0])
                            sim = np.dot(query_embedding, context_embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
                            )
                            similarities.append(float(sim))
            
            if similarities:
                # Return average similarity
                return float(np.mean(similarities))
            else:
                return 0.5  # Default
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Relevance computation failed: {e}")
            return 0.5  # Default
    
    def _compute_coverage(
        self,
        contexts: List[Dict[str, Any]],
        query: str
    ) -> float:
        """
        Compute coverage score using LLM judgment
        
        Method: Ask LLM if contexts can fully answer the query
        """
        try:
            # Build context text (limit to avoid token limits)
            context_text = "\n\n".join([
                f"[{i+1}] {ctx.get('content', '')[:500]}"
                for i, ctx in enumerate(contexts[:5])
            ])
            
            prompt = f"""Query: {query}

Contexts:
{context_text}

Can these contexts fully answer the query?

Rate coverage on a scale of 0-1:
- 1.0: Contexts contain all information needed to fully answer
- 0.7: Contexts contain most information, minor gaps
- 0.4: Contexts contain some relevant info, major gaps
- 0.0: Contexts cannot answer the query

Return ONLY a number between 0 and 1, no explanation."""
            
            response = self.llm_client.chat.completions.create(
                model=getattr(config, 'llm_model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            
            coverage_str = response.choices[0].message.content.strip()
            
            # Parse number
            try:
                coverage = float(coverage_str)
                return max(0.0, min(1.0, coverage))  # Clamp to [0, 1]
            except ValueError:
                return 0.5  # Default if parsing fails
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Coverage computation failed: {e}")
            return 0.5  # Default
    
    def _compute_confidence(self, contexts: List[Dict[str, Any]]) -> float:
        """
        Compute confidence score based on:
        1. Source diversity
        2. Average similarity scores
        3. Information consistency
        """
        if not contexts:
            return 0.0
        
        factors = []
        
        # Factor 1: Source diversity (more sources = higher confidence)
        sources = set(ctx.get('source', 'unknown') for ctx in contexts)
        source_score = min(len(sources) / 5.0, 1.0)  # 5+ sources = perfect
        factors.append(source_score)
        
        # Factor 2: Average similarity/score (higher = more confident)
        scores = []
        for ctx in contexts:
            if 'rerank_score' in ctx:
                scores.append(ctx['rerank_score'])
            elif 'similarity' in ctx:
                scores.append(ctx['similarity'])
        
        if scores:
            avg_score = np.mean(scores)
            factors.append(float(avg_score))
        
        # Factor 3: Information consistency (simplified: assume consistent if from multiple sources)
        consistency_score = source_score  # Simplified
        factors.append(consistency_score)
        
        # Average all factors
        confidence = float(np.mean(factors)) if factors else 0.5
        
        return confidence
    
    def _identify_gaps(
        self,
        contexts: List[Dict[str, Any]],
        query: str,
        overall_score: float
    ) -> List[str]:
        """
        Identify information gaps in the contexts
        
        Returns:
            List of gap identifiers
        """
        gaps = []
        query_lower = query.lower()
        
        # Gap 1: Too few contexts
        if len(contexts) < 3:
            gaps.append('insufficient_contexts')
        
        # Gap 2: Low overall score
        if overall_score < self.quality_threshold:
            gaps.append('low_quality')
        
        # Gap 3: Check for specific missing elements
        context_text = " ".join([ctx.get('content', '') for ctx in contexts[:5]])
        context_lower = context_text.lower()
        
        # Enhanced: Check for numerical values (if query seems to ask for them)
        numerical_keywords = ['how much', 'how many', 'efficiency', 'percent', '%', 'value',
                              'yield', 'quantum', 'mobility', 'stability', 'temperature',
                              'wavelength', 'nm', 'ev', 'cm²', 'degree', '°c']
        needs_numbers = any(word in query_lower for word in numerical_keywords)
        
        if needs_numbers:
            # Check if context has meaningful numbers (not just single digits)
            import re
            numbers_in_context = re.findall(r'\d+\.?\d*', context_text)
            meaningful_numbers = [n for n in numbers_in_context if float(n) > 1 or '.' in n]
            if len(meaningful_numbers) < 2:  # Need at least 2 numbers for numerical questions
                gaps.append('missing_numerical_values')
        
        # Enhanced: Check for comparison questions
        comparison_keywords = ['compare', 'comparison', 'difference', 'versus', 'vs', 
                               'higher', 'lower', 'better', 'worse', 'than']
        is_comparison = any(word in query_lower for word in comparison_keywords)
        
        if is_comparison:
            # For comparisons, we need data for BOTH entities
            # Try to extract entity names (capitalized words or chemical formulas)
            import re
            entities = re.findall(r'\b[A-Z][a-zA-Z0-9-]*\b', query)
            entities = [e for e in entities if len(e) > 1]  # Filter single letters
            
            if entities:
                # Check if context mentions all entities
                entities_found = sum(1 for e in entities if e.lower() in context_lower)
                if len(entities) >= 2 and entities_found < len(entities):
                    gaps.append('missing_comparison_data')
        
        # Check for entities (if query asks about materials, devices, etc.)
        if any(word in query_lower for word in ['what', 'which', 'materials', 'devices', 'methods']):
            if len(context_text.split()) < 50:
                gaps.append('missing_entities')
        
        # Check for relationships (if query asks how/why)
        if any(word in query_lower for word in ['how', 'why', 'relationship', 'affect', 'improve']):
            relational_words = ['because', 'due to', 'results in', 'leads to', 'causes', 'affects']
            if not any(word in context_lower for word in relational_words):
                gaps.append('missing_relationships')
        
        # If no specific gaps identified but score is low
        if not gaps and overall_score < self.quality_threshold:
            gaps.append('general_insufficient_info')
        
        return gaps
    
    def _recommend_action(
        self,
        overall_score: float,
        gaps: List[str]
    ) -> str:
        """
        Recommend next action based on quality score and gaps
        
        Returns:
            Action string: 'generate', 'reformulate_query', 'enhance_kg', etc.
        """
        # Decision 1: High quality → generate answer
        if overall_score >= self.quality_threshold:
            return 'generate'
        
        # Decision 2: Based on specific gaps
        if 'insufficient_contexts' in gaps or 'low_quality' in gaps:
            return 'reformulate_query'
        
        if 'missing_entities' in gaps or 'missing_relationships' in gaps:
            return 'enhance_kg_query'
        
        if 'missing_numerical_values' in gaps:
            return 'expand_retrieval'  # Changed: expand retrieval instead of focus
        
        if 'missing_comparison_data' in gaps:
            return 'expand_retrieval'  # Need more data for comparison
        
        # Default: reformulate query
        return 'reformulate_query'
    
    def _build_reasoning(
        self,
        relevance: float,
        coverage: float,
        confidence: float,
        gaps: List[str]
    ) -> str:
        """Build human-readable reasoning for the evaluation"""
        parts = []
        
        # Analyze each dimension
        if relevance >= 0.8:
            parts.append("High relevance")
        elif relevance >= 0.6:
            parts.append("Moderate relevance")
        else:
            parts.append("Low relevance")
        
        if coverage >= 0.8:
            parts.append("good coverage")
        elif coverage >= 0.6:
            parts.append("partial coverage")
        else:
            parts.append("poor coverage")
        
        if confidence >= 0.8:
            parts.append("high confidence")
        elif confidence >= 0.6:
            parts.append("moderate confidence")
        else:
            parts.append("low confidence")
        
        # Add gap information
        if gaps:
            gap_str = ", ".join(gaps)
            parts.append(f"Gaps: {gap_str}")
        
        return ". ".join(parts) + "."
    
    def explain_decision(
        self,
        reflection_result: Dict[str, Any],
        iteration: int,
        max_iterations: int
    ) -> str:
        """
        Generate explanation for the decision
        
        Args:
            reflection_result: Result from evaluate_context_quality
            iteration: Current iteration number
            max_iterations: Maximum iterations allowed
            
        Returns:
            Human-readable explanation
        """
        overall_score = reflection_result['overall_score']
        recommendation = reflection_result['recommendation']
        gaps = reflection_result['gaps']
        
        if overall_score >= self.quality_threshold:
            return f"✅ Quality sufficient ({overall_score:.2f} >= {self.quality_threshold}). Proceeding to generate answer."
        
        if iteration >= max_iterations:
            return f"⚠️  Max iterations reached ({iteration}/{max_iterations}). Generating answer with best available contexts."
        
        gap_text = ", ".join(gaps) if gaps else "general quality issues"
        return f"🔄 Quality insufficient ({overall_score:.2f} < {self.quality_threshold}). Gaps: {gap_text}. Continuing iteration..."

