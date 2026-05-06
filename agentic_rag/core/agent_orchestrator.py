"""
Agent Orchestrator for Agentic RAG v2
The brain and decision center of the agentic system
Enhanced with:
- Query normalization and abbreviation expansion
- Knowledge graph integration in answer generation
"""

import time
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Import config using relative path to avoid conflicts with data_pipeline/config
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
from .rag_components import RAGComponents
from .reflection_engine import ReflectionEngine
from .context_manager import ContextManager
from .entity_normalizer import get_normalizer
from .answer_validator import AnswerValidator


class AgentOrchestrator:
    """
    Agent Orchestrator - The Decision Center
    
    Implements the core agentic workflow:
    1. Planning: Analyze query and formulate strategy
    2. Tool Call: Execute tools based on plan
    3. Observation: Collect and organize results
    4. Reflection: Evaluate quality and identify gaps
    5. Decision: Continue iteration or generate answer
    """
    
    def __init__(
        self,
        max_iterations: int = None,
        quality_threshold: float = None,
        enable_web_search: bool = None,
        enable_kg: bool = None,
        verbose: bool = False
    ):
        """
        Initialize Agent Orchestrator
        
        Args:
            max_iterations: Maximum number of iterations (default from config)
            quality_threshold: Quality threshold for stopping (default from config)
            enable_web_search: Enable web search tool (default from config)
            enable_kg: Enable knowledge graph enhancement (default True)
            verbose: Enable verbose logging
        """
        self.max_iterations = max_iterations or config.max_iterations
        self.quality_threshold = quality_threshold or config.quality_threshold
        self.enable_web_search = enable_web_search if enable_web_search is not None else getattr(config, 'enable_web_search', False)
        self.verbose = verbose
        
        # Runtime configurable parameters
        self.top_k = 15  # Default top-k for vector search
        self.enable_kg = enable_kg if enable_kg is not None else True  # Enable KG enhancement by default
        
        # Initialize components
        if self.verbose:
            print("🔧 Initializing Agentic RAG System...")
        
        self.rag_components = RAGComponents(verbose=verbose)
        self.reflection_engine = ReflectionEngine(self.rag_components, verbose=verbose)
        self.context_manager = ContextManager(verbose=verbose)
        
        # Initialize query normalizer for abbreviation handling
        self.query_normalizer = get_normalizer()
        
        # Initialize LLM client
        self.llm_client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        # Tool registry
        self.tools = {}
        self._register_tools()
        
        if self.verbose:
            print("✅ Agent Orchestrator initialized")
            print(f"   Max iterations: {self.max_iterations}")
            print(f"   Quality threshold: {self.quality_threshold}")
            print(f"   Web search: {'Enabled' if self.enable_web_search else 'Disabled'}")
            print(f"   Knowledge Graph: {'Enabled' if self.enable_kg else 'Disabled'}")

    def _register_tools(self):
        """Register optional tools based on runtime configuration."""
        # Always clear stale tool registration first (supports runtime toggle).
        self.tools.pop('web', None)

        if not self.enable_web_search:
            return

        try:
            try:
                from agentic_rag.tools.web_search import WebSearchTool
            except ImportError:
                from tools.web_search import WebSearchTool

            raw_sources = getattr(config, 'web_enabled_sources', 'arxiv,pubmed,semantic_scholar,core')
            if isinstance(raw_sources, str):
                enabled_sources = [s.strip() for s in raw_sources.split(',') if s.strip()]
            elif isinstance(raw_sources, list):
                enabled_sources = [str(s).strip() for s in raw_sources if str(s).strip()]
            else:
                enabled_sources = ['arxiv', 'pubmed', 'semantic_scholar', 'core']

            self.tools['web'] = WebSearchTool(
                max_results=getattr(config, 'web_search_results', 5),
                mode=getattr(config, 'web_search_mode', 'both'),
                enabled_sources=enabled_sources,
                semantic_scholar_key=getattr(config, 'semantic_scholar_api_key', ''),
                core_key=getattr(config, 'core_api_key', ''),
                pubmed_email=getattr(config, 'pubmed_email', 'user@example.com'),
                timeout=getattr(config, 'web_search_timeout', 10),
                engine=getattr(config, 'web_search_engine', 'duckduckgo'),
            )
            if self.verbose:
                print("  ✅ WebSearchTool registered")
        except Exception as e:
            # Keep system running even if optional tool fails to initialize.
            if self.verbose:
                print(f"  ⚠️ WebSearchTool registration failed: {e}")

    def configure_web_search(
        self,
        enabled: bool,
        mode: Optional[str] = None,
        max_results: Optional[int] = None
    ):
        """
        Configure web-search behavior at runtime.

        This is used by Streamlit UI so users can toggle web search
        without restarting the app.
        """
        self.enable_web_search = bool(enabled)
        if mode is not None:
            config.web_search_mode = mode
        if max_results is not None:
            config.web_search_results = int(max_results)
        self._register_tools()
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Execute agentic RAG query with iterative refinement
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary with answer, iteration history, and metadata
        """
        # Validate query
        if not user_query or not user_query.strip():
            return {
                'success': False,
                'error': 'Empty query provided',
                'final_answer': 'Please provide a valid question.',
                'iterations': 0,
                'quality_score': 0.0,
                'iteration_history': [],
                'total_time': 0.0
            }
        
        user_query = user_query.strip()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🎯 New Query: {user_query}")
            print(f"{'='*80}")
        
        start_time = time.time()
        
        # Reset context manager
        self.context_manager.reset()
        
        # Iteration state
        current_iteration = 0
        iteration_history = []
        current_query = user_query
        
        # Main iteration loop
        while current_iteration < self.max_iterations:
            current_iteration += 1
            
            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"🔄 Iteration {current_iteration}/{self.max_iterations}")
                print(f"{'─'*80}")
            
            iteration_start = time.time()
            
            # Step 1: Planning
            plan = self._plan(current_query, current_iteration)
            
            # Step 2: Tool Call
            tool_results = self._execute_tools(plan)
            
            # Step 3: Observation
            observation = self._observe(tool_results)
            
            # Step 4: Reflection
            all_contexts = self.context_manager.get_all_contexts()
            reflection = self.reflection_engine.evaluate_context_quality(
                all_contexts,
                user_query  # Use original query for evaluation
            )
            
            iteration_time = time.time() - iteration_start
            
            # Save iteration record
            iteration_record = {
                'iteration': current_iteration,
                'query': current_query,
                'plan': plan,
                'tool_results': {
                    'vector_docs': observation.get('vector_docs_count', 0),
                    'kg_entities': observation.get('kg_entities_count', 0),
                    'web_results': observation.get('web_results_count', 0)
                },
                'contexts_count': observation.get('total_contexts', 0),
                'quality_score': reflection['overall_score'],
                'reflection': reflection,
                'time': iteration_time
            }
            iteration_history.append(iteration_record)
            
            # Save context snapshot
            self.context_manager.save_snapshot(current_iteration)
            
            # Step 5: Decision Making
            decision = self._should_continue(reflection, current_iteration)
            
            if self.verbose:
                explanation = self.reflection_engine.explain_decision(
                    reflection, current_iteration, self.max_iterations
                )
                print(f"\n{explanation}")
            
            if not decision['continue']:
                # Stop iteration
                break
            else:
                # Continue: Need to improve
                action = decision['action']
                
                if action == 'reformulate_query':
                    current_query = self._reformulate_query(
                        user_query,
                        all_contexts,
                        reflection['gaps']
                    )
                    if self.verbose:
                        print(f"\n✏️  Reformulated Query: {current_query}")
                
                elif action == 'expand_retrieval':
                    # Increase retrieval parameters for next iteration
                    self.top_k = min(self.top_k + 10, 30)  # Increase top_k, cap at 30
                    if self.verbose:
                        print(f"\n🔍 Expanding retrieval: top_k increased to {self.top_k}")
                    # Also reformulate to target missing data
                    current_query = self._reformulate_query(
                        user_query,
                        all_contexts,
                        reflection['gaps']
                    )
                    if self.verbose:
                        print(f"✏️  Reformulated Query: {current_query}")
                
                elif action == 'enhance_kg_query':
                    # Enable/enhance KG query on next iteration
                    if self.verbose:
                        print(f"\n🔗 Enhancing KG query for next iteration")
                    # Reformulate query to focus on entities and relationships
                    current_query = self._reformulate_query(
                        user_query,
                        all_contexts,
                        reflection['gaps']
                    )
                    if self.verbose:
                        print(f"✏️  Reformulated Query: {current_query}")
        
        # Generate final answer
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🤖 Generating Final Answer...")
            print(f"{'='*80}")
        
        final_contexts = self.context_manager.get_all_contexts()
        answer = self._generate_answer(user_query, final_contexts)
        
        total_time = time.time() - start_time
        
        # Web search status for UI
        web_triggered = any(
            'web' in (h.get('plan', {}).get('required_tools', []))
            for h in iteration_history
        )
        web_total = sum(h.get('tool_results', {}).get('web_results', 0) for h in iteration_history)
        web_in_final = sum(1 for c in final_contexts if str(c.get('source', '')).startswith('web'))
        
        # Build result
        result = {
            'final_answer': answer,
            'query': user_query,
            'iterations': current_iteration,
            'iteration_history': iteration_history,
            'contexts': final_contexts,
            'context_count': len(final_contexts),
            'sources': self.context_manager.extract_sources(),
            'citation': self.context_manager.format_for_citation(),
            'quality_score': iteration_history[-1]['quality_score'] if iteration_history else 0.0,
            'total_time': round(total_time, 2),
            'config': {
                'max_iterations': self.max_iterations,
                'quality_threshold': self.quality_threshold
            },
            'web_search': {
                'triggered': web_triggered,
                'total_fetched': web_total,
                'in_final': web_in_final,
            },
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✅ Query completed in {total_time:.2f}s")
            print(f"   Iterations: {current_iteration}")
            print(f"   Quality: {result['quality_score']:.3f}")
            print(f"   Contexts: {result['context_count']}")
            print(f"{'='*80}\n")
        
        return result
    
    def _plan(self, query: str, iteration: int) -> Dict[str, Any]:
        """
        Plan the retrieval strategy
        Enhanced with abbreviation expansion and query normalization
        
        Args:
            query: Current query
            iteration: Current iteration number
            
        Returns:
            Plan dictionary with strategy and parameters
        """
        if self.verbose:
            print(f"\n📋 Planning...")
        
        # 🆕 Enhance query with abbreviation expansion
        query_info = self.query_normalizer.enhance_query_for_retrieval(query)
        enhanced_query = query_info['enhanced']
        key_terms = query_info.get('key_terms', [])
        
        if self.verbose and enhanced_query != query:
            print(f"   ✓ Query enhanced with abbreviations")
            if key_terms:
                print(f"   Key terms: {', '.join(key_terms[:5])}")
        
        # Classify query type
        query_type = self._classify_query(query)
        
        # 🆕 Analyze information needs for dynamic parameters
        info_needs = self._analyze_information_needs(query)
        
        # Select tools based on query type
        required_tools = self._select_tools(query_type, iteration, enhanced_query)
        
        # 🆕 Set parameters dynamically based on information needs
        parameters = self._get_dynamic_parameters(info_needs, query_type)
        
        plan = {
            'query': enhanced_query,  # 使用增强后的查询
            'original_query': query,  # 保留原始查询
            'key_terms': key_terms,  # 添加关键术语
            'query_type': query_type,
            'info_needs': info_needs,  # 🆕 信息需求分析
            'required_tools': required_tools,
            'parameters': parameters,
            'iteration': iteration
        }
        
        if self.verbose:
            print(f"   Query: {query[:60]}...")
            print(f"   Type: {query_type}")
            print(f"   Tools: {required_tools}")
        
        return plan
    
    def _classify_query(self, query: str) -> str:
        """Classify query type based on keywords"""
        query_lower = query.lower()
        
        if re.search(r"\b(what is|define|explain|meaning of)\b", query_lower):
            return 'definition'
        elif re.search(r"\b(how|why|relationship|affect)\b", query_lower):
            return 'relationship'
        elif re.search(r"\b(latest|recent|current|2024|2025|2026)\b", query_lower):
            return 'latest'
        elif re.search(r"\b(compare|difference|versus|vs)\b", query_lower):
            return 'comparison'
        elif re.search(r"\b(when|where|who)\b", query_lower):
            return 'factual'
        else:
            return 'general'
    
    def _select_tools(self, query_type: str, iteration: int, query_text: str = "") -> List[str]:
        """Select tools based on query type and iteration"""
        # Base tools (always used)
        tools = ['vector']
        
        # 🆕 Add KG as a base tool (always used if enabled)
        if self.enable_kg:
            tools.append('kg')
        
        # Add reranker if enabled
        if getattr(config, 'enable_reranking', True):
            tools.append('reranker')
        
        # Add web search when enabled. Trigger for: latest, definition, general.
        # "Define PCE" etc. are definition queries; users expect web results for these.
        latest_hint = query_type == 'latest'
        if not latest_hint and query_text:
            latest_hint = re.search(r"\b(latest|recent|current|2024|2025|2026)\b", query_text.lower()) is not None

        web_trigger_types = {'latest', 'definition', 'general'}
        web_should_run = (query_type in web_trigger_types or latest_hint)
        web_start_iter = max(1, int(getattr(config, 'web_search_start_iteration', 1)))
        if web_should_run and iteration >= web_start_iter and self.enable_web_search:
            tools.append('web')
        
        return tools
    
    def _analyze_information_needs(self, query: str) -> Dict[str, bool]:
        """
        🆕 Analyze what type of information the query needs
        
        This helps adjust retrieval parameters dynamically based on query type
        """
        query_lower = query.lower()
        
        return {
            'numerical_question': any(word in query_lower for word in 
                ['plqy', 'efficiency', 'pce', '%', 'how much', 'value', 'temperature', 
                 'mobility', 'conductivity', 'voc', 'jsc', 'ff', 'bandgap']),
            'method_question': any(word in query_lower for word in 
                ['how', 'fabricated', 'prepared', 'method', 'process', 'procedure',
                 'synthesis', 'deposit', 'coat', 'anneal']),
            'comparison_question': any(word in query_lower for word in 
                ['compare', 'difference', 'versus', 'vs', 'better', 'worse',
                 'higher', 'lower']),
            'definition_question': any(word in query_lower for word in 
                ['what is', 'define', 'definition', 'meaning', 'explain']),
        }
    
    def _get_dynamic_parameters(self, info_needs: Dict[str, bool], query_type: str) -> Dict[str, Any]:
        """
        🆕 Get retrieval parameters dynamically based on information needs
        
        Adjusts precision/recall tradeoff based on question type
        """
        # Start with default parameters (use getattr for safety)
        default_vector_top_k = getattr(config, 'vector_top_k', 20)
        default_enable_reranking = getattr(config, 'enable_reranking', True)
        default_reranker_top_n = getattr(config, 'reranker_top_n', 8)
        default_kg_max_entities = getattr(config, 'kg_max_entities', 10)
        
        parameters = {
            'vector_top_k': default_vector_top_k,
            'enable_reranking': default_enable_reranking,
            'reranker_top_n': default_reranker_top_n,
            'kg_max_entities': default_kg_max_entities,
            'prioritize_sections': []
        }
        
        # Adjust for numerical questions - INCREASE recall to find specific values
        # Numerical values may be in different sections (results, tables, methods)
        if info_needs.get('numerical_question'):
            parameters['vector_top_k'] = max(20, default_vector_top_k)  # Increase for better recall
            parameters['reranker_top_n'] = max(8, default_reranker_top_n)  # Keep more candidates
            parameters['prioritize_sections'] = ['results', 'performance', 'experimental', 'characterization']
            
        # Adjust for method questions
        elif info_needs.get('method_question'):
            parameters['vector_top_k'] = 15
            parameters['reranker_top_n'] = 5
            parameters['prioritize_sections'] = ['methodology', 'experimental', 'fabrication']
            
        # Adjust for comparison questions - need more context to compare entities
        elif info_needs.get('comparison_question'):
            parameters['vector_top_k'] = max(20, default_vector_top_k)  # Need more context for comparisons
            parameters['reranker_top_n'] = max(8, default_reranker_top_n)  # Keep more candidates
            parameters['prioritize_sections'] = ['results', 'discussion', 'performance']
            
        return parameters
    
    def _execute_tools(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tools based on plan
        
        Args:
            plan: Plan dictionary
            
        Returns:
            Tool results dictionary
        """
        if self.verbose:
            print(f"\n⚙️  Executing Tools...")
        
        results = {}
        query = plan.get('query', '')
        required_tools = plan.get('required_tools', [])
        params = plan.get('parameters', {})
        
        # Execute Vector Retrieval
        if 'vector' in required_tools:
            vector_results = self.rag_components.search_vectors(
                query=query,
                top_k=params.get('vector_top_k', self.top_k)  # Use runtime top_k
            )
            results['vector'] = vector_results
            if self.verbose:
                print(f"   Vector: {len(vector_results)} documents (top_k={self.top_k})")
        
        # Execute KG Query (only if enabled)
        if 'kg' in required_tools and self.enable_kg:
            kg_results = self.rag_components.query_knowledge_graph(
                query=query,
                max_entities=params.get('kg_max_entities', 10)
            )
            results['kg'] = kg_results
            if self.verbose:
                print(f"   KG: {kg_results.get('count', 0)} entities")
        elif 'kg' in required_tools and not self.enable_kg:
            if self.verbose:
                print(f"   KG: Disabled by user")
        
        # Execute Reranking
        if 'reranker' in required_tools and 'vector' in results:
            reranked = self.rag_components.rerank_documents(
                query=query,
                documents=results['vector'],
                top_n=params.get('reranker_top_n', 5)
            )
            results['vector'] = reranked
            # Note: rerank_documents now prints its own status log
        
        # Web search
        if 'web' in required_tools:
            web_tool = self.tools.get('web')
            if web_tool is not None:
                try:
                    web_output = web_tool.execute(
                        query=query,
                        max_results=getattr(config, 'web_search_results', 5)
                    )
                    web_contexts = web_output.get('contexts', []) if isinstance(web_output, dict) else []
                    results['web'] = web_contexts if isinstance(web_contexts, list) else []
                    if self.verbose:
                        fallback_flag = web_output.get('fallback_used', False) if isinstance(web_output, dict) else False
                        fallback_msg = " (with fallback)" if fallback_flag else ""
                        print(f"   Web: {len(results['web'])} contexts{fallback_msg}")
                except Exception as e:
                    results['web'] = []
                    if self.verbose:
                        print(f"   ⚠️ Web search failed: {e}")
            else:
                results['web'] = []
                if self.verbose:
                    print("   Web: Tool unavailable")
        
        return results
    
    def _observe(self, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe and organize tool results
        
        Args:
            tool_results: Results from tool execution
            
        Returns:
            Observation summary
        """
        if self.verbose:
            print(f"\n👁️  Observing Results...")
        
        # Add vector contexts
        if 'vector' in tool_results:
            self.context_manager.add_contexts(tool_results['vector'], source='vector')
        
        # Add KG contexts
        if 'kg' in tool_results:
            kg_data = tool_results['kg']
            # Convert KG data to context format
            kg_contexts = []
            if kg_data.get('summary'):
                kg_contexts.append({
                    'content': kg_data['summary'],
                    'source': 'kg',
                    'metadata': {
                        'entities': kg_data.get('entities', []),
                        'relationships': kg_data.get('relationships', [])
                    }
                })
            self.context_manager.add_contexts(kg_contexts, source='kg')
        
        # Add web contexts
        if 'web' in tool_results:
            self.context_manager.add_contexts(tool_results['web'], source='web')
        
        # Get statistics
        stats = self.context_manager.get_statistics()
        
        observation = {
            'total_contexts': stats['total_contexts'],
            'vector_docs_count': len(tool_results.get('vector', [])),
            'kg_entities_count': tool_results.get('kg', {}).get('count', 0),
            'web_results_count': len(tool_results.get('web', [])),
            'by_source': stats['by_source']
        }
        
        if self.verbose:
            print(f"   Total contexts: {observation['total_contexts']}")
        
        return observation
    
    def _should_continue(
        self,
        reflection: Dict[str, Any],
        iteration: int
    ) -> Dict[str, bool]:
        """
        Decide whether to continue iteration
        
        Args:
            reflection: Reflection result
            iteration: Current iteration number
            
        Returns:
            Decision dictionary
        """
        overall_score = reflection['overall_score']
        
        # Early stopping if quality is very high
        if getattr(config, 'enable_early_stopping', True) and overall_score >= getattr(config, 'early_stop_threshold', 0.9):
            return {'continue': False, 'reason': 'early_stop', 'action': None}
        
        # Stop if quality threshold met
        if overall_score >= self.quality_threshold:
            return {'continue': False, 'reason': 'quality_sufficient', 'action': None}
        
        # Stop if max iterations reached
        if iteration >= self.max_iterations:
            return {'continue': False, 'reason': 'max_iterations', 'action': None}
        
        # Continue: need to improve
        action = reflection.get('recommendation', 'reformulate_query')
        return {'continue': True, 'reason': 'quality_insufficient', 'action': action}
    
    def _reformulate_query(
        self,
        original_query: str,
        contexts: List[Dict[str, Any]],
        gaps: List[str]
    ) -> str:
        """
        Reformulate query to address information gaps
        
        Args:
            original_query: Original user query
            contexts: Current contexts
            gaps: Identified information gaps
            
        Returns:
            Reformulated query
        """
        gap_descriptions = {
            'missing_entities': 'specific entities or materials',
            'missing_relationships': 'relationships and mechanisms',
            'missing_numerical_values': 'numerical values and measurements',
            'low_quality': 'more relevant and detailed information',
            'insufficient_contexts': 'more comprehensive information'
        }
        
        gap_text = ", ".join([gap_descriptions.get(g, g) for g in gaps])
        
        # 🆕 Extract key terms and expand abbreviations
        query_info = self.query_normalizer.enhance_query_for_retrieval(original_query)
        key_terms = query_info.get('key_terms', [])
        key_terms_text = ", ".join(key_terms[:5]) if key_terms else "N/A"
        
        prompt = f"""Original Query: {original_query}

Key Terms (with abbreviations expanded): {key_terms_text}

The retrieved contexts are missing: {gap_text}

Rewrite the query to be more specific and likely to retrieve the missing information.

Guidelines:
1. Keep the original intent
2. Add specific terms related to the gaps
3. Use full names instead of abbreviations (e.g., "Power Conversion Efficiency" instead of "PCE")
4. Add domain context (photovoltaics, solar cells, materials, etc.)
5. Be more explicit about what information is needed
6. Include relevant technical terms from the key terms list

Return ONLY the reformulated query, no explanation."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=getattr(config, 'llm_model', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            if content is None:
                if self.verbose:
                    print(f"⚠️  Query reformulation returned None")
                return original_query  # Fallback to original
            reformulated = content.strip()
            
            # Remove quotes if present
            if reformulated.startswith('"') and reformulated.endswith('"'):
                reformulated = reformulated[1:-1]
            
            return reformulated
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Query reformulation failed: {e}")
            return original_query  # Fallback to original
    
    def _generate_answer(
        self,
        query: str,
        contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Generate final answer using LLM
        🆕 Enhanced with knowledge graph integration
        
        Args:
            query: Original query
            contexts: Retrieved contexts
            
        Returns:
            Generated answer with integrated KG knowledge
        """
        # Build context text from retrieved documents
        context_text = self.context_manager.build_context_text(
            contexts=contexts,
            include_sources=True,
            max_length=3500  # 减少一些，为KG知识留出空间
        )
        
        # 🆕 Query knowledge graph for relevant structured knowledge
        kg_knowledge = self._get_kg_knowledge(query)
        
        # Build enhanced prompt with KG knowledge
        system_message = """You are a precision-focused scientific assistant specializing in photovoltaic and optoelectronic research.

GOLDEN RULES:
- Exactness > Fluency: Prefer precise technical terms over natural language paraphrasing
- Numbers are sacred: Never round, approximate, or infer numerical values
- Method names matter: Use exact process names from literature (e.g., "vapor-diffusion" ≠ "evaporation")  
- When in doubt, quote: Use direct quotes for critical information

Your users are researchers who need precise, citable information."""
        
        prompt_parts = [
            f"User Question: {query}\n"
        ]
        
        # Add retrieved contexts
        prompt_parts.append("Retrieved Contexts:")
        prompt_parts.append(context_text)
        
        # 🆕 Add knowledge graph information if available
        if kg_knowledge:
            prompt_parts.append("\n\nKnowledge Graph Information:")
            prompt_parts.append(kg_knowledge)
            prompt_parts.append("\n(Use this structured knowledge to enrich your answer with relationships and connections)")
        
        prompt_parts.append("""
Instructions:
1. **CRITICAL: Extract and use EXACT values, numbers, and terminology from the contexts**
   - If contexts mention "98% and 99%", use these exact numbers, NOT approximations like "close to 100%"
   - If contexts use specific technical terms (e.g., "vapor-diffusion"), use them exactly, NOT synonyms
   
2. **For numerical questions**:
   - Always cite the precise values with units
   - Include ranges, comparisons, and specific measurements
   - Quote the exact sentence containing the number if available
   
3. **For method/process questions**:
   - Use the exact terminology from the source
   - Include specific parameters and conditions mentioned
   - Cite step-by-step procedures if described
   
4. **Quote key phrases** directly when they contain critical information

5. **Structure**: [Direct answer with exact values] → [Supporting details] → [Context/implications]

6. **If a value is not found**: State "The exact value is not specified in the retrieved documents" rather than approximating

7. Use knowledge graph data to explain relationships and connections between entities

Answer:""")
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=getattr(config, 'llm_model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=getattr(config, 'llm_temperature', 0.3),
                max_tokens=getattr(config, 'llm_max_tokens', 2000)
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned None content")
            answer = content.strip()
            
            # 🆕 Validate and potentially correct the answer
            if config.enable_answer_validation if hasattr(config, 'enable_answer_validation') else True:
                validator = AnswerValidator(self.llm_client, verbose=self.verbose)
                validation_result = validator.validate_and_correct(
                    question=query,
                    answer=answer,
                    contexts=contexts
                )
                
                if validation_result['needs_correction']:
                    if self.verbose:
                        print(f"\n⚠️  Answer validation issues detected:")
                        for issue in validation_result['issues']:
                            print(f"   - {issue}")
                        print(f"🔧 Attempting auto-correction...")
                    
                    # Use corrected answer if available and better
                    if 'corrected_answer' in validation_result and validation_result['corrected_answer']:
                        answer = validation_result['corrected_answer']
                        if self.verbose:
                            print(f"✅ Answer corrected with precise values")
            
            # Add citation
            citation = self.context_manager.format_for_citation(contexts)
            answer_with_citation = f"{answer}\n\n{citation}"
            
            return answer_with_citation
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _get_kg_knowledge(self, query: str) -> str:
        """
        🆕 Extract relevant knowledge from knowledge graph
        
        Args:
            query: User query
            
        Returns:
            Formatted KG knowledge string
        """
        try:
            # Query knowledge graph
            kg_results = self.rag_components.query_knowledge_graph(
                query=query,
                max_entities=10
            )
            
            if not kg_results or kg_results.get('count', 0) == 0:
                return ""
            
            # Format KG results
            kg_text_parts = []
            entities = kg_results.get('entities', [])
            relationships = kg_results.get('relationships', [])
            
            # Format entities
            if entities:
                kg_text_parts.append("Relevant Entities:")
                for entity in entities[:8]:  # Limit to 8 entities
                    entity_name = entity.get('name', 'Unknown')
                    entity_type = entity.get('type', 'Unknown')
                    entity_desc = entity.get('description', '')
                    
                    entity_info = f"- {entity_name} ({entity_type})"
                    if entity_desc:
                        entity_info += f": {entity_desc[:100]}"
                    kg_text_parts.append(entity_info)
            
            # Format relationships
            if relationships:
                kg_text_parts.append("\nKey Relationships:")
                for rel in relationships[:10]:  # Limit to 10 relationships
                    source = rel.get('source', 'Unknown')
                    target = rel.get('target', 'Unknown')
                    rel_type = rel.get('type', 'RELATES')
                    
                    rel_info = f"- {source} → {rel_type} → {target}"
                    kg_text_parts.append(rel_info)
            
            return "\n".join(kg_text_parts)
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  KG knowledge extraction failed: {e}")
            return ""
    
    def close(self):
        """Close all connections"""
        if hasattr(self, 'rag_components'):
            self.rag_components.close()

