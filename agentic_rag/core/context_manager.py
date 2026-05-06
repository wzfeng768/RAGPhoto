"""
Context Manager for Agentic RAG v2
Manages and organizes contexts from multiple sources
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict


class ContextManager:
    """
    Context Manager that handles context collection, merging, and organization
    
    Responsibilities:
    1. Collect contexts from multiple tools
    2. Deduplicate and merge contexts
    3. Track context sources
    4. Maintain context history across iterations
    """
    
    def __init__(self, max_contexts: int = 20, verbose: bool = False):
        """
        Initialize Context Manager
        
        Args:
            max_contexts: Maximum number of contexts to maintain
            verbose: Enable verbose logging
        """
        self.max_contexts = max_contexts
        self.verbose = verbose
        
        # Storage
        self.contexts = []  # List of context dicts
        self.context_history = []  # History across iterations
        self.source_stats = defaultdict(int)  # Count by source
    
    def add_contexts(
        self,
        new_contexts: List[Dict[str, Any]],
        source: str = "unknown",
        query: Optional[str] = None,
        prioritize_sections: Optional[List[str]] = None
    ):
        """
        Add new contexts from a source
        
        Args:
            new_contexts: List of context dictionaries
            source: Source name
            query: Optional query for filtering
            prioritize_sections: Optional list of section names to prioritize
        """
        # Validate new_contexts is a list
        if not isinstance(new_contexts, list):
            if self.verbose:
                print(f"   ⚠️  new_contexts is not a list: {type(new_contexts)}")
            return
        
        added_count = 0
        filtered_count = 0
        
        for context in new_contexts:
            # Validate context is a dict
            if not isinstance(context, dict):
                if self.verbose:
                    print(f"   ⚠️  Skipping non-dict context: {type(context)}")
                filtered_count += 1
                continue
            
            # Ensure context has required fields
            if 'content' not in context:
                filtered_count += 1
                continue
            
            # 🆕 Apply relevance filtering
            if not self._is_relevant_enough(context, query, prioritize_sections):
                filtered_count += 1
                continue
            
            # Add source tag
            context['source'] = context.get('source', source)
            
            # Add to collection
            self.contexts.append(context)
            self.source_stats[source] += 1
            added_count += 1
        
        if self.verbose:
            print(f"   Added {added_count} contexts from {source} (filtered {filtered_count})")
    
    def merge_and_deduplicate(self) -> List[Dict[str, Any]]:
        """
        Merge contexts and remove duplicates.
        Web contexts (source.startswith('web')) are reserved slots so they
        are not pushed out by high-scoring vector docs.
        """
        if not self.contexts:
            return []
        
        # Separate web vs non-web
        web_contexts = []
        other_contexts = []
        seen_content = set()
        for context in self.contexts:
            content = context.get('content', '')
            content_key = content[:200].strip() if content else ''
            if not content_key or content_key in seen_content:
                continue
            seen_content.add(content_key)
            src = str(context.get('source', ''))
            if src.startswith('web'):
                web_contexts.append(context)
            else:
                other_contexts.append(context)
        
        # Sort each by relevance
        web_contexts.sort(
            key=lambda x: x.get('rerank_score', x.get('similarity', 0)),
            reverse=True
        )
        other_contexts.sort(
            key=lambda x: x.get('rerank_score', x.get('similarity', 0)),
            reverse=True
        )
        
        # Reserve slots for web (max 10) so they are always included when present
        max_web_slots = min(10, len(web_contexts)) if web_contexts else 0
        other_slots = max(0, self.max_contexts - max_web_slots)
        final_contexts = other_contexts[:other_slots] + web_contexts[:max_web_slots]
        
        if self.verbose:
            print(f"   Merged: {len(self.contexts)} → {len(final_contexts)} final (web: {len(web_contexts[:max_web_slots])})")
        
        return final_contexts
    
    def get_all_contexts(self) -> List[Dict[str, Any]]:
        """Get all current contexts (merged and deduplicated)"""
        return self.merge_and_deduplicate()
    
    def get_contexts_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get contexts from a specific source"""
        return [ctx for ctx in self.contexts if ctx.get('source') == source]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about current contexts"""
        all_contexts = self.get_all_contexts()
        
        return {
            'total_contexts': len(all_contexts),
            'by_source': dict(self.source_stats),
            'unique_sources': len(set(ctx.get('source') for ctx in all_contexts)),
            'has_scores': sum(1 for ctx in all_contexts if 'similarity' in ctx or 'rerank_score' in ctx),
        }
    
    def save_snapshot(self, iteration: int):
        """Save a snapshot of current contexts for this iteration"""
        snapshot = {
            'iteration': iteration,
            'contexts': self.get_all_contexts().copy(),
            'statistics': self.get_statistics()
        }
        self.context_history.append(snapshot)
    
    def get_iteration_history(self) -> List[Dict[str, Any]]:
        """Get context history across all iterations"""
        return self.context_history
    
    def clear(self):
        """Clear all contexts"""
        self.contexts = []
        self.source_stats = defaultdict(int)
    
    def reset(self):
        """Reset for new query (clears contexts and history)"""
        self.clear()
        self.context_history = []
    
    def _is_relevant_enough(
        self, 
        context: Dict[str, Any], 
        query: Optional[str] = None,
        prioritize_sections: Optional[List[str]] = None
    ) -> bool:
        """
        🆕 Filter contexts based on relevance and metadata
        
        Args:
            context: Context dictionary to check
            query: Optional query for semantic filtering
            prioritize_sections: Optional list of priority section types
            
        Returns:
            True if context should be kept, False if filtered out
        """
        # Check similarity threshold
        similarity = context.get('similarity', context.get('rerank_score', 0))
        min_threshold = 0.30  # Minimum threshold for any context
        
        # Priority sections get lower threshold
        metadata = context.get('metadata', {})
        # Ensure metadata is a dict, not a string
        if not isinstance(metadata, dict):
            metadata = {}
        section_type = metadata.get('section_type', '')
        
        if prioritize_sections and section_type in prioritize_sections:
            # Priority sections: keep if above 0.25
            return similarity >= 0.25
        
        # For numerical questions, prioritize chunks with numerical data
        if query:
            query_lower = query.lower()
            has_numerical_query = any(word in query_lower for word in 
                ['plqy', 'efficiency', '%', 'pce', 'value', 'temperature'])
            
            if has_numerical_query and metadata.get('has_numerical_data'):
                # Chunks with numerical data: keep if above 0.25
                return similarity >= 0.25
        
        # Default: apply standard threshold
        return similarity >= min_threshold
    
    def build_context_text(
        self,
        contexts: Optional[List[Dict[str, Any]]] = None,
        include_sources: bool = True,
        max_length: int = 4000
    ) -> str:
        """
        Build formatted context text for LLM
        
        Args:
            contexts: List of contexts (uses all if None)
            include_sources: Include source information
            max_length: Maximum total length in characters
            
        Returns:
            Formatted context string
        """
        if contexts is None:
            contexts = self.get_all_contexts()
        
        if not contexts:
            return "No context available."
        
        context_parts = []
        current_length = 0
        
        for i, ctx in enumerate(contexts, 1):
            content = ctx.get('content', '')
            
            # Build context entry
            if include_sources:
                source = ctx.get('source', 'Unknown')
                similarity = ctx.get('similarity', ctx.get('rerank_score', 0))
                header = f"[Context {i}] (Source: {source}, Score: {similarity:.2f})"
            else:
                header = f"[Context {i}]"
            
            entry = f"{header}\n{content}\n"
            
            # Check length limit
            if current_length + len(entry) > max_length:
                context_parts.append("\n[... Additional contexts truncated due to length ...]")
                break
            
            context_parts.append(entry)
            current_length += len(entry)
        
        return "\n".join(context_parts)
    
    def extract_sources(
        self,
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Extract unique source names from contexts
        
        Args:
            contexts: List of contexts (uses all if None)
            
        Returns:
            List of unique source names
        """
        if contexts is None:
            contexts = self.get_all_contexts()
        
        # Validate contexts is a list
        if not isinstance(contexts, list):
            return ['Unknown']
        
        sources = set()
        for ctx in contexts:
            # Validate ctx is a dict
            if not isinstance(ctx, dict):
                continue
            source = ctx.get('source', 'Unknown')
            sources.add(source)
        
        return sorted(list(sources))
    
    def format_for_citation(
        self,
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Format contexts for citation in answer
        
        Returns:
            Citation string like "[Sources: 5 vector docs + 3 KG entities]"
        """
        if contexts is None:
            contexts = self.get_all_contexts()
        
        # Validate contexts is a list
        if not isinstance(contexts, list):
            return "[No sources]"
        
        if not contexts:
            return "[No sources]"
        
        # Count by source type
        source_counts = defaultdict(int)
        for ctx in contexts:
            # Validate ctx is a dict
            if not isinstance(ctx, dict):
                continue
            source_type = ctx.get('source', 'unknown')
            # Normalize source type
            if 'vector' in source_type.lower() or '.pdf' in source_type or '.md' in source_type:
                source_counts['vector'] += 1
            elif 'kg' in source_type.lower() or 'neo4j' in source_type.lower():
                source_counts['kg'] += 1
            elif 'web' in source_type.lower():
                source_counts['web'] += 1
            else:
                source_counts['other'] += 1
        
        # Build citation string
        parts = []
        if source_counts['vector']:
            parts.append(f"{source_counts['vector']} vector docs")
        if source_counts['kg']:
            parts.append(f"{source_counts['kg']} KG entities")
        if source_counts['web']:
            parts.append(f"{source_counts['web']} web results")
        if source_counts['other']:
            parts.append(f"{source_counts['other']} other sources")
        
        if not parts:
            return f"[Sources: {len(contexts)} contexts]"
        
        return f"[Sources: {' + '.join(parts)}]"

