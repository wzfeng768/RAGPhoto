"""
RAG Components Integration
Integrates core components from RAGPhoto_5 for use in Agentic RAG v2
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Get project root and add to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import config


class RAGComponents:
    """
    Integration wrapper for RAGPhoto_5 components
    
    This class provides a unified interface to:
    - Vector Store (Milvus)
    - Knowledge Graph (Neo4j)
    - Embedding Generator
    - LLM Generator
    - Reranker Model
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize RAG components from RAGPhoto_5
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self._initialized = False
        
        if self.verbose:
            print("🔧 Initializing RAG Components from RAGPhoto_5...")
        
        try:
            # Strategy: Use absolute imports with fully qualified names
            ragphoto5_path = '/home/wzfeng/RAGPhoto/Data_Agentic_RAG/data_pipeline'
            
            # Save original state
            original_path = sys.path.copy()
            
            # Save Agentic_RAG_v2's core modules before clearing
            agentic_core_modules = {}
            for k in list(sys.modules.keys()):
                if k.startswith('core.') or k == 'core':
                    # Only save if it's from Agentic_RAG_v2
                    mod = sys.modules[k]
                    if hasattr(mod, '__file__') and mod.__file__ and 'Agentic_RAG_v2' in mod.__file__:
                        agentic_core_modules[k] = mod
                    # Clear all core modules temporarily
                    del sys.modules[k]
            
            # Set up clean path with only RAGPhoto_5
            sys.path.clear()
            sys.path.append(ragphoto5_path)
            sys.path.extend([p for p in original_path if p != ragphoto5_path])
            
            # Now import RAGPhoto_5 components
            from core.vector_store import MilvusVectorStore
            from core.embeddings import EmbeddingGenerator
            from core.llm import LLMGenerator
            from core.neo4j_manager import Neo4jManager
            
            # Create instances immediately (while imports are still valid)
            self.vector_store = MilvusVectorStore()
            self.embedding_generator = EmbeddingGenerator()
            self.llm_generator = LLMGenerator()
            self.neo4j_manager = Neo4jManager()
            
            # Store the classes for later use
            self._MilvusVectorStore = MilvusVectorStore
            self._EmbeddingGenerator = EmbeddingGenerator
            self._LLMGenerator = LLMGenerator
            self._Neo4jManager = Neo4jManager
            
            # Restore original sys.path
            sys.path = original_path
            
            # Restore Agentic_RAG_v2's core modules
            for k, mod in agentic_core_modules.items():
                sys.modules[k] = mod
            
            if self.verbose:
                print("  ✅ Vector Store (Milvus) initialized")
                print("  ✅ Embedding Generator initialized")
                print("  ✅ LLM Generator initialized")
                print("  ✅ Neo4j Manager initialized")
            
            # Initialize Reranker if enabled
            if config.enable_reranking:
                try:
                    # Save Agentic_RAG_v2 core modules
                    agentic_core_modules = {}
                    for k in list(sys.modules.keys()):
                        if k.startswith('core.reranker'):
                            mod = sys.modules[k]
                            if hasattr(mod, '__file__') and mod.__file__ and 'Agentic_RAG_v2' in mod.__file__:
                                agentic_core_modules[k] = mod
                            del sys.modules[k]
                    
                    # Temporarily set path
                    ragphoto5_path = '/home/wzfeng/RAGPhoto/Data_Agentic_RAG/data_pipeline'
                    temp_path = sys.path.copy()
                    sys.path.clear()
                    sys.path.append(ragphoto5_path)
                    sys.path.extend([p for p in temp_path if p != ragphoto5_path])
                    
                    from core.reranker import RerankerModel
                    self.reranker = RerankerModel()
                    self._RerankerModel = RerankerModel
                    
                    # Restore path and modules
                    sys.path = temp_path
                    for k, mod in agentic_core_modules.items():
                        sys.modules[k] = mod
                    if self.verbose:
                        print("  ✅ Reranker Model initialized")
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️  Reranker initialization failed: {e}")
                    self.reranker = None
            else:
                self.reranker = None
                if self.verbose:
                    print("  ℹ️  Reranker disabled")
            
            self._initialized = True
            
            if self.verbose:
                print("✅ RAG Components initialized successfully")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RAG components: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        # Use RAGPhoto_5's batch method
        return self.embedding_generator.generate_embeddings_batch(texts)
    
    def search_vectors(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Search vector database
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of documents with metadata
        """
        # Validate query
        if not query or not query.strip():
            if self.verbose:
                print("⚠️  Empty query provided to search_vectors")
            return []
        
        query = query.strip()
        
        # Generate query embedding using RAGPhoto_5's method
        query_embedding = self.embedding_generator.generate_embedding(query, is_query=True)
        
        # Search in Milvus (RAGPhoto_5 uses singular 'query_embedding', not 'query_vectors')
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Format results - RAGPhoto_5 already returns a list of dicts
        documents = []
        if results:
            # Validate results is a list
            if not isinstance(results, list):
                if self.verbose:
                    print(f"⚠️  search results is not a list: {type(results)}")
                return []
            
            for hit in results:
                # Validate hit is a dict
                if not isinstance(hit, dict):
                    if self.verbose:
                        print(f"⚠️  Skipping non-dict hit: {type(hit)}")
                    continue
                
                # RAGPhoto_5's search returns dicts with 'content', 'source', 'score' etc.
                # Handle None values properly
                content = hit.get('content', '')
                if content is None:
                    content = ''
                
                metadata = hit.get('metadata', {})
                if metadata is None:
                    metadata = {}
                
                documents.append({
                    'content': content,
                    'source': hit.get('source', 'Unknown'),
                    'similarity': hit.get('score', 0.0),  # RAGPhoto_5 uses 'score'
                    'metadata': metadata,
                    'chunk_id': hit.get('chunk_id', '')
                })
        
        return documents
    
    def query_knowledge_graph(
        self, 
        query: str,
        max_entities: int = 10
    ) -> Dict[str, Any]:
        """
        Query Neo4j knowledge graph
        
        Args:
            query: Query string
            max_entities: Maximum number of entities to return
            
        Returns:
            Dictionary with entities, relationships, and summary
        """
        try:
            # Extract entities from query (simple keyword extraction)
            # In production, use NER model
            keywords = self._extract_keywords(query)
            
            # Query knowledge graph
            entities = []
            relationships = []
            
            for keyword in keywords[:max_entities]:
                # Find entity
                entity_query = f"""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower('{keyword}')
                RETURN e.name as name, e.type as type, e.description as description
                LIMIT 5
                """
                entity_results = self.neo4j_manager.execute_custom_query(entity_query)
                entities.extend(entity_results)
                
                # Find relationships
                rel_query = f"""
                MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
                WHERE toLower(e1.name) CONTAINS toLower('{keyword}')
                RETURN e1.name as source, type(r) as relation, e2.name as target, r.type as rel_type
                LIMIT 10
                """
                rel_results = self.neo4j_manager.execute_custom_query(rel_query)
                relationships.extend(rel_results)
            
            # Build summary
            summary = self._build_kg_summary(entities, relationships)
            
            return {
                'entities': entities,
                'relationships': relationships,
                'summary': summary,
                'count': len(entities)
            }
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Knowledge graph query failed: {e}")
            return {
                'entities': [],
                'relationships': [],
                'summary': '',
                'count': 0
            }
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using reranker model
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_n: Number of top results to return
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Check if reranking is enabled (using agentic_rag config)
        if not config.enable_reranking:
            if self.verbose:
                print("   ℹ️ Reranking disabled in config")
            return documents[:top_n]
        
        try:
            # Use data_pipeline's RerankerModel which has proper /rerank API
            import sys
            ragphoto5_path = '/home/wzfeng/RAGPhoto/Data_Agentic_RAG/data_pipeline'
            if ragphoto5_path not in sys.path:
                sys.path.insert(0, ragphoto5_path)
            
            from core.reranker import RerankerModel
            
            # Use singleton pattern - initialize with agentic_rag config values
            if not hasattr(self, '_reranker_model'):
                self._reranker_model = RerankerModel(
                    model_name=config.reranker_model,
                    api_key=config.reranker_api_key,
                    base_url=config.reranker_base_url
                )
            
            # Extract document contents for reranking
            doc_contents = [doc.get('content', '') for doc in documents]
            
            # Call reranker API
            rerank_results = self._reranker_model.rerank(
                query=query,
                documents=doc_contents,
                top_k=top_n
            )
            
            # Apply rerank scores to documents
            reranked_docs = []
            for original_idx, rerank_score in rerank_results:
                if original_idx < len(documents):
                    doc = documents[original_idx].copy()
                    doc['rerank_score'] = rerank_score
                    reranked_docs.append(doc)
            
            # Print status (compact form)
            if reranked_docs:
                top_score = reranked_docs[0].get('rerank_score', 0)
                print(f"   🔄 Reranked: {len(documents)}→{len(reranked_docs)} (score: {top_score:.3f})")
            
            return reranked_docs
        except Exception as e:
            print(f"   ⚠️ Reranking failed: {e}")
            return documents[:top_n]
    
    def generate_text(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate text using LLM
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        temperature = temperature or config.llm_temperature
        max_tokens = max_tokens or config.llm_max_tokens
        
        # RAGPhoto_5's LLMGenerator doesn't have a 'generate' method
        # Use the underlying OpenAI client directly for custom parameters
        try:
            response = self.llm_generator.client.chat.completions.create(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            if self.verbose:
                print(f"⚠️  LLM generation failed: {e}")
            # Fallback to simple_chat
            return self.llm_generator.simple_chat(prompt)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query (simple implementation)
        
        In production, use:
        - NER model for entity extraction
        - KeyBERT for keyword extraction
        - spaCy for linguistic analysis
        """
        # Simple tokenization and filtering
        import re
        
        # Remove common words
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'which', 'who',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'can'
        }
        
        # Tokenize
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:5]  # Top 5 keywords
    
    def _build_kg_summary(
        self,
        entities: List[Dict],
        relationships: List[Dict]
    ) -> str:
        """Build a text summary from KG entities and relationships"""
        if not entities and not relationships:
            return ""
        
        summary_parts = []
        
        # Summarize entities
        if entities:
            entity_names = [e.get('name', '') for e in entities[:5]]
            summary_parts.append(f"Entities: {', '.join(entity_names)}")
        
        # Summarize relationships
        if relationships:
            rel_summaries = []
            for rel in relationships[:5]:
                source = rel.get('source', '')
                rel_type = rel.get('rel_type', rel.get('relation', ''))
                target = rel.get('target', '')
                rel_summaries.append(f"{source} {rel_type} {target}")
            summary_parts.append(f"Relationships: {'; '.join(rel_summaries)}")
        
        return ". ".join(summary_parts)
    
    def close(self):
        """Close all connections"""
        try:
            if hasattr(self, 'neo4j_manager'):
                self.neo4j_manager.close()
            if hasattr(self, 'vector_store'):
                # Milvus connections are managed by pymilvus
                pass
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error closing connections: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False

