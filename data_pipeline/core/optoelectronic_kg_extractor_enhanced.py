"""
Enhanced Traceable Optoelectronic Knowledge Graph Extractor
============================================================

Enhanced version with improved entity and relation extraction for photovoltaic
research literature, with special focus on:
- Machine learning algorithms and methods
- Performance metrics and their values
- Material-performance relationships
- Experimental conditions and parameters

Version: 3.0.0 - Domain-Enhanced Extraction
Date: 2025-11-30
Author: RAGPhoto Team
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import openai

from config.config import config
from config.kg_extraction_prompts_enhanced import (
    ENTITY_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    RELATION_EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_CONFIG,
    validate_entity_response,
    validate_relation_response,
    is_ml_algorithm_keyword,
    is_performance_metric_keyword,
    extract_numerical_values
)

# Import base classes from parent module
from .optoelectronic_kg_extractor import (
    OptoelectronicKGExtractor,
    SourceTraceability,
    TraceableEntity,
    TraceableRelation
)


class EnhancedOptoelectronicKGExtractor(OptoelectronicKGExtractor):
    """
    Enhanced knowledge graph extractor with improved domain-specific extraction
    for photovoltaic and optoelectronic research literature
    """
    
    def __init__(self, enable_traceability: bool = True):
        """
        Initialize the enhanced optoelectronic KG extractor
        
        Args:
            enable_traceability: Whether to enforce traceability requirements
        """
        print("🔬 Initializing Enhanced Traceable Optoelectronic KG Extractor:")
        print(f"  Model: {config.llm_model}")
        print(f"  API URL: {config.openai_base_url}")
        print(f"  Traceability: {'✓ Enabled' if enable_traceability else '✗ Disabled'}")
        print(f"  🆕 Enhanced Features:")
        print(f"     - ML algorithm detection")
        print(f"     - Performance metric extraction")
        print(f"     - Material-performance relationships")
        print(f"     - Experimental condition tracking")
        
        self.enable_traceability = enable_traceability
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        # Load enhanced extraction configuration
        self.entity_config = EXTRACTION_CONFIG["entity_extraction"]
        self.relation_config = EXTRACTION_CONFIG["relation_extraction"]
        self.traceability_config = EXTRACTION_CONFIG["traceability"]
        self.domain_keywords = EXTRACTION_CONFIG["domain_keywords"]
        
        # Enhanced statistics tracking
        self.stats = {
            "total_entities": 0,
            "total_relations": 0,
            "entities_with_traceability": 0,
            "relations_with_traceability": 0,
            "extraction_failures": 0,
            "reference_chunks_filtered": 0,
            # Enhanced stats
            "ml_algorithms_extracted": 0,
            "performance_metrics_extracted": 0,
            "research_methods_extracted": 0,
            "prediction_relations_extracted": 0,
            "optimization_relations_extracted": 0
        }
    
    def _analyze_chunk_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze chunk content to identify domain-specific features
        
        Returns:
            Dictionary with content analysis results
        """
        text_lower = text.lower()
        
        analysis = {
            "has_ml_content": is_ml_algorithm_keyword(text),
            "has_performance_metrics": is_performance_metric_keyword(text),
            "has_numerical_data": len(extract_numerical_values(text)) > 0,
            "ml_keywords_found": [],
            "metric_keywords_found": [],
            "numerical_values": extract_numerical_values(text)
        }
        
        # Find specific ML keywords
        for keyword in self.domain_keywords["ml_algorithms"]:
            if keyword in text_lower:
                analysis["ml_keywords_found"].append(keyword)
        
        # Find specific metric keywords
        for keyword in self.domain_keywords["performance_metrics"]:
            if keyword in text_lower:
                analysis["metric_keywords_found"].append(keyword)
        
        return analysis
    
    def extract_entities_from_text(
        self, 
        text: str, 
        paper_id: str = None,
        chunk_id: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TraceableEntity]:
        """
        Enhanced entity extraction with domain-specific analysis
        
        Args:
            text: Text to extract entities from
            paper_id: Identifier for the source paper
            chunk_id: Identifier for the text chunk
            metadata: Additional metadata (section, page, etc.)
        
        Returns:
            List of traceable entities with enhanced domain coverage
        """
        if not text or len(text.strip()) < self.entity_config["min_text_length"]:
            return []
        
        # Check if it's reference content
        if self._is_reference_content(text, metadata):
            self.stats["reference_chunks_filtered"] += 1
            print(f"      🚫 跳过参考文献内容 (chunk_id: {chunk_id or 'unknown'})")
            return []
        
        # Analyze chunk content for domain features
        content_analysis = self._analyze_chunk_content(text)
        
        if content_analysis["has_ml_content"]:
            print(f"      🤖 检测到ML内容: {', '.join(content_analysis['ml_keywords_found'][:3])}")
        
        if content_analysis["has_performance_metrics"]:
            print(f"      📊 检测到性能指标: {', '.join(content_analysis['metric_keywords_found'][:3])}")
        
        try:
            # Prepare the prompt using enhanced version
            from string import Template
            template_str = ENTITY_EXTRACTION_PROMPT.replace('{text}', '$text')
            template = Template(template_str)
            prompt = template.safe_substitute(text=text)
            
            # Add analysis hints to the prompt if available
            if content_analysis["has_ml_content"] or content_analysis["has_performance_metrics"]:
                hint = "\n\n**EXTRACTION HINTS** (based on content analysis):\n"
                if content_analysis["has_ml_content"]:
                    hint += f"- This text mentions ML methods: {', '.join(content_analysis['ml_keywords_found'][:5])}\n"
                    hint += "- Pay special attention to extracting ML_Algorithm entities\n"
                if content_analysis["has_performance_metrics"]:
                    hint += f"- This text mentions performance metrics: {', '.join(content_analysis['metric_keywords_found'][:5])}\n"
                    hint += "- Pay special attention to extracting Metric entities with values\n"
                if content_analysis["numerical_values"]:
                    hint += f"- Found {len(content_analysis['numerical_values'])} numerical values\n"
                
                prompt += hint
            
            # Call LLM with optimized parameters
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.entity_config["temperature"],
                max_tokens=self.entity_config["max_tokens"],
                top_p=self.entity_config["top_p"]
            )
            
            # Parse response with enhanced analysis
            content = response.choices[0].message.content
            entities = self._parse_entity_response(
                content, 
                paper_id=paper_id,
                chunk_id=chunk_id,
                metadata=metadata,
                content_analysis=content_analysis
            )
            
            # Update statistics
            self.stats["total_entities"] += len(entities)
            self.stats["entities_with_traceability"] += sum(
                1 for e in entities if e.source_traceability.text_snippet
            )
            
            # Track domain-specific entities
            for entity in entities:
                if entity.type.value == "ML_Algorithm":
                    self.stats["ml_algorithms_extracted"] += 1
                elif entity.type.value == "Metric":
                    self.stats["performance_metrics_extracted"] += 1
                elif entity.type.value == "Research_Method":
                    self.stats["research_methods_extracted"] += 1
            
            print(f"      🎯 提取了 {len(entities)} 个实体")
            if self.enable_traceability:
                traced = sum(1 for e in entities if e.source_traceability.text_snippet)
                print(f"         ✓ {traced}/{len(entities)} 具有完整追溯信息")
            
            # Report domain-specific extraction
            ml_count = sum(1 for e in entities if e.type.value == "ML_Algorithm")
            metric_count = sum(1 for e in entities if e.type.value == "Metric")
            if ml_count > 0 or metric_count > 0:
                print(f"         🎯 领域特定: {ml_count} ML算法, {metric_count} 性能指标")
            
            return entities
            
        except Exception as e:
            print(f"   ⚠️ Entity extraction failed: {e}")
            self.stats["extraction_failures"] += 1
            return []
    
    def _parse_entity_response(
        self,
        content: str,
        paper_id: str = None,
        chunk_id: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        content_analysis: Optional[Dict[str, Any]] = None
    ) -> List[TraceableEntity]:
        """
        Enhanced entity response parsing with domain-specific validation
        """
        # Use parent class parsing
        entities = super()._parse_entity_response(content, paper_id, chunk_id, metadata)
        
        # Enhanced post-processing based on content analysis
        if content_analysis:
            # Validate ML algorithms
            if content_analysis["has_ml_content"]:
                ml_entities = [e for e in entities if e.type.value == "ML_Algorithm"]
                if len(ml_entities) == 0 and len(content_analysis["ml_keywords_found"]) > 0:
                    print(f"      ⚠️ 警告: 检测到ML关键词但未提取到ML_Algorithm实体")
            
            # Validate performance metrics
            if content_analysis["has_performance_metrics"]:
                metric_entities = [e for e in entities if e.type.value == "Metric"]
                if len(metric_entities) == 0 and len(content_analysis["metric_keywords_found"]) > 0:
                    print(f"      ⚠️ 警告: 检测到性能指标关键词但未提取到Metric实体")
        
        return entities
    
    def extract_relations_from_text(
        self,
        text: str,
        entities: List[TraceableEntity],
        paper_id: str = None,
        chunk_id: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TraceableRelation]:
        """
        Enhanced relation extraction with domain-specific analysis
        """
        if not text or not entities or len(entities) < self.relation_config["min_entities"]:
            return []
        
        # Analyze entities for domain-specific relationship extraction
        entity_types = [e.type.value for e in entities]
        has_ml_algorithm = "ML_Algorithm" in entity_types
        has_metric = "Metric" in entity_types
        has_material = "Material" in entity_types
        has_process = "Process" in entity_types
        
        try:
            # Prepare entity list with enhanced formatting
            entity_list = self._format_entity_list_for_prompt(entities)
            
            # Add relationship hints based on entity composition
            from string import Template
            template_str = RELATION_EXTRACTION_PROMPT.replace('{entities}', '$entities').replace('{text}', '$text')
            template = Template(template_str)
            prompt = template.safe_substitute(entities=entity_list, text=text)
            
            # Add extraction hints
            if has_ml_algorithm and has_metric:
                hint = "\n\n**RELATIONSHIP HINTS**:\n"
                hint += "- This text contains both ML algorithms and metrics\n"
                hint += "- Look for PREDICTS relationships (ML_Algorithm → Metric)\n"
                prompt += hint
            
            if has_material and has_metric:
                hint = "\n\n**RELATIONSHIP HINTS**:\n"
                hint += "- This text contains both materials and metrics\n"
                hint += "- Look for HAS_PROPERTY relationships (Material → Metric)\n"
                prompt += hint
            
            if has_process and has_metric:
                hint = "\n\n**RELATIONSHIP HINTS**:\n"
                hint += "- This text contains both processes and metrics\n"
                hint += "- Look for INFLUENCES relationships (Process → Metric)\n"
                prompt += hint
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": RELATION_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.relation_config["temperature"],
                max_tokens=self.relation_config["max_tokens"],
                top_p=self.relation_config["top_p"]
            )
            
            # Parse response
            content = response.choices[0].message.content
            relations = self._parse_relation_response(
                content,
                entities,
                paper_id=paper_id,
                chunk_id=chunk_id,
                metadata=metadata
            )
            
            # Update statistics
            self.stats["total_relations"] += len(relations)
            self.stats["relations_with_traceability"] += sum(
                1 for r in relations if r.source_traceability.text_snippet
            )
            
            # Track domain-specific relations
            for relation in relations:
                if relation.type.value == "PREDICTS":
                    self.stats["prediction_relations_extracted"] += 1
                elif relation.type.value == "OPTIMIZES":
                    self.stats["optimization_relations_extracted"] += 1
            
            print(f"      🔗 提取了 {len(relations)} 个关系")
            if self.enable_traceability:
                traced = sum(1 for r in relations if r.source_traceability.text_snippet)
                print(f"         ✓ {traced}/{len(relations)} 具有完整追溯信息")
            
            # Report domain-specific relations
            pred_count = sum(1 for r in relations if r.type.value == "PREDICTS")
            opt_count = sum(1 for r in relations if r.type.value == "OPTIMIZES")
            if pred_count > 0 or opt_count > 0:
                print(f"         🎯 领域特定: {pred_count} 预测关系, {opt_count} 优化关系")
            
            return relations
            
        except Exception as e:
            print(f"   ⚠️ Relation extraction failed: {e}")
            self.stats["extraction_failures"] += 1
            return []
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get enhanced extraction statistics"""
        stats = super().get_extraction_statistics()
        
        # Add domain-specific statistics
        stats.update({
            "ml_algorithms_extracted": self.stats["ml_algorithms_extracted"],
            "performance_metrics_extracted": self.stats["performance_metrics_extracted"],
            "research_methods_extracted": self.stats["research_methods_extracted"],
            "prediction_relations_extracted": self.stats["prediction_relations_extracted"],
            "optimization_relations_extracted": self.stats["optimization_relations_extracted"],
        })
        
        return stats
    
    def print_statistics(self):
        """Print enhanced extraction statistics"""
        stats = self.get_extraction_statistics()
        print("\n" + "="*70)
        print("📊 ENHANCED EXTRACTION STATISTICS")
        print("="*70)
        print(f"Total Entities Extracted:        {stats['total_entities']}")
        print(f"  - With Traceability:           {stats['entities_with_traceability']} ({stats['entity_traceability_rate']:.1f}%)")
        print(f"  - 🤖 ML Algorithms:             {stats['ml_algorithms_extracted']}")
        print(f"  - 📊 Performance Metrics:       {stats['performance_metrics_extracted']}")
        print(f"  - 🎓 Research Methods:          {stats['research_methods_extracted']}")
        print(f"Total Relations Extracted:       {stats['total_relations']}")
        print(f"  - With Traceability:           {stats['relations_with_traceability']} ({stats['relation_traceability_rate']:.1f}%)")
        print(f"  - 🔮 Prediction Relations:      {stats['prediction_relations_extracted']}")
        print(f"  - ⚡ Optimization Relations:    {stats['optimization_relations_extracted']}")
        print(f"Reference Chunks Filtered:       {stats['reference_chunks_filtered']}")
        print(f"Extraction Failures:             {stats['extraction_failures']}")
        print("="*70 + "\n")
