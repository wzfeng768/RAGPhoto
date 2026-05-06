"""
Advanced Photovoltaic Literature Knowledge Extractor
Specialized LLM-based entity and relation extraction for photovoltaic research papers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from typing import Dict, List, Any, Optional, Tuple
import openai

from config.config import config
from .knowledge_graph import Entity, Relation, EntityType, RelationType

class PhotovoltaicLLMExtractor:
    """Advanced LLM extractor specialized for photovoltaic literature"""
    
    def __init__(self):
        """Initialize the photovoltaic literature extractor"""
        print("🔬 Initializing Advanced Photovoltaic Literature Extractor:")
        print(f"  Model: {config.llm_model}")
        print(f"  API URL: {config.openai_base_url}")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        # Advanced entity extraction system prompt for photovoltaic literature
        self.entity_extraction_prompt = """
You are an expert AI assistant specialized in extracting entities from photovoltaic and solar cell research literature. You have deep knowledge of materials science, device physics, and solar cell technology.

**TASK**: Extract entities from photovoltaic research text with high precision and domain expertise.

**ENTITY CATEGORIES** (with specific photovoltaic focus):

🔹 **Material** - Photovoltaic materials and components.
   - Examples: MAPbI3, FAPbI3, P3HT, PCBM, PM6, Y6, ITO, Ag, Au
   - **Note**: Chemical formulas in LaTeX format (e.g., $\\mathrm{MAPbI_3}$) should be extracted as Material entities

🔹 **Device** - Photovoltaic devices and their layers/components.
   - Examples: Tandem Cell, PERC Cell, ETL, HTL, Active Layer, Electrode

🔹 **Metric** - **IMPORTANT**: A performance indicator or measured property. It MUST have a numerical value. Do NOT extract concepts without values here.
   - Examples: "PCE of 25.2%", "Voc of 1.1 V", "bandgap of 1.5 eV"
   - **Rule**: A metric is the CONCEPT, not the value. E.g., for "PCE of 25.2%", extract "PCE" as the Metric. The value "25.2%" goes into its properties.
   - **LaTeX Numbers**: When you see LaTeX formatted numbers like $1 9 . 2 0 \\%$, interpret it as 19.20% (remove spaces between digits)

🔹 **Process** - Fabrication and characterization techniques.
   - Examples: Spin Coating, Annealing, Sputtering, PECVD, SEM, XRD

🔹 **Parameter** - **NEW**: Conditions that influence an experiment or measurement.
   - Examples: Temperature, Illumination (e.g., AM1.5G), Humidity, Time
   - **Note**: Mathematical parameters in formulas (e.g., $V_{OC}$, $J_{SC}$) should be extracted as Parameter entities

🔹 **Formula** - **NEW**: Mathematical equations, formulas, and expressions.
   - Examples: $Z = \\mathrm{ReLU}[\\mathrm{Conv1d}(x)]$, $PCE = \\frac{V_{OC} \\times J_{SC} \\times FF}{P_{in}}$
   - **Extraction Rule**: Extract complete formulas, including both inline ($...$) and block ($$...$$) formulas
   - **Properties**: Store the formula in LaTeX format, and if applicable, extract variables and their meanings

🔹 **Concept** - General scientific concepts that are not metrics.
   - Examples: Charge Transport, Recombination, Quantum Efficiency, Phase Stability

**OUTPUT FORMAT**:
\`\`\`json
{
  "entities": [
    {
      "name": "Entity name (standardized, e.g., 'PCE' not 'Power Conversion Efficiency')",
      "type": "Material|Device|Metric|Process|Parameter|Concept|Formula",
      "properties": {
        "value": "25.2 (if applicable, numeric only, cleaned from LaTeX format)",
        "unit": "% (if applicable)",
        "condition": "AM1.5G (if applicable)",
        "latex_formula": "$Z = \\mathrm{ReLU}[\\mathrm{Conv1d}(x)]$ (for Formula entities)",
        "variables": ["x", "Z"] (for Formula entities, list of variables),
        "description": "Brief description of what the formula represents"
      }
    }
  ]
}
\`\`\`

**EXTRACTION GUIDELINES**:
1.  **Strict Typing**: Adhere strictly to the types. If "Efficiency" is mentioned without a number, it's a 'Concept', but if it's "Efficiency of 22%", it's a 'Metric'.
2.  **Atomize Metrics**: For "a PCE of 25.2% was achieved", create ONE 'Metric' entity named "PCE" with properties `{"value": 25.2, "unit": "%"}`.
3.  **Standardize Names**: Use common acronyms (PCE, Voc, Jsc, ETL).
4.  **Formula Extraction**: 
   - Extract complete mathematical formulas as 'Formula' entities
   - Preserve LaTeX format in the formula property
   - Extract variables from formulas as separate 'Parameter' entities when they represent physical quantities
   - For formulas like $V_{OC}$ or $J_{SC}$, extract both the formula and the parameter
5.  **LaTeX Number Interpretation**: 
   - When you see $1 9 . 2 0 \\%$, interpret it as 19.20% (spaces between digits should be removed)
   - When you see $1 6 . 5 5 \\%$, interpret it as 16.55%
   - Extract the cleaned numeric value in the properties
6.  **Chemical Formulas**: 
   - Extract chemical formulas in LaTeX format (e.g., $\\mathrm{MAPbI_3}$) as 'Material' entities
   - Store both the LaTeX representation and the cleaned name (e.g., "MAPbI3")

**TEXT TO ANALYZE**:
${text}
`;
    }

    /**
     * @param {string} text
     * @param {string} entities
     * @returns {string}
     */
    buildRelationExtractionPrompt(text, entities) {
        return `
You are an expert AI assistant specialized in extracting relationships between entities in photovoltaic research literature.

**TASK**: Extract meaningful scientific relationships between the provided entities based on the text.

**RELATIONSHIP TYPES**:

🔗 **HAS_PROPERTY** - Connects a Material/Device to a Metric.
   - Example: (Perovskite Layer) -[HAS_PROPERTY]-> (PCE)

🔗 **FABRICATED_BY** - Connects a Device/Material to a Process.
   - Example: (Active Layer) -[FABRICATED_BY]-> (Spin Coating)

🔗 **MEASURES** - Connects a Process (measurement technique) to a Metric.
   - Example: (J-V Measurement) -[MEASURES]-> (Voc)

🔗 **INFLUENCES** - **NEW**: Connects a Parameter/Process/Material to a Metric, describing a cause-and-effect relationship.
   - Example: (High Temperature) -[INFLUENCES]-> (PCE)
   - **Property**: Capture the effect, e.g., `{"effect": "decreases", "magnitude": "significantly"}`.

🔗 **IS_A** - **NEW**: Represents a classification or hierarchical relationship.
   - Example: (PM6) -[IS_A]-> (Polymer Donor)

🔗 **USES** - Connects a device or process to a material.
   - Example: (Tandem Solar Cell) -[USES]-> (ITO)

**ENTITY LIST**:
${entities}

**OUTPUT FORMAT**:
\`\`\`json
{
  "relations": [
    {
      "source_entity": "Source entity name",
      "target_entity": "Target entity name",
      "relation_type": "HAS_PROPERTY|FABRICATED_BY|MEASURES|INFLUENCES|IS_A|USES",
      "properties": {
        "effect": "increases/decreases (for INFLUENCES)",
        "context": "Under illumination"
      }
    }
  ]
}
\`\`\`

**EXTRACTION GUIDELINES**:
1.  **Accuracy First**: Only extract relationships explicitly stated or strongly implied in the text.
2.  **Focus on New Types**: Pay special attention to extracting `INFLUENCES` and `IS_A` relationships.
3.  **Capture Effect**: For `INFLUENCES` relationships, always try to fill the `effect` property.

**TEXT CONTENT**:
${text}
`;
    }

    def extract_entities_from_text(self, text: str, paper_id: str = None) -> List[Entity]:
        """Extract entities using advanced photovoltaic-specific prompts"""
        if not text or len(text.strip()) < 30:
            return []
        
        try:
            # Prepare the prompt - 使用字符串模板来避免.format()的大括号问题
            from string import Template
            # 将 {text} 替换为 $text 以便使用 Template
            template_str = self.entity_extraction_prompt.replace('{text}', '$text')
            template = Template(template_str)
            prompt = template.safe_substitute(text=text)
            
            # Call LLM with optimized parameters for scientific extraction
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a world-renowned expert in photovoltaic research with deep knowledge of materials science, device physics, and solar cell technology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,  # Very low temperature for precise scientific extraction
                max_tokens=8000,  # 增加到8000以支持更多实体提取
                top_p=0.9
            )
            
            # Parse response
            content = response.choices[0].message.content
            entities = self._parse_entity_response(content, paper_id)
            
            print(f"      🎯 Advanced extraction: {len(entities)} high-confidence entities")
            return entities
            
        except Exception as e:
            print(f"   ⚠️ Advanced entity extraction failed: {e}")
            return []
    
    def extract_relations_from_text(self, text: str, entities: List[Entity], paper_id: str = None) -> List[Relation]:
        """Extract relations using advanced photovoltaic-specific understanding"""
        if not text or not entities or len(entities) < 2:
            return []
        
        try:
            # Prepare entity list with technical details
            entity_list = []
            for entity in entities:
                entity_info = f"- {entity.name} ({entity.type.value})"
                if entity.properties:
                    props = [f"{k}: {v}" for k, v in entity.properties.items() if v]
                    if props:
                        entity_info += f" [{', '.join(props)}]"
                entity_list.append(entity_info)
            
            entity_str = "\n".join(entity_list)
            # Prepare the prompt - 使用字符串模板来避免.format()的大括号问题
            from string import Template
            # 将占位符替换为 $ 格式以便使用 Template
            template_str = self.relation_extraction_prompt.replace('{entities}', '$entities').replace('{text}', '$text')
            template = Template(template_str)
            prompt = template.safe_substitute(entities=entity_str, text=text)
            
            # Call LLM with scientific expertise
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a leading expert in photovoltaic device physics and materials science, capable of understanding complex relationships between materials, processes, and performance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,  # Very low temperature for precise relationship extraction
                max_tokens=8000,  # 增加到8000以支持更多关系提取
                top_p=0.9
            )
            
            # Parse response
            content = response.choices[0].message.content
            relations = self._parse_relation_response(content, entities, paper_id)
            
            print(f"      🔗 Advanced relations: {len(relations)} scientific relationships")
            return relations
            
        except Exception as e:
            print(f"   ⚠️ Advanced relation extraction failed: {e}")
            return []
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean JSON string to handle common issues:
        - Invalid escape sequences (e.g., \\\ -> \\)
        - Control characters
        - Unicode issues
        """
        try:
            # Remove control characters except newlines and tabs
            json_str = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', json_str)
            
            # Fix invalid escape sequences in string values
            # Pattern: match string values (content between quotes)
            def fix_escapes_in_string(match):
                content = match.group(1)
                
                # Step 1: Fix three or more consecutive backslashes (\\\ -> \\)
                # This handles cases like $\\\mathrm{...}$ -> $\\mathrm{...}$
                # Strategy: replace odd sequences (3, 5, 7, ...) with even pairs
                # 3 backslashes -> 2, 5 -> 4, 7 -> 6, etc.
                # Even sequences (4, 6, 8, ...) are kept as is
                def fix_backslashes(m):
                    count = len(m.group(0))
                    if count % 2 == 1:
                        # Odd: reduce by 1 to make it even (3->2, 5->4, 7->6)
                        return '\\\\' * ((count - 1) // 2)
                    else:
                        # Even: keep as pairs (4->4, 6->6, 8->8)
                        return '\\\\' * (count // 2)
                content = re.sub(r'\\{3,}', fix_backslashes, content)
                
                # Step 2: Fix invalid escape sequences (but preserve valid ones)
                # Valid escapes: \n, \t, \r, \b, \f, \", \\, \/, \uXXXX
                # Invalid escapes like \m, \a, etc. should be escaped as \\m, \\a
                # But we need to be careful not to break valid sequences
                def fix_invalid_escape(m):
                    seq = m.group(0)
                    # If it's a valid escape sequence, keep it
                    if seq in ['\\n', '\\t', '\\r', '\\b', '\\f', '\\"', '\\\\', '\\/']:
                        return seq
                    # If it's a Unicode escape \uXXXX, keep it
                    if seq.startswith('\\u') and len(seq) == 6:
                        return seq
                    # Otherwise, escape the backslash: \m -> \\m
                    return '\\\\' + seq[1:]
                
                # Match backslash followed by a character
                content = re.sub(r'\\.', fix_invalid_escape, content)
                
                return f'"{content}"'
            
            # Match string values (content between quotes)
            # This regex matches: "..." where ... can contain escaped quotes
            json_str = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', fix_escapes_in_string, json_str)
            
            return json_str.strip()
        except Exception as e:
            print(f"      ⚠️ Error cleaning JSON string: {e}")
            return json_str  # Return original if cleaning fails
    
    def _parse_entity_response(self, content: str, paper_id: str = None) -> List[Entity]:
        """Parse LLM entity response with enhanced error handling"""
        entities = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_str = content.strip()
            
            # Clean up JSON string before parsing
            json_str = self._clean_json_string(json_str)
            
            # Parse JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to clean up common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                json_str = self._clean_json_string(json_str)  # Clean again
                try:
                data = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print(f"      ⚠️ JSON parsing error after cleanup: {e2}")
                    print(f"      ⚠️ Problematic JSON (first 500 chars): {json_str[:500]}")
                    return entities  # Return empty or partial results
            
            # Process entities
            for entity_data in data.get('entities', []):
                try:
                    # Get entity type with enhanced mapping
                    entity_type_str = entity_data.get('type', 'Concept').strip()
                    entity_name = entity_data.get('name', '').strip()
                    
                    # Enhanced type mapping for photovoltaic domain
                    entity_type = self._map_entity_type(entity_type_str, entity_name)
                    
                    # Generate unique ID based on name and type (for entity merging)
                    entity_id = Entity.generate_id(entity_name, entity_type)
                    
                    # Create entity with enhanced properties
                    entity = Entity(
                        id=entity_id,
                        name=entity_name,
                        type=entity_type,
                        description=entity_data.get('description', ''),
                        properties=entity_data.get('properties', {}),
                        confidence=float(entity_data.get('confidence', 0.8)),
                        source_papers=[paper_id] if paper_id else []
                    )
                    
                    # Only include high-confidence entities for scientific accuracy
                    if entity.confidence >= 0.7:
                        entities.append(entity)
                        
                except Exception as e:
                    print(f"      ⚠️ Error processing entity: {e}")
                    continue
                    
        except Exception as e:
            print(f"   ⚠️ Entity response parsing failed: {e}")
            
        return entities
    
    def _map_entity_type(self, type_str: str, entity_name: str) -> EntityType:
        """Enhanced mapping of entity types using both type string and entity name"""
        type_str = type_str.upper()
        entity_name_lower = entity_name.lower()
        
        # Direct type mapping
        if type_str in ['MATERIAL', 'MAT']:
            return EntityType.MATERIAL
        elif type_str in ['DEVICE', 'DEV']:
            return EntityType.DEVICE
        elif type_str in ['METRIC', 'MET']:
            return EntityType.METRIC
        elif type_str in ['PROCESS', 'PROC']:
            return EntityType.PROCESS
        elif type_str in ['MEASUREMENT', 'MEAS']:
            return EntityType.MEASUREMENT
        elif type_str in ['AUTHOR', 'AUTH']:
            return EntityType.AUTHOR
        elif type_str in ['INSTITUTION', 'INST']:
            return EntityType.INSTITUTION
        elif type_str in ['FORMULA', 'FORM', 'EQUATION', 'EQ']:
            return EntityType.FORMULA
        elif type_str in ['PARAMETER', 'PARAM']:
            return EntityType.PARAMETER
        elif type_str in ['CONCEPT', 'CONC']:
            return EntityType.CONCEPT
        
        # Enhanced name-based mapping for photovoltaic domain
        
        # Materials
        material_keywords = ['mapbi3', 'fapbi3', 'cspbi3', 'perovskite', 'silicon', 'si', 'sno2', 'tio2', 
                           'fto', 'ito', 'azo', 'spiro-ometad', 'p3ht', 'pcbm', 'ptb7', 'pm6', 'y6',
                           'cdte', 'cigs', 'cigse', 'gaas', 'inp', 'au', 'ag', 'al', 'cu', 'eva', 'poe',
                           'kf', 'licf3so2', 'oxide', 'metal', 'polymer', 'organic', 'dopant']
        
        # Devices
        device_keywords = ['solar cell', 'cell', 'module', 'panel', 'tandem', 'heterojunction', 'perc',
                          'bifacial', 'led', 'laser', 'photodetector', 'etl', 'htl', 'layer', 'junction',
                          'electrode', 'contact', 'busbar', 'ribbon']
        
        # Metrics
        metric_keywords = ['pce', 'efficiency', 'voc', 'jsc', 'ff', 'fill factor', 'bandgap', 'mobility',
                          'conductivity', 'resistance', 'transmittance', 'reflectance', 'absorption',
                          't50', 't80', 'lifetime', 'degradation', 'thickness', 'temperature', 'voltage',
                          'current', 'power', 'energy', 'lcoe', 'yield', 'ratio', '%', 'ev', 'mv', 'ma', 'cm']
        
        # Processes
        process_keywords = ['spin coating', 'coating', 'evaporation', 'sputtering', 'pecvd', 'ald',
                           'annealing', 'thermal', 'sintering', 'etching', 'plasma', 'deposition',
                           'fabrication', 'passivation', 'doping', 'treatment', 'cleaning', 'texturing']
        
        # Measurements
        measurement_keywords = ['xrd', 'sem', 'tem', 'afm', 'xps', 'ftir', 'raman', 'uv-vis', 'pl', 'el',
                               'j-v', 'i-v', 'cv', 'eqe', 'iqe', 'ipce', 'impedance', 'spectroscopy',
                               'microscopy', 'diffraction', 'analysis', 'characterization', 'am1.5g']
        
        # Check entity name against keyword lists
        for keyword in material_keywords:
            if keyword in entity_name_lower:
                return EntityType.MATERIAL
        
        for keyword in device_keywords:
            if keyword in entity_name_lower:
                return EntityType.DEVICE
        
        for keyword in metric_keywords:
            if keyword in entity_name_lower:
                return EntityType.METRIC
        
        for keyword in process_keywords:
            if keyword in entity_name_lower:
                return EntityType.PROCESS
        
        for keyword in measurement_keywords:
            if keyword in entity_name_lower:
                return EntityType.MEASUREMENT
        
        # Formula detection: check if entity name contains LaTeX-like patterns
        if '$' in entity_name or '\\' in entity_name or '=' in entity_name:
            # Check for common formula patterns
            formula_patterns = ['=', '\\frac', '\\mathrm', '\\sum', '\\int', '\\sqrt', '^', '_', '\\times', '\\div']
            if any(pattern in entity_name for pattern in formula_patterns):
                return EntityType.FORMULA
        
        # Default fallback
        return EntityType.CONCEPT
    
    def _parse_relation_response(self, content: str, entities: List[Entity], paper_id: str = None) -> List[Relation]:
        """Parse LLM relation response with scientific validation"""
        relations = []
        entity_name_to_id = {entity.name: entity.id for entity in entities}
        
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content.strip()
            
            # Clean up JSON string before parsing
            json_str = self._clean_json_string(json_str)
            
            # Parse JSON with cleanup
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to clean up common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = self._clean_json_string(json_str)  # Clean again
                try:
                data = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print(f"      ⚠️ JSON parsing error after cleanup: {e2}")
                    print(f"      ⚠️ Problematic JSON (first 500 chars): {json_str[:500]}")
                    return relations  # Return empty or partial results
            
            # Process relations
            for relation_data in data.get('relations', []):
                try:
                    source_name = relation_data.get('source_entity', '').strip()
                    target_name = relation_data.get('target_entity', '').strip()
                    
                    # Validate entity existence
                    if source_name not in entity_name_to_id or target_name not in entity_name_to_id:
                        continue
                    
                    # Get relation type
                    relation_type_str = relation_data.get('relation_type', 'RELATED_TO')
                    try:
                        relation_type = RelationType(relation_type_str.upper())
                    except ValueError:
                        relation_type = RelationType.RELATED_TO  # Default fallback
                    
                    # Create relation with scientific validation
                    relation = Relation(
                        id=f"{paper_id}_{len(relations)}" if paper_id else f"relation_{len(relations)}",
                        source_id=entity_name_to_id[source_name],
                        target_id=entity_name_to_id[target_name],
                        type=relation_type,
                        description=relation_data.get('description', ''),
                        properties=relation_data.get('properties', {}),
                        confidence=float(relation_data.get('confidence', 0.8)),
                        source_papers=[paper_id] if paper_id else []
                    )
                    
                    # Only include high-confidence relationships
                    if relation.confidence >= 0.7:
                        relations.append(relation)
                        
                except Exception as e:
                    print(f"      ⚠️ Error processing relation: {e}")
                    continue
                    
        except Exception as e:
            print(f"   ⚠️ Relation response parsing failed: {e}")
            
        return relations
    
    def extract_entities_and_relations_batch(
        self, 
        text_chunks: List[str], 
        paper_id: str = None,
        paper_ids: List[str] = None
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Batch extraction with advanced processing for scientific literature
        
        Args:
            text_chunks: List of text chunks to process
            paper_id: Single paper ID (if all chunks from same paper)
            paper_ids: List of paper IDs (one per chunk, if provided)
        
        Returns:
            Tuple of (entities, relations)
        """
        all_entities = []
        all_relations = []
        
        print(f"   🔬 Advanced batch processing: {len(text_chunks)} chunks")
        
        for i, chunk in enumerate(text_chunks):
            # 确定当前chunk的paper_id
            if paper_ids and i < len(paper_ids):
                current_paper_id = paper_ids[i]
            elif paper_id:
                current_paper_id = f"{paper_id}_chunk_{i}"
            else:
                current_paper_id = f"chunk_{i}"
            
            print(f"      Processing chunk {i+1}/{len(text_chunks)} (paper: {current_paper_id})...")
            
            # Extract entities
            entities = self.extract_entities_from_text(chunk, current_paper_id)
            all_entities.extend(entities)
            
            # Extract relations
            if entities:
                relations = self.extract_relations_from_text(chunk, entities, current_paper_id)
                all_relations.extend(relations)
        
        print(f"   ✅ Advanced batch complete: {len(all_entities)} entities, {len(all_relations)} relations")
        return all_entities, all_relations 