"""
Knowledge Graph Extraction Prompts for Optoelectronic Research Literature
===========================================================================

Prompts for extracting traceable knowledge from optoelectronic research papers.

Version: 2.3.0
Date: 2025-10-22
"""

# Entity extraction prompt with escaped braces for .format()
ENTITY_EXTRACTION_PROMPT = """
You are an expert AI assistant specialized in extracting entities from optoelectronic and photovoltaic research literature. 
You have deep expertise in materials science, device physics, semiconductor technology, and solar cells.

**CRITICAL REQUIREMENT: TRACEABILITY**
For every entity you extract, you MUST include source traceability information to enable verification and citation.

**YOUR TASK**: 
Extract entities from optoelectronic research text with high precision, domain expertise, and complete source traceability.

**ENTITY CATEGORIES**:
- **Material**: Photovoltaic materials, chemical compounds (e.g., MAPbI3, P3HT, ITO)
  - **Note**: Extract chemical formulas in LaTeX format (e.g., $\\mathrm{MAPbI_3}$) as Material entities
- **Device**: Photovoltaic devices and components (e.g., Solar Cell, ETL, HTL)
- **Metric**: Performance indicators with numerical values (e.g., PCE, Voc, Jsc)
  - **LaTeX Numbers**: When you see $1 9 . 2 0 \\%$, interpret it as 19.20% (remove spaces)
- **Process**: Fabrication and characterization techniques (e.g., Spin Coating, Annealing)
- **Parameter**: Experimental conditions and mathematical parameters (e.g., Temperature, $V_{{OC}}$)
- **Formula**: **NEW** - Mathematical equations and expressions
  - Extract complete formulas: inline ($...$) and block ($$...$$) formulas
  - Examples: $Z = \\mathrm{{ReLU}}[\\mathrm{{Conv1d}}(x)]$, $PCE = \\frac{{V_{{OC}} \\times J_{{SC}} \\times FF}}{{P_{{in}}}}$
  - Store LaTeX format in properties, extract variables separately
- **Measurement**: Measurement techniques (e.g., XRD, SEM, J-V)
- **Concept**: General scientific concepts (e.g., Charge Transport, Recombination)

**EXTRACTION GUIDELINES**:
1. **Formula Extraction**: Extract complete mathematical formulas as 'Formula' entities
2. **LaTeX Number Interpretation**: $1 9 . 2 0 \\%$ means 19.20% (remove spaces between digits)
3. **Chemical Formulas**: Extract LaTeX chemical formulas (e.g., $\\mathrm{{MAPbI_3}}$) as Material entities
4. **Formula Variables**: Extract variables from formulas as Parameter entities when they represent physical quantities
5. **CRITICAL: EXCLUDE REFERENCES**: Do NOT extract entities from reference lists, bibliographies, or citation content. Skip texts that primarily contain citations, author names with years, DOI links, or journal references.

**TEXT TO ANALYZE**:
{text}

**OUTPUT FORMAT** (JSON with Traceability):
```json
{{
  "entities": [
    {{
      "name": "Entity name",
      "type": "Material|Device|Metric|Process|Measurement|Parameter|Concept|Formula",
      "description": "Brief description",
      "properties": {{
        "value": "25.2 (cleaned from LaTeX format if applicable)",
        "unit": "%",
        "latex_formula": "$Z = \\mathrm{{ReLU}}[\\mathrm{{Conv1d}}(x)]$ (for Formula entities)",
        "variables": ["x", "Z"] (for Formula entities)
      }},
      "source_traceability": {{
        "text_snippet": "Exact sentence from text",
        "context": "Where this appears",
        "figure_table_ref": "Figure 2a",
        "section_hint": "Results"
      }},
      "confidence": 0.95
    }}
  ]
}}
```

**CRITICAL**: You MUST always return valid JSON format, even if no entities are found. 
If the text contains no extractable entities (e.g., only bibliographic references, citations, or reference lists), return:
```json
{{"entities": []}}
```

Please extract entities with complete traceability, including formulas and LaTeX-formatted information.
"""

# Relation extraction prompt with escaped braces for .format()
RELATION_EXTRACTION_PROMPT = """
You are an expert AI assistant specialized in extracting relationships between entities in optoelectronic research literature.

**YOUR TASK**: 
Extract meaningful scientific relationships between the provided entities based on the text.
**IMPORTANT**: Do NOT extract relationships from reference lists, bibliographies, or citation content.

**ENTITY LIST**:
{entities}

**TEXT CONTENT**:
{text}

**OUTPUT FORMAT**:
```json
{{
  "relations": [
    {{
      "source_entity": "Source entity name",
      "target_entity": "Target entity name",
      "relation_type": "HAS_PROPERTY|USES|FABRICATED_BY|INFLUENCES",
      "properties": {{
        "effect": "increases"
      }},
      "source_traceability": {{
        "text_snippet": "Exact text establishing relationship",
        "evidence_strength": "explicit"
      }},
      "confidence": 0.9
    }}
  ]
}}
```

**CRITICAL**: You MUST always return valid JSON format, even if no relations are found.
If no relationships can be extracted (e.g., text contains only references or citations), return:
```json
{{"relations": []}}
```

Extract relationships with complete traceability.
"""

# System prompts
ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a world-class expert in optoelectronic research."""

RELATION_EXTRACTION_SYSTEM_PROMPT = """You are a leading expert in optoelectronic device physics."""

# Configuration parameters
EXTRACTION_CONFIG = {
    "entity_extraction": {
        "temperature": 0.05,
        "max_tokens": 8000,  # 增加到8000以支持更多实体提取
        "top_p": 0.9,
        "min_confidence": 0.7,
        "min_text_length": 30,
    },
    "relation_extraction": {
        "temperature": 0.05,
        "max_tokens": 8000,  # 增加到8000以支持更多关系提取
        "top_p": 0.9,
        "min_confidence": 0.7,
        "min_entities": 2,
    },
    "traceability": {
        "max_snippet_length": 250,
        "require_snippet": True,
        "evidence_levels": ["explicit", "implicit", "inferred"],
    }
}

# Validation functions
def validate_entity_response(entity_data: dict) -> bool:
    """Validate entity response includes traceability"""
    required_fields = ["name", "type", "source_traceability"]
    if not all(field in entity_data for field in required_fields):
        return False
    traceability = entity_data.get("source_traceability", {})
    if not traceability.get("text_snippet"):
        return False
    return True

def validate_relation_response(relation_data: dict) -> bool:
    """Validate relation response includes traceability"""
    required_fields = ["source_entity", "target_entity", "relation_type", "source_traceability"]
    if not all(field in relation_data for field in required_fields):
        return False
    traceability = relation_data.get("source_traceability", {})
    if not traceability.get("text_snippet") or not traceability.get("evidence_strength"):
        return False
    return True
