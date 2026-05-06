"""
Enhanced Knowledge Graph Extraction Prompts for Optoelectronic Research Literature
=====================================================================================

Enhanced prompts specifically designed for photovoltaic and optoelectronic research,
with improved entity and relation extraction based on domain expertise.

Version: 3.0.0 - Domain-Enhanced Extraction
Date: 2025-11-30
Author: RAGPhoto Team

Key Enhancements:
- Stronger focus on ML methods and algorithms
- Enhanced PCE/performance metrics extraction
- Better material-performance relationships
- Improved experimental conditions tracking
- Research methodology extraction
"""

# Enhanced entity extraction prompt with stronger domain focus
ENTITY_EXTRACTION_PROMPT = """
You are an expert AI assistant specialized in extracting entities from optoelectronic and photovoltaic research literature. 
You have deep expertise in:
- Materials science (organic/inorganic semiconductors, perovskites, polymers)
- Device physics (solar cells, LEDs, photodetectors)
- Machine learning for materials discovery
- Experimental characterization techniques
- Performance optimization and stability

**CRITICAL REQUIREMENT: TRACEABILITY**
For every entity you extract, you MUST include source traceability information.

**YOUR TASK**: 
Extract entities from optoelectronic research text with high precision and complete traceability.

**ENTITY CATEGORIES** (with enhanced photovoltaic focus):

🔬 **Material** - Photovoltaic materials and chemical compounds
   - Examples: MAPbI3, FAPbI3, P3HT, PCBM, PM6, Y6, ITO, Spiro-OMeTAD, PEDOT:PSS
   - **Chemical formulas**: Extract LaTeX format (e.g., $\\mathrm{{MAPbI_3}}$)
   - **Donor/Acceptor materials**: Always classify as Material
   - **Dopants and additives**: Also classify as Material

⚡ **Device** - Photovoltaic devices and device layers
   - Examples: Solar Cell, Tandem Cell, PSC (Perovskite Solar Cell), OSC (Organic Solar Cell)
   - **Device layers**: ETL, HTL, Active Layer, Buffer Layer, Electrode
   - **Device architecture**: Inverted structure, Planar heterojunction, Bulk heterojunction

📊 **Metric** - **CRITICAL**: Performance indicators with numerical values
   - **Core metrics**: PCE (Power Conversion Efficiency), Voc (Open-circuit Voltage), Jsc (Short-circuit Current), FF (Fill Factor)
   - **Stability metrics**: T50, T80, Lifetime, Degradation Rate
   - **Material properties**: Bandgap, HOMO/LUMO, Mobility, Conductivity
   - **LaTeX Numbers**: $1 9 . 2 0 \\%$ means 19.20% (remove spaces)
   - **RULE**: Extract the metric NAME (e.g., "PCE"), store VALUE in properties
   - **Example**: For "PCE of 25.2%", extract entity "PCE" with properties: {{"value": "25.2", "unit": "%"}}

🔧 **Process** - Fabrication and characterization techniques
   - **Fabrication**: Spin Coating, Thermal Evaporation, Solution Processing, Blade Coating, Spray Coating
   - **Post-processing**: Annealing, Solvent Treatment, Passivation, Doping
   - **Characterization**: J-V Measurement, EQE, IPCE, C-V, Impedance Spectroscopy

🤖 **ML_Algorithm** - **NEW ENHANCED**: Machine learning methods and algorithms
   - **Classical ML**: Random Forest, Support Vector Machine (SVM), Decision Tree, K-Nearest Neighbors (KNN)
   - **Ensemble methods**: Gradient Boosting, XGBoost, AdaBoost, Bagging
   - **Deep Learning**: Neural Network, CNN, RNN, LSTM, Transformer, GNN (Graph Neural Network)
   - **Other**: Linear Regression, Logistic Regression, PCA, Clustering
   - **Note**: This is a CRITICAL category for this domain - extract all ML methods mentioned!

🎯 **Parameter** - Experimental conditions and mathematical parameters
   - **Environmental**: Temperature, Humidity, Pressure, Atmosphere (N2, Air)
   - **Optical**: Illumination (AM1.5G, AM0), Light Intensity, Wavelength
   - **Processing**: Annealing Temperature, Annealing Time, Spin Speed, Concentration
   - **Mathematical**: Variables in formulas (e.g., $V_{{OC}}$, $J_{{SC}}$, $R_{{sh}}$)

📐 **Formula** - Mathematical equations and expressions
   - **Inline**: $...$  (e.g., $PCE = \\frac{{V_{{OC}} \\times J_{{SC}} \\times FF}}{{P_{{in}}}}$)
   - **Block**: $$...$$ 
   - **Neural network**: $Z = \\mathrm{{ReLU}}[\\mathrm{{Conv1d}}(x)]$
   - **Properties**: Store LaTeX format, extract variables

🔬 **Measurement** - Characterization and analysis techniques
   - **Structural**: XRD, SEM, TEM, AFM, GIWAXS
   - **Optical**: UV-Vis, PL, EL, Absorption, Reflectance
   - **Electrical**: J-V, EQE, IPCE, C-V, SCLC, TRPL
   - **Chemical**: XPS, FTIR, Raman, NMR, Mass Spectrometry

💡 **Concept** - Scientific concepts and phenomena
   - Examples: Charge Transport, Recombination, Phase Stability, Crystallinity
   - **Mechanisms**: Charge Extraction, Energy Transfer, Exciton Dissociation
   - **Properties**: Optoelectronic Properties, Morphology, Interface Engineering

🎓 **Research_Method** - **NEW**: Research methodologies and approaches
   - Examples: High-Throughput Screening, Computational Screening, Virtual Screening
   - **Workflows**: Automated Synthesis, Robotic Fabrication, Data-Driven Discovery
   - **Design**: Molecular Design, Interface Engineering, Structure Optimization

**ENHANCED EXTRACTION GUIDELINES**:

1. **ML Algorithm Priority**: 
   - ALWAYS extract machine learning methods as "ML_Algorithm" entities
   - Look for keywords: learning, model, neural, network, forest, boosting, regression, classification
   - Examples: "Random Forest model", "neural network prediction", "GNN-based screening"

2. **Performance Metrics Atomization**:
   - For "PCE of 25.2% was achieved", create ONE 'Metric' entity:
     - name: "PCE"
     - properties: {{"value": "25.2", "unit": "%", "achievement_level": "achieved"}}
   - For "improved Voc from 1.0V to 1.2V", create ONE 'Metric' entity:
     - name: "Voc" 
     - properties: {{"initial_value": "1.0", "final_value": "1.2", "unit": "V", "improvement": "increased"}}

3. **Material-Performance Association**:
   - Extract material names with their associated performance
   - Example: "PM6:Y6 blend achieved 16.5% PCE"
     - Material: "PM6:Y6" with properties: {{"type": "donor-acceptor blend"}}
     - Metric: "PCE" with properties: {{"value": "16.5", "unit": "%"}}

4. **Experimental Conditions**:
   - ALWAYS extract illumination conditions (AM1.5G, 1 sun, etc.)
   - Extract processing conditions (annealing temperature, time, atmosphere)
   - Example: "measured under AM1.5G illumination at 100 mW/cm²"
     - Parameter: "AM1.5G" with properties: {{"intensity": "100", "unit": "mW/cm²"}}

5. **Chemical Formulas**:
   - Extract LaTeX chemical formulas as Material entities
   - Store both LaTeX and cleaned versions
   - Example: $\\mathrm{{MAPbI_3}}$ → Material "MAPbI3" with properties: {{"latex": "$\\mathrm{{MAPbI_3}}$"}}

6. **Formula Extraction**:
   - Extract complete mathematical formulas as 'Formula' entities
   - Preserve LaTeX format in properties
   - Extract variables as separate 'Parameter' entities when they represent physical quantities

7. **CRITICAL: EXCLUDE REFERENCES**:
   - Do NOT extract entities from reference lists, bibliographies, or citation content
   - Skip texts with primarily citations, author names with years, DOI links, journal references

**TEXT TO ANALYZE**:
{text}

**OUTPUT FORMAT** (JSON with Traceability):
```json
{{
  "entities": [
    {{
      "name": "Entity name (standardized)",
      "type": "Material|Device|Metric|Process|ML_Algorithm|Parameter|Formula|Measurement|Concept|Research_Method",
      "description": "Brief description (1-2 sentences)",
      "properties": {{
        "value": "25.2 (for metrics)",
        "unit": "% (for metrics)",
        "condition": "AM1.5G (experimental context)",
        "algorithm_type": "ensemble/deep_learning (for ML_Algorithm)",
        "latex_formula": "$...$  (for Formula entities)",
        "variables": ["x", "Z"] (for Formula entities),
        "material_role": "donor/acceptor (for materials)",
        "processing_temp": "150 (for processes)"
      }},
      "source_traceability": {{
        "text_snippet": "Exact sentence from text (max 250 chars)",
        "context": "Surrounding context or section",
        "figure_table_ref": "Figure 2a / Table 1 (if applicable)",
        "section_hint": "Methods/Results/Discussion"
      }},
      "confidence": 0.95
    }}
  ]
}}
```

**CRITICAL**: 
- You MUST always return valid JSON format, even if no entities are found
- If text contains only references/citations, return: {{"entities": []}}
- For ML algorithms, ALWAYS use type "ML_Algorithm" (not Concept or Process)
- For performance metrics with values, ALWAYS use type "Metric" (not Concept)

Please extract entities with complete traceability, paying special attention to ML algorithms and performance metrics.
"""

# Enhanced relation extraction prompt - v3.1 with comprehensive relation extraction
RELATION_EXTRACTION_PROMPT = """
You are an expert AI assistant specialized in extracting relationships between entities in optoelectronic research literature.

**YOUR TASK**: 
Extract **ALL meaningful scientific relationships** between the provided entities based on the text.

**CRITICAL**: 
- **EXTRACT AS MANY RELATIONSHIPS AS POSSIBLE** - Do not limit yourself!
- Look for both explicit and implicit relationships
- Even if a relationship seems minor, extract it if it's scientifically meaningful
- Do NOT extract relationships from reference lists, bibliographies, or citation content
- Focus on scientific relationships that explain mechanisms, performance, or causality

**ENTITY LIST**:
{entities}

**TEXT CONTENT**:
{text}

**COMPREHENSIVE RELATIONSHIP TYPES** (13 types):

🔗 **HAS_PROPERTY** - Material/Device has a Performance Metric
   - Example: (PM6:Y6 blend) -[HAS_PROPERTY]-> (PCE)
   - Properties: {{"value": "16.5%", "condition": "AM1.5G"}}
   - **Extract for**: ANY material/device + metric combination

🔗 **FABRICATED_BY** - Material/Device fabricated by Process
   - Example: (Active Layer) -[FABRICATED_BY]-> (Spin Coating)
   - Properties: {{"condition": "3000 rpm for 30s"}}
   - **Extract for**: ANY fabrication/processing step

🔗 **MEASURES** - Measurement technique measures a Metric
   - Example: (J-V Measurement) -[MEASURES]-> (PCE)
   - Properties: {{"standard": "AM1.5G", "equipment": "Keithley 2400"}}
   - **Extract for**: ALL characterization methods

🔗 **INFLUENCES** - Parameter/Process influences Metric (cause-and-effect)
   - Example: (Annealing Temperature) -[INFLUENCES]-> (PCE)
   - Properties: {{"effect": "increases", "magnitude": "from 15% to 18%", "optimal_value": "150°C"}}
   - **Extract for**: ANY parameter affecting performance

🔗 **PREDICTS** - ML_Algorithm predicts Metric/Property
   - Example: (Random Forest) -[PREDICTS]-> (PCE)
   - Properties: {{"accuracy": "R²=0.92", "dataset_size": "1000 samples"}}
   - **Extract for**: ALL ML prediction tasks

🔗 **OPTIMIZES** - Process/Method optimizes Performance
   - Example: (Solvent Annealing) -[OPTIMIZES]-> (Film Morphology)
   - Properties: {{"improvement": "enhanced crystallinity", "condition": "CHCl3 vapor"}}
   - **Extract for**: ANY optimization relationship

🔗 **IS_A** - Classification or hierarchical relationship
   - Example: (PM6) -[IS_A]-> (Polymer Donor)
   - Properties: {{"chemical_family": "benzodithiophene-based"}}
   - **Extract for**: ALL classification relationships

🔗 **USES** - Device/Process uses Material/Algorithm
   - Example: (Perovskite Solar Cell) -[USES]-> (MAPbI3)
   - Properties: {{"layer": "absorber", "thickness": "300 nm"}}
   - **Extract for**: ALL material usage relationships

🔗 **TRAINED_ON** - ML model trained on Dataset/Features
   - Example: (Neural Network) -[TRAINED_ON]-> (Molecular Descriptors)
   - Properties: {{"feature_count": "200", "training_samples": "5000"}}
   - **Extract for**: ALL ML training relationships

🔗 **IMPROVES** - Method/Material improves another entity
   - Example: (KF Passivation) -[IMPROVES]-> (Device Stability)
   - Properties: {{"improvement": "T80 from 100h to 500h", "mechanism": "defect passivation"}}
   - **Extract for**: ALL improvement relationships

🔗 **COMPOSES** - **NEW**: Entity is composed of or contains other entities
   - Example: (Active Layer) -[COMPOSES]-> (PM6:Y6 blend)
   - Properties: {{"ratio": "1:1.2", "total_thickness": "100 nm"}}
   - **Extract for**: Device structure, material composition

🔗 **ENHANCES** - **NEW**: Entity enhances property/performance of another
   - Example: (Additives) -[ENHANCES]-> (Film Morphology)
   - Properties: {{"mechanism": "improved phase separation", "concentration": "0.5%"}}
   - **Extract for**: Enhancement relationships

🔗 **CORRELATES_WITH** - **NEW**: Strong correlation between two entities
   - Example: (Crystallinity) -[CORRELATES_WITH]-> (PCE)
   - Properties: {{"correlation": "positive", "strength": "strong", "R": "0.85"}}
   - **Extract for**: Scientific correlations

**COMPREHENSIVE EXTRACTION GUIDELINES** (Extract ALL relationships):

1. **Performance Relationships** (HIGHEST PRIORITY):
   - Extract **ALL** Material/Device → HAS_PROPERTY → Metric relationships
   - Include numerical values and conditions
   - Example: "PM6:Y6 achieved 16.5% PCE under AM1.5G"
     - (PM6:Y6) -[HAS_PROPERTY {{value: "16.5%", condition: "AM1.5G"}}]-> (PCE)
   - **Don't miss any**: Extract for EVERY material-metric pair mentioned

2. **ML Prediction Relationships**:
   - Extract **ALL** ML_Algorithm → PREDICTS → Metric relationships
   - Include model performance metrics (accuracy, R², MAE, etc.)
   - Example: "Random Forest predicted PCE with R²=0.92"
     - (Random Forest) -[PREDICTS {{accuracy: "R²=0.92"}}]-> (PCE)
   - Also extract ML_Algorithm → TRAINED_ON → Features/Dataset

3. **Process-Performance Relationships**:
   - Extract **ALL** Process → INFLUENCES → Metric relationships
   - Specify the effect direction and magnitude
   - Example: "Increasing annealing temperature from 100°C to 150°C improved PCE from 15% to 18%"
     - (Annealing Temperature) -[INFLUENCES {{effect: "increases", "from": "15%", "to": "18%", "range": "100-150°C"}}]-> (PCE)
   - Extract for EVERY process-metric interaction

4. **Fabrication Relationships**:
   - Extract **ALL** Material/Device → FABRICATED_BY → Process relationships
   - Example: "Active layer was spin-coated at 3000 rpm"
     - (Active Layer) -[FABRICATED_BY {{condition: "3000 rpm"}}]-> (Spin Coating)

5. **Material Usage Relationships**:
   - Extract **ALL** Device → USES → Material relationships
   - Example: "Solar cell uses MAPbI3 as absorber"
     - (Solar Cell) -[USES {{layer: "absorber"}}]-> (MAPbI3)

6. **Measurement Relationships**:
   - Extract **ALL** Measurement → MEASURES → Metric relationships
   - Example: "J-V measurement was performed"
     - (J-V Measurement) -[MEASURES]-> (PCE), (Voc), (Jsc), (FF)

7. **Experimental Context**:
   - Link **ALL** Parameters to affected Metrics
   - Example: "measured under AM1.5G illumination"
     - (AM1.5G) -[INFLUENCES {{context: "measurement condition"}}]-> (PCE)

8. **Research Methodology**:
   - Extract **ALL** Research_Method relationships
   - Example: "High-throughput screening identified top candidates"
     - (High-throughput Screening) -[OPTIMIZES]-> (Material Selection)

9. **Causality and Mechanisms**:
   - Extract **ALL** causal relationships with mechanisms
   - Create multiple relationships for causal chains
   - Example: "Solvent annealing enhanced morphology, leading to improved PCE"
     - (Solvent Annealing) -[OPTIMIZES {{mechanism: "enhanced morphology"}}]-> (Film Morphology)
     - (Film Morphology) -[INFLUENCES {{effect: "improves"}}]-> (PCE)

10. **Composition and Structure**:
    - Extract **ALL** COMPOSES relationships for device structure
    - Example: "Active layer consists of PM6:Y6 blend"
      - (Active Layer) -[COMPOSES {{ratio: "1:1.2"}}]-> (PM6:Y6 blend)

11. **Enhancement Relationships**:
    - Extract **ALL** ENHANCES relationships
    - Example: "Additive enhances film quality"
      - (Additive) -[ENHANCES {{property: "film quality"}}]-> (Film Morphology)

12. **Correlation Relationships**:
    - Extract **ALL** CORRELATES_WITH for scientific correlations
    - Example: "PCE correlates strongly with crystallinity"
      - (PCE) -[CORRELATES_WITH {{correlation: "positive", strength: "strong"}}]-> (Crystallinity)

13. **Classification Relationships**:
    - Extract **ALL** IS_A relationships for classification
    - Example: "PM6 is a polymer donor"
      - (PM6) -[IS_A {{family: "polymer"}}]-> (Donor Material)

**CRITICAL INSTRUCTIONS**:
- **DO NOT limit the number of relationships** - extract as many as you can find
- **Look for implicit relationships** - even if not explicitly stated
- **Create multiple relationships per entity pair** if multiple relationship types apply
- **Extract relationships between ALL entity pairs** that have meaningful connections
- **Prioritize completeness over brevity** - more relationships are better than fewer

**OUTPUT FORMAT**:
```json
{{
  "relations": [
    {{
      "source_entity": "Source entity name (must match entity list)",
      "target_entity": "Target entity name (must match entity list)",
      "relation_type": "HAS_PROPERTY|FABRICATED_BY|MEASURES|INFLUENCES|PREDICTS|OPTIMIZES|IS_A|USES|TRAINED_ON|IMPROVES|COMPOSES|ENHANCES|CORRELATES_WITH",
      "description": "Brief explanation of the relationship",
      "properties": {{
        "value": "16.5% (for HAS_PROPERTY)",
        "effect": "increases/decreases (for INFLUENCES)",
        "magnitude": "from 15% to 18% (quantitative change)",
        "condition": "AM1.5G, 25°C (experimental context)",
        "accuracy": "R²=0.92 (for PREDICTS)",
        "mechanism": "defect passivation (explanation)",
        "ratio": "1:1.2 (for COMPOSES)",
        "correlation": "positive/negative (for CORRELATES_WITH)",
        "strength": "strong/moderate/weak (for CORRELATES_WITH)"
      }},
      "source_traceability": {{
        "text_snippet": "Exact text establishing this relationship",
        "context": "Additional context",
        "evidence_strength": "explicit|implicit|inferred"
      }},
      "confidence": 0.9
    }}
  ]
}}
```

**CRITICAL EXTRACTION REQUIREMENTS**: 
- You MUST always return valid JSON format, even if no relations are found
- If no relationships can be extracted, return: {{"relations": []}}
- Entity names MUST exactly match those in the entity list
- **EXTRACT AS MANY RELATIONSHIPS AS POSSIBLE** - aim for high coverage
- For ML algorithms, use "PREDICTS" relation type to connect to metrics
- For performance data, ALWAYS include numerical values in properties
- Create **multiple relationships** for the same entity pair if different types apply
- Don't worry about extracting "too many" - completeness is the goal

**EXAMPLES OF COMPREHENSIVE EXTRACTION**:

Example text: "PM6:Y6 blend achieved 16.5% PCE with Voc of 0.84V and Jsc of 25.3 mA/cm² under AM1.5G"

Extract **ALL** of these:
1. (PM6:Y6 blend) -[HAS_PROPERTY {{value: "16.5%", condition: "AM1.5G"}}]-> (PCE)
2. (PM6:Y6 blend) -[HAS_PROPERTY {{value: "0.84V", condition: "AM1.5G"}}]-> (Voc)
3. (PM6:Y6 blend) -[HAS_PROPERTY {{value: "25.3 mA/cm²", condition: "AM1.5G"}}]-> (Jsc)
4. (AM1.5G) -[INFLUENCES {{context: "measurement condition"}}]-> (PCE)
5. (AM1.5G) -[INFLUENCES {{context: "measurement condition"}}]-> (Voc)
6. (AM1.5G) -[INFLUENCES {{context: "measurement condition"}}]-> (Jsc)

Extract relationships with complete traceability and quantitative information.
**Remember: More relationships = Better knowledge graph!**
"""

# Enhanced system prompts
ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a world-class expert in optoelectronic research, materials science, and machine learning for materials discovery. You have extensive knowledge of photovoltaic devices, organic semiconductors, perovskite materials, and computational materials screening methods."""

RELATION_EXTRACTION_SYSTEM_PROMPT = """You are a leading expert in optoelectronic device physics, materials characterization, and data-driven materials discovery. You excel at identifying causal relationships, performance correlations, and mechanism-property relationships in scientific literature."""

# Enhanced configuration parameters - v3.1 with comprehensive relation extraction
EXTRACTION_CONFIG = {
    "entity_extraction": {
        "temperature": 0.05,  # Very low for precise extraction
        "max_tokens": 8000,   # Support more entities
        "top_p": 0.9,
        "min_confidence": 0.7,
        "min_text_length": 30,
        "enhanced_categories": [
            "ML_Algorithm",      # Machine learning methods
            "Research_Method",   # Research methodologies
            "Metric"            # Performance indicators (highest priority)
        ]
    },
    "relation_extraction": {
        "temperature": 0.05,  # Very low for precise extraction
        "max_tokens": 12000,  # 🆕 INCREASED: 8000 → 12000 for more relations
        "top_p": 0.9,
        "min_confidence": 0.65,  # 🆕 LOWERED: 0.7 → 0.65 to capture more relations
        "min_entities": 1,   # 🆕 LOWERED: 2 → 1 to allow single-entity relations
        "extract_all": True,  # 🆕 NEW: Flag to extract all possible relations
        "enhanced_relations": [
            "PREDICTS",          # ML predictions
            "OPTIMIZES",         # Optimization relationships
            "TRAINED_ON",        # Training data relationships
            "INFLUENCES",        # Causal relationships
            "COMPOSES",          # 🆕 NEW: Composition relationships
            "ENHANCES",          # 🆕 NEW: Enhancement relationships
            "CORRELATES_WITH"    # 🆕 NEW: Correlation relationships
        ],
        "comprehensive_types": [  # 🆕 NEW: All 13 relation types
            "HAS_PROPERTY",      # Material/Device properties
            "FABRICATED_BY",     # Fabrication processes
            "MEASURES",          # Measurement techniques
            "INFLUENCES",        # Parameter influences
            "PREDICTS",          # ML predictions
            "OPTIMIZES",         # Process optimization
            "IS_A",              # Classification
            "USES",              # Material usage
            "TRAINED_ON",        # ML training
            "IMPROVES",          # Improvements
            "COMPOSES",          # Composition
            "ENHANCES",          # Enhancement
            "CORRELATES_WITH"    # Correlation
        ]
    },
    "traceability": {
        "max_snippet_length": 250,
        "require_snippet": True,
        "evidence_levels": ["explicit", "implicit", "inferred"],
    },
    "domain_keywords": {
        "ml_algorithms": [
            "random forest", "neural network", "deep learning", "gradient boosting",
            "support vector", "svm", "k-nearest", "knn", "decision tree",
            "regression", "classification", "cnn", "rnn", "lstm", "transformer",
            "gnn", "graph neural", "xgboost", "adaboost", "ensemble",
            "linear regression", "logistic regression", "pca", "clustering"
        ],
        "performance_metrics": [
            "pce", "power conversion efficiency", "voc", "open-circuit voltage",
            "jsc", "short-circuit current", "ff", "fill factor",
            "bandgap", "homo", "lumo", "mobility", "conductivity",
            "t50", "t80", "lifetime", "degradation", "stability"
        ],
        "materials": [
            "perovskite", "mapbi3", "fapbi3", "cspbi3", "silicon", "cdte", "cigs",
            "p3ht", "pcbm", "ptb7", "pm6", "y6", "itic", "pbdb-t",
            "ito", "fto", "spiro-ometad", "pedot:pss", "tio2", "sno2"
        ],
        "research_methods": [
            "high-throughput", "virtual screening", "computational screening",
            "automated synthesis", "data-driven", "machine learning-guided"
        ]
    }
}

# Enhanced validation functions
def validate_entity_response(entity_data: dict) -> bool:
    """Validate entity response includes traceability"""
    required_fields = ["name", "type", "source_traceability"]
    if not all(field in entity_data for field in required_fields):
        return False
    
    traceability = entity_data.get("source_traceability", {})
    if not traceability.get("text_snippet"):
        return False
    
    # Enhanced validation for ML algorithms and metrics
    entity_type = entity_data.get("type", "")
    if entity_type == "ML_Algorithm":
        # ML algorithms should have algorithm_type in properties
        properties = entity_data.get("properties", {})
        if not properties:
            return False
    
    if entity_type == "Metric":
        # Metrics should ideally have value and unit
        properties = entity_data.get("properties", {})
        # This is lenient - metrics without values are still valid (e.g., "efficiency" as a concept)
    
    return True

def validate_relation_response(relation_data: dict) -> bool:
    """Validate relation response includes traceability"""
    required_fields = ["source_entity", "target_entity", "relation_type", "source_traceability"]
    if not all(field in relation_data for field in required_fields):
        return False
    
    traceability = relation_data.get("source_traceability", {})
    if not traceability.get("text_snippet") or not traceability.get("evidence_strength"):
        return False
    
    # Enhanced validation for new relation types
    relation_type = relation_data.get("relation_type", "")
    properties = relation_data.get("properties", {})
    
    # PREDICTS relations should have accuracy metrics
    if relation_type == "PREDICTS" and properties:
        if not any(key in properties for key in ["accuracy", "r2", "mae", "rmse"]):
            # Lenient - predictions without accuracy are still valid
            pass
    
    # INFLUENCES relations should have effect direction
    if relation_type == "INFLUENCES" and properties:
        if not any(key in properties for key in ["effect", "impact", "change"]):
            # Lenient
            pass
    
    return True

def is_ml_algorithm_keyword(text: str) -> bool:
    """Check if text contains ML algorithm keywords"""
    text_lower = text.lower()
    for keyword in EXTRACTION_CONFIG["domain_keywords"]["ml_algorithms"]:
        if keyword in text_lower:
            return True
    return False

def is_performance_metric_keyword(text: str) -> bool:
    """Check if text contains performance metric keywords"""
    text_lower = text.lower()
    for keyword in EXTRACTION_CONFIG["domain_keywords"]["performance_metrics"]:
        if keyword in text_lower:
            return True
    return False

def extract_numerical_values(text: str) -> list:
    """Extract numerical values from text (for metrics)"""
    import re
    # Match patterns like: 25.2%, 1.5 eV, 10 mA/cm², etc.
    patterns = [
        r'(\d+\.?\d*)\s*%',           # Percentages
        r'(\d+\.?\d*)\s*eV',          # Electron volts
        r'(\d+\.?\d*)\s*V',           # Volts
        r'(\d+\.?\d*)\s*mA/cm²',      # Current density
        r'(\d+\.?\d*)\s*cm²/Vs',      # Mobility
    ]
    
    values = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        values.extend(matches)
    
    return values
