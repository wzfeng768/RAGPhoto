"""
Chunk Metadata Enricher
Adds semantic content type tags to document chunks for better retrieval
"""

import re
from typing import Dict, List, Any


class ChunkMetadataEnricher:
    """Enrich chunks with semantic content type tags"""
    
    def __init__(self):
        """Initialize the metadata enricher with pattern matchers"""
        # Numerical value patterns
        self.numerical_patterns = [
            r'\d+\.?\d*\s*%',          # Percentages: 98%, 25.5%
            r'\d+\.?\d*\s*eV',         # Energy values: 1.5 eV
            r'\d+\.?\d*\s*°[CF]',      # Temperatures: 25°C, 450°F
            r'\d+\.?\d*\s*nm',         # Wavelengths: 532 nm
            r'\d+\.?\d*\s*[mµ]m',      # Lengths: 5 mm, 10 μm
            r'\d+\.?\d*\s*[VvMm][Aa]', # Voltages/Current: 1.2 V, 20 mA
            r'\d+\.?\d*\s*W',          # Power: 100 W
            r'\d+\.?\d*\s*[Ωω]',       # Resistance: 10 Ω
            r'\d+\.?\d*\s*cm[-−]?[²2]', # Areas: 1 cm²
        ]
        
        # Method/process keywords
        self.method_keywords = [
            'fabricat', 'prepar', 'synthes', 'deposit', 'coat',
            'anneal', 'spin', 'evaporat', 'sputter', 'diffusion',
            'method', 'process', 'procedure', 'technique', 'approach',
            'protocol', 'recipe'
        ]
        
        # Comparison keywords
        self.comparison_keywords = [
            'compar', 'versus', 'vs', 'differ', 'higher', 'lower',
            'better', 'worse', 'superior', 'inferior', 'outperform',
            'exceed', 'relative to', 'contrast'
        ]
        
        # Key performance metrics
        self.metric_keywords = {
            'PCE': r'\bPCE\b',
            'Voc': r'\bV[_\s]?oc\b',
            'Jsc': r'\bJ[_\s]?sc\b',
            'FF': r'\bFF\b|\bfill\s+factor\b',
            'efficiency': r'\befficiency\b',
            'PLQY': r'\bPLQY\b',
            'mobility': r'\bmobility\b',
            'stability': r'\bstability\b',
            'lifetime': r'\blifetime\b',
            'degradation': r'\bdegradation\b',
            'bandgap': r'\bbandgap\b|\bband\s+gap\b',
        }
    
    def enrich_chunk(self, chunk_content: str) -> Dict[str, Any]:
        """
        Add semantic tags to chunk metadata
        
        Args:
            chunk_content: Text content of the chunk
            
        Returns:
            Dictionary with semantic metadata tags
        """
        metadata = {
            'has_numerical_data': self._contains_numerical_data(chunk_content),
            'has_methods': self._contains_methods(chunk_content),
            'has_comparisons': self._contains_comparisons(chunk_content),
            'numerical_values': self._extract_all_numbers(chunk_content),
            'key_metrics': self._extract_metrics(chunk_content),
            'technical_terms': self._extract_technical_terms(chunk_content),
            'content_density': self._compute_content_density(chunk_content)
        }
        
        return metadata
    
    def _contains_numerical_data(self, text: str) -> bool:
        """Check if chunk contains numerical values"""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.numerical_patterns)
    
    def _contains_methods(self, text: str) -> bool:
        """Check if chunk describes methods or processes"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.method_keywords)
    
    def _contains_comparisons(self, text: str) -> bool:
        """Check if chunk contains comparison statements"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.comparison_keywords)
    
    def _extract_all_numbers(self, text: str) -> List[str]:
        """Extract all numerical values with units"""
        numbers = []
        for pattern in self.numerical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        # Deduplicate while preserving order
        seen = set()
        unique_numbers = []
        for num in numbers:
            if num not in seen:
                seen.add(num)
                unique_numbers.append(num)
        
        return unique_numbers[:20]  # Limit to 20 to avoid metadata bloat
    
    def _extract_metrics(self, text: str) -> List[str]:
        """Extract key performance metrics mentioned in the chunk"""
        metrics_found = []
        
        for metric_name, pattern in self.metric_keywords.items():
            if re.search(pattern, text, re.IGNORECASE):
                metrics_found.append(metric_name)
        
        return metrics_found
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terminology (simplified version)"""
        # Common photovoltaic material patterns
        material_patterns = [
            r'\b[A-Z][a-z]?[A-Z][a-z]*(?:I[₃₂]|Br[₃₂]|Cl[₃₂])?',  # Chemical formulas
            r'\bperovskite\b', r'\bsilicon\b', r'\bCIGS\b', r'\bCdTe\b',
            r'\bP3HT\b', r'\bPCBM\b', r'\bSpiro\b', r'\bPTAA\b'
        ]
        
        terms = []
        for pattern in material_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend([m for m in matches if len(m) > 2])
        
        # Deduplicate
        return list(set(terms))[:10]  # Limit to 10
    
    def _compute_content_density(self, text: str) -> str:
        """Estimate content density (high/medium/low)"""
        # Count information markers
        numerical_count = sum(1 for pattern in self.numerical_patterns 
                            if re.search(pattern, text, re.IGNORECASE))
        
        # Compute density score
        text_length = len(text)
        if text_length == 0:
            return 'low'
        
        density_score = numerical_count / (text_length / 1000)  # per 1000 chars
        
        if density_score > 5:
            return 'high'
        elif density_score > 2:
            return 'medium'
        else:
            return 'low'


def test_enricher():
    """Test the metadata enricher"""
    enricher = ChunkMetadataEnricher()
    
    # Test text with numerical data
    test_text = """
    TPA-An and TBA-An single crystals exhibit exceptionally high photoluminescence 
    quantum yields (PLQYs) of 98% and 99%, respectively, at room temperature. 
    The devices were fabricated using a slow solvent vapor-diffusion method. 
    The PCE reached 25.2% under AM1.5G illumination with Voc of 1.15 V.
    """
    
    metadata = enricher.enrich_chunk(test_text)
    
    print("Test Metadata Enricher:")
    print(f"  Has numerical data: {metadata['has_numerical_data']}")
    print(f"  Has methods: {metadata['has_methods']}")
    print(f"  Numerical values: {metadata['numerical_values']}")
    print(f"  Key metrics: {metadata['key_metrics']}")
    print(f"  Content density: {metadata['content_density']}")


if __name__ == "__main__":
    test_enricher()

